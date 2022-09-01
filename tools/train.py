import argparse
import os
import sys
import pathlib
import time

import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.optim.lr_scheduler
from randaugment import RandAugment
import torchvision.transforms as tf
from PIL import Image

import todd

sys.path.insert(0, '')
import clip
import clip.model
from mldec.helper_functions import mAP, CocoDetection, CutoutPIL, add_weight_decay
from mldec.losses import AsymmetricLoss
from mldec.debug import debug
import mldec.coop as coop


class Convert:

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


def odps_init(kwargs: todd.base.Config) -> None:
    logger = todd.base.get_logger()
    logger.debug("ODPS initializing.")
    kwargs.setdefault('LOCAL_RANK', '0')
    os.environ.update(kwargs)
    if not os.path.lexists('data'):
        os.symlink('/data/oss_bucket_0', 'data')
    if not os.path.lexists('pretrained'):
        os.symlink('/data/oss_bucket_0/ckpts', 'pretrained')
    if not os.path.lexists('work_dirs'):
        os.symlink('/data/oss_bucket_0/work_dirs', 'work_dirs')
    logger.debug(f"ODPS initialization done with {os.listdir('.')}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('job_name', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--cfg-options', action=todd.base.DictAction)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = todd.base.Config.load(args.config)
    work_dir: pathlib.Path = 'work_dirs' / args.job_name
    if args.odps is not None:
        odps_init(args.odps)
    debug.init(config=cfg)
    if args.cfg_options is not None:
        for k, v in args.cfg_options.items():
            todd.base.setattr_recur(cfg, k, v)

    work_dir.mkdir(parents=True, exist_ok=True)

    if not debug.CPU:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.base.get_local_rank())

    if todd.base.get_rank() == 0:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = work_dir / f'{timestamp}.log'
        todd.base.init_log_file(log_file)
    logger = todd.base.get_logger()
    logger.info(f"Version: {todd.base.git_commit_id()}")

    train_pipe = tf.Compose([
        tf.Resize(cfg.image_size, interpolation=tf.InterpolationMode.BICUBIC),
        tf.CenterCrop(cfg.image_size),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        Convert(),
        tf.ToTensor(),
        tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    train_dataset = CocoDetection(transform=train_pipe, **cfg.train)
    train_sampler = (
        None if debug.CPU else
        torch.utils.data.distributed.DistributedSampler(train_dataset)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.workers,
        )

    val_pipe = tf.Compose([
        tf.Resize(cfg.image_size, interpolation=tf.InterpolationMode.BICUBIC),
        tf.CenterCrop(cfg.image_size),
        Convert(),
        tf.ToTensor(),
        tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    val_dataset = CocoDetection(transform=val_pipe, **cfg.val)
    val_sampler = (
        None if debug.CPU else
        torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.workers,
        )

    model, _ = clip.load('pretrained/clip/RN50.pt', 'cpu')
    model = coop.CustomCLIP(model, [cat['name'] for cat in train_loader.dataset.coco.cats.values()])
    model.float()
    model.requires_grad_()
    if not debug.CPU:
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
        )

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, cfg.weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=cfg.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, steps_per_epoch=len(train_loader), epochs=cfg.epoch, pct_start=0.2,
    )

    highest_mAP = 0
    for epoch in range(cfg.epoch):
        if not debug.CPU:
            torch.distributed.barrier()
            train_sampler.set_epoch(epoch)
        model.train()
        for i, (inputData, target) in enumerate(train_loader):
            if not debug.CPU:
                inputData = inputData.cuda()
                target = target.cuda()
            target = target.max(dim=1)[0]
            output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            loss.backward()

            optimizer.step()

            scheduler.step()

            if i % cfg.print_freq == 0:
                logger.info(
                        f'Epoch [{epoch}/{cfg.epoch}] '
                        f'Train Step [{i}/{len(train_loader)}] '
                        f'LR {scheduler.get_last_lr()[0]:.2e} '
                        f'Loss {loss.item():.1f}'
                    )
                if not debug.CPU:
                    logger.debug(
                        f'Scaler {model.module._scaler.item():.4f} '
                        f'Bias {model.module._bias.item():.4f}'
                    )
                if debug.LESS_DATA: break

        if todd.base.get_rank() == 0:
            todd.base.save_checkpoint(
                model, work_dir / f'model-{epoch+1}.ckpt',
                optimizer=optimizer, scheduler=scheduler,
            )

        model.eval()

        preds = []
        targets = []
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                if not debug.CPU:
                    input = input.cuda()
                    target = target.cuda()
                pred = model(input).sigmoid()
                target = target.max(dim=1)[0]
                preds.append(pred)
                targets.append(target)
                if i % cfg.print_freq == 0:
                    logger.info(
                        f'Epoch [{epoch}/{cfg.epoch}] '
                        f'Val Step [{i}/{len(val_loader)}]'
                    )
                    if debug.LESS_DATA: break
        if not debug.CPU:
            preds_ = torch.cat(preds)
            targets_ = torch.cat(targets)
            preds = [torch.zeros_like(preds_) for _ in range(todd.base.get_world_size())]
            targets = [torch.zeros_like(targets_) for _ in range(todd.base.get_world_size())]
            torch.distributed.all_gather(preds, preds_)
            torch.distributed.all_gather(targets, targets_)

        if todd.base.get_rank() == 0:
            preds_ = torch.cat(preds)
            targets_ = torch.cat(targets)
            mAP_score = mAP(targets_.cpu().numpy(), preds_.cpu().numpy())
            if mAP_score > highest_mAP:
                highest_mAP = mAP_score
            logger.info(f'current_mAP = {mAP_score:.2f}, highest_mAP = {highest_mAP:.2f}')


if __name__ == '__main__':
    main()
