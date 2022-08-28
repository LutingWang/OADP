import argparse
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
from src_files.helper_functions import mAP, CocoDetection, CutoutPIL, \
    add_weight_decay
from src_files.losses import AsymmetricLoss
from randaugment import RandAugment
import clip
import clip.model
import torchvision.transforms as tf

import todd
import coop


class Debug:
    TRAIN_WITH_VAL_DATASET = todd.base.DebugMode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('work_dir', type=pathlib.Path)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = todd.base.Config.load(args.config)
    work_dir: pathlib.Path = 'work_dirs' / args.work_dir

    if Debug.TRAIN_WITH_VAL_DATASET:
        cfg.train = cfg.val
    work_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(todd.base.get_rank())
    torch.distributed.init_process_group(backend='nccl')

    if todd.base.get_rank() == 0:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = work_dir / f'{timestamp}.log'
        todd.base._extensions.logging.init_log_file(log_file)  # TODO: fix this in the next version of todd
    logger = todd.base.get_logger()

    train_pipe = tf.Compose([
        tf.Resize(cfg.image_size, interpolation=tf.InterpolationMode.BICUBIC),
        tf.CenterCrop(cfg.image_size),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        lambda image: image.convert("RGB"),
        tf.ToTensor(),
        tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    train_dataset = CocoDetection(transform=train_pipe, **cfg.train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.workers,
        )

    val_pipe = tf.Compose([
        tf.Resize(cfg.image_size, interpolation=tf.InterpolationMode.BICUBIC),
        tf.CenterCrop(cfg.image_size),
        lambda image: image.convert("RGB"),
        tf.ToTensor(),
        tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    val_dataset = CocoDetection(transform=val_pipe, **cfg.val)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.workers,
        )

    model, _ = clip.load('RN50', 'cpu')
    model = coop.CustomCLIP(model, [cat['name'] for cat in train_loader.dataset.coco.cats.values()])
    model.float()
    model.requires_grad_()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, cfg.weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=cfg.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, steps_per_epoch=len(train_loader), epochs=cfg.epoch, pct_start=0.2,
    )

    highest_mAP = 0
    for epoch in range(cfg.epoch):
        torch.distributed.barrier()
        train_sampler.set_epoch(epoch)
        model.train()
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda().max(dim=1)[0]
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
                logger.debug(
                        f'Scaler {model.module._scaler.item():.4f} '
                        f'Bias {model.module._bias.item():.4f}'
                    )
                break

        if todd.base.get_rank() == 0:
            torch.save(model.state_dict(), work_dir / f'model-{epoch+1}.ckpt')

        model.eval()

        preds = []
        targets = []
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                pred = model(input.cuda()).sigmoid()
                preds.append(pred)
                target = target.cuda().max(dim=1)[0]
                targets.append(target)
                if i % cfg.print_freq == 0:
                    logger.info(
                        f'Epoch [{epoch}/{cfg.epoch}] '
                        f'Val Step [{i}/{len(val_loader)}]'
                    )
                    break
        preds_ = torch.cat(preds)
        targets_ = torch.cat(targets)
        all_preds = [torch.zeros_like(preds_) for _ in range(todd.base.get_world_size())] if todd.base.get_rank() == 0 else None
        all_targets = [torch.zeros_like(targets_) for _ in range(todd.base.get_world_size())] if todd.base.get_rank() == 0 else None
        torch.distributed.gather(preds_, all_preds)
        torch.distributed.gather(targets_, all_targets)

        if todd.base.get_rank() == 0:
            all_preds_ = torch.cat(all_preds)
            all_targets_ = torch.cat(all_targets)
            mAP_score = mAP(all_targets_.cpu().numpy(), all_preds_.cpu().numpy())
            if mAP_score > highest_mAP:
                highest_mAP = mAP_score
            logger.info(f'current_mAP = {mAP_score:.2f}, highest_mAP = {highest_mAP:.2f}')


if __name__ == '__main__':
    main()
