import argparse
import functools
import os
import sys
import pathlib
from typing import List, Tuple

import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as tf
from PIL import Image

import todd

sys.path.insert(0, '')
import clip
import clip.model
from mldec.datasets import mAP, CocoDetection
import mldec.coop
from mldec.debug import debug


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
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--cfg-options', action=todd.base.DictAction)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = todd.base.Config.load(args.config)
    if args.odps is not None:
        odps_init(args.odps)
    debug.init(config=cfg)
    if args.cfg_options is not None:
        for k, v in args.cfg_options.items():
            todd.base.setattr_recur(cfg, k, v)

    if not debug.CPU:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.base.get_local_rank())

    model, transform = clip.load('pretrained/clip/RN50.pt', 'cpu')

    val_dataset = CocoDetection(transform=transform, **cfg.val)
    val_sampler = (
        None if debug.CPU else
        torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.workers, collate_fn=val_dataset.collate,
        )

    classnames = [cat['name'] for cat in val_dataset.coco.cats.values()]
    model = mldec.coop.CustomCLIP(clip_model=model, classnames=classnames, frozen_config=todd.base.Config())
    model.requires_grad_(False)
    if not debug.CPU:
        model = model.cuda()
    state_dict = torch.load('pretrained/mldec_prompt.pth.bak')
    model.load_state_dict(state_dict, strict=False)

    preds = []
    targets = []
    ids = []
    for i, (input, target, num_patches, ids_) in enumerate(val_loader):
        if not debug.CPU:
            input = input.cuda()
            target = target.cuda()
            ids_ = ids_.cuda()
        pred = model(input)
        preds.extend(p.max(0, keepdim=True).values for p in pred[0].split(num_patches.tolist()))
        targets.extend(t[[0]] for t in target.split(num_patches.tolist()))
        ids.append(ids_)
        if i % cfg.log_interval == 0:
            print(f'Val Step [{i}/{len(val_loader)}]')
            if debug.LESS_DATA and i: break
    if not debug.CPU:
        preds_ = torch.cat(preds)
        targets_ = torch.cat(targets)
        ids_ = torch.cat(ids)
        preds = [torch.zeros_like(preds_) for _ in range(todd.base.get_world_size())]
        targets = [torch.zeros_like(targets_) for _ in range(todd.base.get_world_size())]
        ids = [torch.zeros_like(ids_) for _ in range(todd.base.get_world_size())]
        torch.distributed.all_gather(preds, preds_)
        torch.distributed.all_gather(targets, targets_)
        torch.distributed.all_gather(ids, ids_)

    if todd.base.get_rank() == 0:
        preds_ = torch.cat(preds).argsort(descending=True)
        targets_ = torch.cat(targets).gather(-1, preds_)
        ids_ = torch.cat(ids)
        with open('index.html', 'w') as f:
            for i in range(preds_.shape[0]):
                f.write(f'data/coco/val2017/{ids_[i]:012d}.jpg\n<br/>\n')
                for j in range(preds_.shape[1]):
                    if targets_[i][j]:
                        f.write('<div style="color:red;display:inline-block">')
                    f.write(classnames[preds_[i][j]])
                    if targets_[i][j]:
                        f.write('</div>\n')
                    f.write(',\n')
                f.write(f'<img src="data/coco/val2017/{ids_[i]:012d}.jpg">\n<br/>\n')


if __name__ == '__main__':
    main()
