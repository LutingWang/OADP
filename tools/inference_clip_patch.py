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
from mldec.helper_functions import mAP, CocoDetection
from mldec.debug import debug


class Transform:

    def __init__(self, transform: tf.Compose, max_stride: int = 122) -> None:
        self._transform = transform
        self._max_stride = max_stride
        self._patch_size = 224

    def _cut(self, length: int) -> List[int]:
        assert length >= self._patch_size
        if length == self._patch_size:
            return [0]
        n = (length - self._patch_size - 1) // self._max_stride + 1
        stride = (length - self._patch_size) // n
        mod = (length - self._patch_size) % n

        result = [0]
        for i in range(n):
            result.append(result[-1] + stride + (i < mod))
        return result

    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        if image.width < self._patch_size or image.height < self._patch_size:
            return [self._transform(image)]
        patches = []
        for x in self._cut(image.width):
            for y in self._cut(image.height):
                patch = image.crop((x, y, x + self._patch_size, y + self._patch_size))
                patches.append(self._transform(patch))
        return patches

    @staticmethod
    def collate(batch: List[Tuple[List[torch.Tensor], torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches, targets = zip(*batch)
        num_patches = list(map(len, patches))
        patches = sum(patches, [])
        return tuple(map(torch.utils.data.default_collate, (patches, num_patches, targets)))


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
    model.requires_grad_(False)
    if not debug.CPU:
        model = model.cuda()

    val_dataset = CocoDetection(transform=Transform(transform), **cfg.val)
    val_sampler = (
        None if debug.CPU else
        torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.workers, collate_fn=Transform.collate,
        )

    preds = []
    targets = []
    classnames = [f'a photo of a {cat["name"]}' for cat in val_dataset.coco.cats.values()]
    tokens = clip.tokenize(classnames, 77)
    if not debug.CPU:
        tokens = tokens.cuda()
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    for i, (input, num_patches, target) in enumerate(val_loader):
        if not debug.CPU:
            input = input.cuda()
            target = target.cuda()
        image_features, _ = model.encode_image(input)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        pred = image_features @ text_features.t()
        preds.extend(p.max(0, keepdim=True).values for p in pred.split(num_patches.tolist()))
        targets.append(target)
        if i % cfg.print_freq == 0:
            print(f'Val Step [{i}/{len(val_loader)}]')
            if debug.LESS_DATA and i: break
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
        print(f'mAP = {mAP_score:.2f}')


if __name__ == '__main__':
    main()
