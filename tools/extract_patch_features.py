import itertools
from typing import List, Tuple

import argparse
import sys
import pathlib

import todd
import torch.cuda
import torch.distributed

sys.path.insert(0, '')
import clip
from mldec.debug import debug

from torchvision import datasets as datasets
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.distributed


class PatchedCocoClassification(datasets.coco.CocoDetection):

    def __init__(
        self,
        *args,
        patch_size: int = 224,
        max_stride: int = 112,
        rescale: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
        self._max_stride = max_stride
        self._rescale = rescale

    def _cut(self, length: int) -> List[int]:
        assert length >= self._patch_size

        result = [0]
        if length != self._patch_size:
            n = (length - self._patch_size - 1) // self._max_stride + 1
            q, r = divmod(length - self._patch_size, n)
            for i in range(n):
                result.append(result[-1] + q + (i < r))
        return result

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image_id = self.ids[index]

        image = self._load_image(image_id)

        scale = 1
        bbox = (
            (image.width - min(image.size)) / 2,
            (image.height - min(image.size)) / 2,
            min(image.size),
            scale,
        )

        patches = [image]
        bboxes = [bbox]
        while image.width >= self._patch_size and image.height >= self._patch_size:
            for x, y in itertools.product(*map(self._cut, image.size)):
                patches.append(image.crop((x, y, x + self._patch_size, y + self._patch_size)))
                bboxes.append((x, y, self._patch_size, scale))
            image = image.resize((int(image.width / self._rescale), int(image.height / self._rescale)))
            scale *= self._rescale

        patches_ = torch.stack(list(map(self.transforms.transform, patches)))
        bboxes_ = torch.tensor(bboxes)
        bboxes_[:, :-1] *= bboxes_[:, [-1]]
        bboxes_[:, -1] = bboxes_[:, -2]

        return patches_, bboxes_, image_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', type=pathlib.Path)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = todd.base.Config.load(args.config)

    logger = todd.base.get_logger()

    debug.init(config=config)

    clip_model, clip_transform = clip.load('pretrained/clip/RN50.pt', 'cpu')
    if not debug.CPU:
        clip_model = clip_model.cuda()

    dataset = PatchedCocoClassification(transform=clip_transform, **config.train.dataset)
    work_dir = pathlib.Path(config.train.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        patches, bboxes, image_id = dataset[i]
        if not debug.CPU:
            patches = patches.cuda()
        patch_features = clip_model.encode_image(patches)
        patch_features = F.normalize(patch_features)
        torch.save(dict(patches=patch_features, bboxes=bboxes), work_dir / f'{image_id:012d}.pth')
        if i % config.log_interval == 0:
            logger.info(f"Train [{i}/{len(dataset)}]")
        if debug.LESS_DATA and i: break

    dataset = PatchedCocoClassification(transform=clip_transform, **config.val.dataset)
    work_dir = pathlib.Path(config.val.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        patches, bboxes, image_id = dataset[i]
        if not debug.CPU:
            patches = patches.cuda()
        patch_features = clip_model.encode_image(patches)
        patch_features = F.normalize(patch_features)
        torch.save(dict(patches=patch_features, bboxes=bboxes), work_dir / f'{image_id:012d}.pth')
        if i % config.log_interval == 0:
            logger.info(f"Val [{i}/{len(dataset)}]")
        if debug.LESS_DATA and i: break


if __name__ == '__main__':
    main()
