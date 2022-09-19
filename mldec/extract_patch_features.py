from collections import namedtuple
import itertools
from typing import List, Literal, Tuple

import argparse
import sys
import pathlib

import todd
import torch
import torch.cuda
import torch.nn.functional as F
import torch.utils.data
import torchvision

import clip
import clip.model

from .debug import debug

Batch = namedtuple('Batch', ['patches', 'bboxes', 'image_ids', 'num_patches'])


class PatchedCocoClassification(torchvision.datasets.coco.CocoDetection):

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

    @staticmethod
    def collate(
        batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
    ) -> Batch:
        patches_list, bboxes, image_ids = zip(*batch)
        patches = torch.cat(patches_list)
        num_patches = [p.shape[0] for p in patches_list]
        return Batch(patches, bboxes, image_ids, num_patches)


class Runner:

    def __init__(self, config: todd.base.Config) -> None:
        self._config = config
        self._logger = todd.base.get_logger()

        if debug.CPU:
            self._model, self._transform = clip.load('pretrained/clip/RN50.pt', 'cpu')
        else:
            self._model, self._transform = clip.load('pretrained/clip/RN50.pt')
        self._model.requires_grad_(False)

    def _run_iter(self, batch: Batch, patches_root: pathlib.Path) -> None:
        if not debug.CPU:
            batch = Batch(
                batch.patches.cuda(),
                batch.bboxes,
                batch.image_ids,
                batch.num_patches,
            )
        patch_features = self._model.encode_image(batch.patches)
        patch_features = F.normalize(patch_features)
        for patch_features_, bboxes_, image_id in zip(
            patch_features.split(batch.num_patches),
            batch.bboxes,
            batch.image_ids,
        ):
            torch.save(
                dict(patches=patch_features_.clone(), bboxes=bboxes_),
                patches_root / f'{image_id:012d}.pth',
            )

    def _run(self, mode: Literal['train', 'val']) -> None:
        config = self._config[mode]
        dataset = PatchedCocoClassification(transform=self._transform, **config.dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=dataset.collate,
        )
        patches_root = pathlib.Path(config.patches_root)
        patches_root.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(dataloader):
            self._run_iter(batch, patches_root)
            if i % self._config.log_interval == 0:
                self._logger.info(f"{mode.capitalize()} [{i * dataloader.batch_size}/{len(dataset)}]")
                if debug.LESS_DATA and i: break

    def train(self) -> None:
        self._run('train')

    def val(self) -> None:
        self._run('val')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', type=pathlib.Path)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = todd.base.Config.load(args.config)
    debug.init(config=config)

    runner = Runner(config)
    runner.train()
    runner.val()
