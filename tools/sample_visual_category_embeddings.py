import argparse
import os
import pathlib
import random
from collections import defaultdict
from typing import TypedDict

import todd.tasks.object_detection as od
import torch
from todd.datasets import BaseDataset
from todd.datasets.access_layers import PthAccessLayer
from todd.patches.torch import get_world_size
from torch.utils.data import DataLoader
import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    return args


class Batch(TypedDict):
    tensors: torch.Tensor
    bboxes: od.FlattenBBoxesXYWH
    categories: torch.Tensor


class Dataset(BaseDataset[Batch | None, str, Batch | None]):

    def __getitem__(self, index: int) -> Batch | None:
        _, batch = super()._access(index)
        return batch


class ReservoirSampler:

    def __init__(self, size: int = 1000) -> None:
        self._size = size
        self._reservoir: list[torch.Tensor] = []
        self._index = size

    @property
    def reservoir(self) -> list[torch.Tensor]:
        return self._reservoir

    def __call__(self, tensor: torch.Tensor) -> None:
        if len(self._reservoir) < self._size:
            self._reservoir.append(tensor)
            return
        self._index += 1
        i = random.randrange(self._index)
        if i < self._size:
            self._reservoir[i] = tensor


def oake(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    data_root = pathlib.Path(
        f'work_dirs/oake/{args.dataset}_objects_cuda_train'
    )

    access_layer: PthAccessLayer[Batch] = PthAccessLayer(
        data_root=str(data_root / 'output'),
    )
    dataset = Dataset(access_layer=access_layer)

    cpu = os.cpu_count() or 1
    cpu = max(cpu // get_world_size(), 1)
    dataloader = DataLoader(dataset, None, num_workers=cpu)

    embeddings = defaultdict(ReservoirSampler)

    batch: Batch | None
    for batch in tqdm.tqdm(dataloader):
        if batch is None:
            continue
        assert (
            batch['tensors'].shape[0] == len(batch['bboxes']) ==
            batch['categories'].shape[0]
        )
        for i in range(batch['tensors'].shape[0]):
            tensor = batch['tensors'][i]
            category = batch['categories'][i].item()
            embeddings[category](tensor)

    categories = torch.load(data_root / 'categories.pth', 'cpu')
    return {
        c['name']: torch.stack(embeddings[i].reservoir)
        for i, c in enumerate(categories)
    }


def main() -> None:
    args = parse_args()

    oake_embeddings = oake(args)
    sample_image_embeddings: dict[str, torch.Tensor] = torch.load(
        f'work_dirs/sample_image_embeddings/{args.dataset}.pth',
        'cpu',
    )
    assert set(sample_image_embeddings).issubset(oake_embeddings)

    embeddings = {
        k: torch.cat([v, oake_embeddings[k]])
        for k, v in sample_image_embeddings.items()
    }

    work_dir = pathlib.Path('work_dirs/visual_category_embeddings')
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, work_dir / f'{args.dataset}.pth')


if __name__ == '__main__':
    main()
