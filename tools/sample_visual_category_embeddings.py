import argparse
import os
import pathlib
import random
from collections import defaultdict
from typing import TypedDict, cast

import todd
import todd.tasks.object_detection as od
import torch
import tqdm
from todd.datasets import BaseDataset
from todd.datasets.access_layers import PthAccessLayer
from todd.patches.torch import get_world_size
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('model')
    args = parser.parse_args()
    return args


class Batch(TypedDict):
    tensors: torch.Tensor
    bboxes: od.FlattenBBoxesXYWH
    categories: torch.Tensor


class Dataset(BaseDataset[Batch, str, Batch]):

    def __getitem__(self, index: int) -> Batch:
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
        f'work_dirs/oake/{args.dataset}/{args.model}_objects_cuda_train'
    )

    access_layer: PthAccessLayer[Batch] = PthAccessLayer(
        data_root=str(data_root / 'output'),
    )
    dataset = Dataset(access_layer=access_layer)

    cpu = os.cpu_count() or 1
    cpu = max(cpu // get_world_size(), 1)
    dataloader = DataLoader(dataset, None, num_workers=cpu)

    embeddings: defaultdict[int, ReservoirSampler] = \
        defaultdict(ReservoirSampler)

    batch: Batch
    for batch in tqdm.tqdm(dataloader):
        assert (
            batch['tensors'].shape[0] == len(batch['bboxes']) ==
            batch['categories'].shape[0]
        )
        for tensor, category in zip(batch['tensors'], batch['categories']):
            embeddings[category.item()](tensor)

    categories = torch.load(data_root / 'categories.pth', 'cpu')
    return {
        c['name']: torch.stack(reservoir)
        for i, c in enumerate(categories)
        if len(reservoir := embeddings[i].reservoir) > 0
    }


def main() -> None:
    args = parse_args()

    oake_embeddings = oake(args)
    sample_image_embeddings: dict[str, torch.Tensor] = torch.load(
        f'work_dirs/sample_image_embeddings/{args.dataset}.pth',
        'cpu',
    )
    assert set(oake_embeddings).issubset(sample_image_embeddings)

    embeddings = {
        k: torch.cat([v, oake_embeddings[k]]) if k in oake_embeddings else v
        for k, v in sample_image_embeddings.items()
    }

    work_dir = pathlib.Path('work_dirs/visual_category_embeddings')
    if todd.Store.DRY_RUN:
        work_dir = work_dir / 'dry_run'
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, work_dir / f'{args.dataset}_{args.model}.pth')


if __name__ == '__main__':
    main()
