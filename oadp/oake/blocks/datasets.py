__all__ = [
    'BlockDatasetMixin',
]

import itertools
from typing import TypedDict, TypeVar

import todd.tasks.object_detection as od
import torch
import torch.cuda
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image

from ..datasets import BaseDataset
from ..registries import OAKEDatasetRegistry

BBoxes = od.FlattenBBoxesXYXY
Blocks = list[Image.Image]


class Batch(TypedDict):
    id_: str
    bboxes: BBoxes
    blocks: torch.Tensor


class Partitioner:

    def __init__(self, block_size: int = 224, max_stride: int = 112) -> None:
        self._block_size = block_size
        self._max_stride = max_stride

    def partition_1d(self, length: int) -> list[int]:
        if length < self._block_size:
            return []

        result = [0]
        if length == self._block_size:
            return result

        n = (length - self._block_size - 1) // self._max_stride + 1
        q, r = divmod(length - self._block_size, n)
        for i in range(n):
            result.append(result[-1] + q + (i < r))
        return result

    def partition_2d(self, w: int, h: int) -> list[tuple[int, int]]:
        partition_w = self.partition_1d(w)
        partition_h = self.partition_1d(h)
        return list(itertools.product(partition_w, partition_h))

    def partition_image(
        self,
        image: Image.Image,
    ) -> tuple[BBoxes, Blocks] | None:
        w, h = image.size
        partitions = self.partition_2d(w, h)
        if len(partitions) == 0:
            return None
        bboxes = [(x, y, self._block_size, self._block_size)
                  for x, y in partitions]
        bboxes = od.FlattenBBoxesXYWH(torch.tensor(bboxes)).to(BBoxes)
        blocks = [image.crop(bbox) for bbox in bboxes]
        return bboxes, blocks


@OAKEDatasetRegistry.register_()
class BlockDatasetMixin(BaseDataset[Batch]):

    def __init__(
        self,
        *args,
        block_size: int = 224,
        max_stride: int = 112,
        rescale: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._partitioner = Partitioner(block_size, max_stride)
        self._rescale = rescale

    def _partition(
        self,
        image: Image.Image,
    ) -> tuple[BBoxes, torch.Tensor]:
        w, h = image.size
        if w > h:
            offset = (w - h) / 2
            bbox = (offset, 0., h + offset, h)
        else:
            offset = (h - w) / 2
            bbox = (0., offset, w, w + offset)

        bbox_list = [torch.tensor([bbox])]
        block_list = [image]

        scale = 1.0
        while True:
            partitions = self._partitioner.partition_image(image)
            if partitions is None:
                break

            bboxes, blocks = partitions
            bbox_list.append(bboxes.to_tensor() * scale)
            block_list.extend(blocks)

            # cannot use `floordiv` which returns floats
            w, h = image.size
            w = int(w / self._rescale)
            h = int(h / self._rescale)
            image = image.resize((w, h))
            scale *= self._rescale

        bboxes_ = BBoxes(torch.cat(bbox_list))
        blocks_ = torch.stack(list(map(self._transforms, block_list)))
        return bboxes_, blocks_
