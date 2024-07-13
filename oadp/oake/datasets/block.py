__all__ = [
    'BlockBatch',
    'BlockDataset',
]

import itertools
from typing import Generator, NamedTuple

import PIL.Image
import todd.tasks.object_detection as od
import torch
import torch.cuda
import torch.utils.data
import torch.utils.data.distributed

from ..registries import OADPDatasetRegistry
from .base import BaseDataset


class BlockBatch(NamedTuple):
    id_: int
    blocks: torch.Tensor
    bboxes: torch.Tensor


@OADPDatasetRegistry.register_()
class BlockDataset(BaseDataset[BlockBatch]):

    def __init__(
        self,
        *args,
        block_size: int = 224,
        max_stride: int = 112,
        rescale: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._r = block_size
        self._s = max_stride
        self._rescale = rescale

    def _partition(self, length: int) -> list[int]:
        if length < self._r:
            return []

        result = [0]
        if length == self._r:
            return result

        n = (length - self._r - 1) // self._s + 1
        q, r = divmod(length - self._r, n)
        for i in range(n):
            result.append(result[-1] + q + (i < r))
        return result

    def _partitions(
        self,
        image: PIL.Image.Image,
    ) -> Generator[tuple[PIL.Image.Image, float, int, int], None, None]:
        scale = 1.0
        while True:
            w, h = image.size
            partitions = list(
                itertools.product(
                    self._partition(w),
                    self._partition(h),
                )
            )
            if len(partitions) == 0:
                return
            for x, y in partitions:
                yield image, scale, x, y

            # cannot use `floordiv` which returns floats
            image = image.resize((
                int(w / self._rescale),
                int(h / self._rescale),
            ))
            scale *= self._rescale

    def _block(self, image: PIL.Image.Image, x: int, y: int) -> torch.Tensor:
        block = image.crop((x, y, x + self._r, y + self._r))
        return self._transforms(block)

    def _bbox(self, scale: float, x: int, y: int) -> od.BBox:
        x1 = x * scale
        y1 = y * scale
        r = self._r * scale
        return (x1, y1, x1 + r, y1 + r)

    def _preprocess(
        self,
        id_: int,
        image: PIL.Image.Image,
    ) -> BlockBatch:
        block = self._transforms(image)

        bbox: od.BBox
        w, h = image.size
        if w > h:
            bbox = ((w - h) / 2, 0., h, h)
        else:
            bbox = (0., (h - w) / 2, w, w)

        blocks = [block]
        bboxes = [bbox]
        for image_, scale, x, y in self._partitions(image):
            blocks.append(self._block(image_, x, y))
            bboxes.append(self._bbox(scale, x, y))

        return BlockBatch(id_, torch.stack(blocks), torch.tensor(bboxes))
