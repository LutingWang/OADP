import itertools
import pathlib
from typing import Generator, NamedTuple

import clip
import clip.model
import PIL.Image
import todd
import torch
import torch.cuda
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from .base import BaseDataset, BaseValidator


class Batch(NamedTuple):
    output: pathlib.Path
    blocks: torch.Tensor
    bboxes: torch.Tensor


class Dataset(BaseDataset[Batch]):

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
        return self.transforms.transform(block)

    def _bbox(self, scale: float, x: int, y: int) -> todd.BBox:
        x1 = x * scale
        y1 = y * scale
        r = self._r * scale
        return (x1, y1, x1 + r, y1 + r)

    def _preprocess(
        self,
        id_: int,
        output: pathlib.Path,
        image: PIL.Image.Image,
    ) -> Batch:
        block = self.transforms.transform(image)

        w, h = image.size
        if w > h:
            bbox = ((w - h) / 2, 0, h, h)
        else:
            bbox = (0, (h - w) / 2, w, w)

        blocks = [block]
        bboxes = [bbox]
        for image, scale, x, y in self._partitions(image):
            blocks.append(self._block(image, x, y))
            bboxes.append(self._bbox(scale, x, y))

        return Batch(output, torch.stack(blocks), torch.tensor(bboxes))


class Validator(BaseValidator[Batch]):

    def _build_dataloader(
        self,
        config: todd.Config,
    ) -> torch.utils.data.DataLoader:
        config.dataset = Dataset(**config.dataset)
        return super()._build_dataloader(config)

    @classmethod
    def _build_model(cls) -> tuple[clip.model.CLIP, transforms.Compose]:
        return clip.load_default(False)

    def _run_iter(self, batch: Batch, memo: todd.utils.Memo) -> torch.Tensor:
        blocks = batch.blocks
        if todd.Store.CUDA:
            blocks = blocks.cuda()
        embeddings = self._model.encode_image(blocks)
        embeddings = F.normalize(embeddings)
        memo['result'] = dict(
            embeddings=embeddings.half(),
            bboxes=batch.bboxes.half(),
        )
        return super()._run_iter(batch, memo)


if __name__ == '__main__':
    Validator.main()
