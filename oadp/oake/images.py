import pathlib
from typing import NamedTuple

import clip
import clip.model
import PIL.Image
import todd
import torch
import torch.cuda
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from .base import BaseDataset, BaseValidator


class Batch(NamedTuple):
    output: pathlib.Path
    image: torch.Tensor


class Dataset(BaseDataset[Batch]):

    def _preprocess(
        self,
        id_: int,
        output: pathlib.Path,
        image: PIL.Image.Image,
    ) -> Batch:
        image = self.transforms.transform(image)
        return Batch(output, image)


class Validator(BaseValidator[Batch]):

    def _build_dataloader(
        self,
        config: todd.Config,
    ) -> torch.utils.data.DataLoader:
        config.dataset = Dataset(**config.dataset)
        return super()._build_dataloader(config)

    @classmethod
    def _build_model(cls) -> tuple[clip.model.CLIP, transforms.Compose]:
        return clip.load_default(True)

    def _run_iter(
        self,
        batch: Batch,
        memo: todd.utils.Memo,
    ) -> torch.Tensor:
        image = batch.image.unsqueeze(0)
        if todd.Store.CUDA:
            image = image.cuda()
        embedding = self._model.encode_image(image)
        embedding = F.normalize(embedding)
        memo['result'] = embedding.squeeze(0).half()
        return super()._run_iter(batch, memo)


if __name__ == '__main__':
    Validator.main()
