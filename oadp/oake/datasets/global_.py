__all__ = [
    'GlobalBatch',
    'GlobalDataset',
]

from typing import NamedTuple

import PIL.Image
import torch
import torch.cuda
import torch.distributed
import torch.utils.data
import torch.utils.data.distributed

from ..registries import OADPDatasetRegistry
from .base import BaseDataset


class GlobalBatch(NamedTuple):
    id_: int
    image: torch.Tensor


@OADPDatasetRegistry.register_()
class GlobalDataset(BaseDataset[GlobalBatch]):

    def _preprocess(
        self,
        id_: int,
        image: PIL.Image.Image,
    ) -> GlobalBatch:
        return GlobalBatch(id_, self._transforms(image))
