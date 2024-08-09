__all__ = [
    'GlobalDataset',
]

from typing import NamedTuple

from PIL import Image
import torch
import torch.cuda
import torch.distributed
import torch.utils.data
import torch.utils.data.distributed

from ..registries import OAKEDatasetRegistry
from .base import BaseDataset
from torch import nn
from typing import TypeVar


class T(NamedTuple):
    key: str
    image: torch.Tensor


@OAKEDatasetRegistry.register_()
class GlobalDataset(BaseDataset[T]):

    def _preprocess(self, key: str, image: Image.Image) -> T:
        return T(key, self._transforms(image))
