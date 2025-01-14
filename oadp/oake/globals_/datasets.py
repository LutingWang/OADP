__all__ = [
    'GlobalDataset',
]

from typing import TypedDict

import torch

from ..datasets import BaseDataset
from ..registries import OAKEDatasetRegistry


class Batch(TypedDict):
    id_: str
    image: torch.Tensor


@OAKEDatasetRegistry.register_()
class GlobalDataset(BaseDataset[Batch]):
    pass
