__all__ = [
    'ObjectDataset',
]

from abc import ABC, abstractmethod
from typing import Any, TypedDict

import todd.tasks.object_detection as od
import torch.distributed
import torch.utils.data.distributed

from .runners import ObjectValidator
from ..datasets import BaseDataset

from ..registries import OAKEDatasetRegistry


class Batch(TypedDict):
    id_: str
    bboxes: od.FlattenBBoxesXYWH
    categories: torch.Tensor
    crops: torch.Tensor
    masks: torch.Tensor


@OAKEDatasetRegistry.register_()
class ObjectDataset(BaseDataset[Batch], ABC):
    runner: ObjectValidator

    @property
    @abstractmethod
    def categories(self) -> list[Any]:
        pass
