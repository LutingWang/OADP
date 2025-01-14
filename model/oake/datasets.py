__all__ = [
    'BaseDataset',
]

from abc import ABC
from typing import TypedDict, TypeVar

import torch.distributed
import torch.utils.data.distributed
from todd.datasets import PILDataset
from todd.runners.utils import RunnerHolderMixin
from torch import nn

from .runners import BaseValidator


class Batch(TypedDict):
    id_: str


T = TypeVar('T', bound=Batch)


class BaseDataset(RunnerHolderMixin[nn.Module], PILDataset[T], ABC):
    runner: BaseValidator

    def __init__(self, *args, auto_fix: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._auto_fix = auto_fix

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        assert self._transforms is None
        self._transforms = self.runner.transforms

    def exists(self, index: int) -> bool:
        key = self._keys[index]
        output_path = self.runner.output_path(key)
        if not output_path.exists():
            return False
        if not self._auto_fix:
            return True
        try:
            torch.load(output_path, 'cpu')
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            self.runner.logger.info("Fixing %s", output_path)
            return False

    def _getitem(self, index: int) -> T:
        return super().__getitem__(index)  # type: ignore[safe-super]

    def __getitem__(self, index: int) -> T | None:  # type: ignore[override]
        if self.exists(index):
            return None
        return self._getitem(index)
