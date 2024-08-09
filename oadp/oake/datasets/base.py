__all__ = [
    'BaseDataset',
]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import torch.distributed
import torch.utils.data.distributed
from PIL import Image
from todd.runners.utils import RunnerHolderMixin
from todd.datasets import COCODataset
from torch import nn

if TYPE_CHECKING:
    from ..runners import BaseValidator

T = TypeVar('T')


class BaseDataset(
    RunnerHolderMixin[nn.Module],
    COCODataset,
    Generic[T],
    ABC,
):
    validator: 'BaseValidator[nn.Module]'

    def __init__(self, *args, check: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._check = check

    def _access(self, index: int) -> tuple[str, Image.Image | None]:
        key = self._keys[index]
        output_path = self.validator.output_path(key)
        if output_path.exists():
            if not self._check:
                return key, None
            try:
                torch.load(output_path, 'cpu')
                return key, None
            except Exception:  # pylint: disable=broad-exception-caught
                self.validator.logger.info("Fixing %s", output_path)
        return super()._access(index)

    def __getitem__(self, index: int) -> T | None:
        key, image = self._access(index)
        if image is None:
            return None
        return self._preprocess(key, image)

    @abstractmethod
    def _preprocess(self, key: str, image: Image.Image) -> T:
        pass
