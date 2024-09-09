__all__ = [
    'BaseModel',
]

from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import nn


class BaseModel(nn.Module, ABC):

    @abstractmethod
    def forward(
        self,
        texts: Iterable[str],
        batch_size: str | None = None,
    ) -> torch.Tensor:
        pass
