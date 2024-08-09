__all__ = [
    'BaseLoader',
]

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')


class BaseLoader(Generic[T], ABC):

    @abstractmethod
    def __call__(self, category_name: str) -> T:
        pass
