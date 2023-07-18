__all__ = [
    'DuplicatedHook',
]

from typing import Any

from .base import HookRegistry
from .standard import StandardHook


@HookRegistry.register()
class DuplicatedHook(StandardHook):

    def __init__(self, *args, num: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num = num

    def _tensor(self):
        return [super()._tensor()] * self._num
