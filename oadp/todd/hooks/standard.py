__all__ = [
    'StandardHook',
]

from .base import BaseHook, HookRegistry


@HookRegistry.register()
class StandardHook(BaseHook):

    def _reset(self):
        self.__tensor = None

    def _tensor(self):
        return self.__tensor

    def _register_tensor(self, tensor) -> None:
        self.__tensor = tensor
