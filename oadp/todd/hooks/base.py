__all__ = [
    'HookRegistry',
    'BaseHook',
]

from abc import abstractmethod
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Protocol

import torch.nn as nn

from ..base import Registry, StatusMixin, get_

if TYPE_CHECKING:

    class _HookProto(Protocol):
        _path: str

        def register_tensor(self, tensor) -> None:
            pass

else:
    _HookProto = object


class Status(IntEnum):
    INITIALIZED = auto()
    BINDED = auto()
    REGISTERED = auto()


class _HookMixin(_HookProto):

    def _forward_hook(self, module: nn.Module, input_, output) -> None:
        self.register_tensor(output)

    def bind(self, model: nn.Module) -> None:
        module: nn.Module = get_(model, self._path)
        self._handle = module.register_forward_hook(self._forward_hook)

    def unbind(self) -> None:
        self._handle.remove()


class _TrackingMixin(_HookProto):

    def __init__(
        self,
        tracking_mode: bool = False,
    ) -> None:
        self._tracking_mode = tracking_mode

    @property
    def tracking_mode(self) -> bool:
        return self._tracking_mode

    def bind(self, model: nn.Module) -> None:
        self._model = model

    def unbind(self) -> None:
        delattr(self, '_model')

    def track_tensor(self) -> None:
        if not self._tracking_mode:
            raise AttributeError(f"Hook {self._path} does not support track.")
        tensor = get_(self._model, self._path)
        self.register_tensor(tensor)


class BaseHook(StatusMixin[Status], _HookMixin, _TrackingMixin):

    def __init__(
        self,
        path: str,
        tracking_mode: bool = False,
    ) -> None:
        StatusMixin.__init__(self, Status.INITIALIZED)
        _HookMixin.__init__(self)
        _TrackingMixin.__init__(self, tracking_mode)
        self._path = path
        self._reset()

    def __call__(self):
        if self.status != Status.REGISTERED:
            raise RuntimeError(f"Hook {self._path} is not registered.")
        return self._tensor()

    @property
    def path(self) -> str:
        return self._path

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def _register_tensor(self, tensor) -> None:
        pass

    @abstractmethod
    def _tensor(self):
        pass

    @StatusMixin.transit(Status.INITIALIZED, Status.BINDED)
    def bind(self, model: nn.Module) -> None:
        if self._tracking_mode:
            _TrackingMixin.bind(self, model)
        else:
            _HookMixin.bind(self, model)

    @StatusMixin.transit(Status.BINDED, Status.INITIALIZED)
    def unbind(self) -> None:
        if self._tracking_mode:
            _TrackingMixin.unbind(self)
        else:
            _HookMixin.unbind(self)

    @StatusMixin.transit(
        (Status.BINDED, Status.REGISTERED),
        Status.REGISTERED,
    )
    def register_tensor(self, tensor) -> None:
        self._register_tensor(tensor)

    @StatusMixin.transit(
        (Status.BINDED, Status.REGISTERED),
        Status.BINDED,
    )
    def reset(self) -> None:
        self._reset()


class HookRegistry(Registry):
    pass
