import torch

from ..utils import ListTensor
from .base import AdaptRegistry, BaseAdapt


class ListTensorAdapt(BaseAdapt):
    func: staticmethod

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.func(*args, **kwargs)


@AdaptRegistry.register()
class Stack(ListTensorAdapt):
    func = staticmethod(ListTensor.stack)


@AdaptRegistry.register()
class Index(ListTensorAdapt):
    func = staticmethod(ListTensor.index)
