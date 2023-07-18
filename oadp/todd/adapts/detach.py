import torch
from typing import List
from .base import AdaptRegistry, BaseAdapt


@AdaptRegistry.register()
class Detach(BaseAdapt):

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()


@AdaptRegistry.register()
class ListDetach(BaseAdapt):

    def forward(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        return [tensor.detach() for tensor in tensors]
