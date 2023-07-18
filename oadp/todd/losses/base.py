__all__ = [
    'BaseLoss',
    'LossRegistry',
]

from typing import Literal

import torch

from ..base import Config, Module, Registry
from .schedulers import BaseScheduler, SchedulerRegistry

Reduction = Literal['none', 'mean', 'sum', 'prod']


class BaseLoss(Module):

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight = 1.0,
        bound = None,
        **kwargs,
    ) -> None:
        if isinstance(weight, float):
            weight = Config(type='ConstantScheduler', gain=weight)
        if bound is not None and bound <= 1e-4:
            raise ValueError('bound must be greater than 1e-4')
        super().__init__(**kwargs)
        self._reduction = reduction
        self._weight: BaseScheduler = SchedulerRegistry.build(weight)
        self._threshold = None if bound is None else bound / self.weight

        self.register_forward_hook(forward_hook) # type: ignore

    @property
    def reduction(self) -> Reduction:
        return self._reduction # type: ignore

    @property
    def weight(self) -> float:
        return self._weight()

    @property
    def threshold(self):
        return self._threshold

    def reduce(
        self,
        loss: torch.Tensor,
        mask = None,
    ) -> torch.Tensor:
        if mask is not None:
            loss = loss * mask
        if self._reduction == 'none':
            pass
        elif self._reduction in ['sum', 'mean', 'prod']:
            loss = getattr(loss, self._reduction)()
        else:
            raise NotImplementedError(self._reduction)
        return loss


def forward_hook(
    module: BaseLoss,
    input_,
    output: torch.Tensor,
) -> torch.Tensor:
    weight = module.weight
    if module.threshold is None:
        return weight * output

    # coef = bound / (weight * output)
    coef = module.threshold / output.item()
    # if bound < weight * output
    if coef < 1.0:
        # weight = bound / output
        weight *= coef

    output = weight * output
    return output


class LossRegistry(Registry):
    pass
