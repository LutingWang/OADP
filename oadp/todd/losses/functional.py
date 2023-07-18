__all__ = [
    'L1Loss',
    'MSELoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    'CrossEntropyLoss',
]

from abc import abstractmethod

import torch
import torch.nn.functional as F

from .base import BaseLoss, LossRegistry


class FunctionalLoss(BaseLoss):

    @staticmethod
    @abstractmethod
    def func(*args, **kwargs) -> torch.Tensor:
        pass

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if mask is None:
            loss = self.func(
                pred,
                target,
                *args,
                reduction=self.reduction,
                **kwargs,
            )
        else:
            loss = self.func(pred, target, *args, reduction='none', **kwargs)
            loss = self.reduce(loss, mask)
        return loss


class NormMixin(FunctionalLoss):

    def __init__(self, *args, norm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._norm = norm

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self._norm:
            pred = F.normalize(pred)
            target = F.normalize(target)
        return super().forward(pred, target, *args, **kwargs)


@LossRegistry.register()
class L1Loss(NormMixin, FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.l1_loss(*args, **kwargs)


@LossRegistry.register()
class MSELoss(NormMixin, FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.mse_loss(*args, **kwargs)


@LossRegistry.register()
class BCELoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy(*args, **kwargs)


@LossRegistry.register()
class BCEWithLogitsLoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(*args, **kwargs)


@LossRegistry.register()
class CrossEntropyLoss(FunctionalLoss):

    @staticmethod
    def func(*args, **kwargs) -> torch.Tensor:
        return F.cross_entropy(*args, **kwargs)
