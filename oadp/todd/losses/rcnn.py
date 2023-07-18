__all__ = [
    'SGFILoss',
]

from typing import Callable

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry
from .functional import MSELoss


@LossRegistry.register()
class SGFILoss(MSELoss):

    def __init__(
        self,
        *args,
        in_channels: int = 256,
        hidden_channels: int = 128,
        out_channels: int = 64,
        **kwargs,
    ):
        from mmcv.cnn import ConvModule

        super().__init__(*args, **kwargs)
        self._embed: Callable[..., torch.Tensor] = nn.Sequential(
            ConvModule(in_channels, hidden_channels, 3, stride=2),
            ConvModule(hidden_channels, out_channels, 3, stride=2)
        )
        self._tau = nn.Parameter(torch.tensor([1.0], dtype=torch.float32)) # type: ignore

    def forward(  # type: ignore[override]
        self,
        preds,
        targets: torch.Tensor,
        *args,
        **kwargs,
    ):
        """Re-implementation of G-DetKD.

        Refer to http://arxiv.org/abs/2108.07482.

        Args:
            pred: l x r x c x h x w
                Each of the ``l`` levels generate ``r`` RoIs.
                Typical shape is 4 x 1024 x 256 x 7 x 7.

            target: r x c x h x w

        Returns:
            loss
        """
        fused_preds = torch.stack(preds)
        embed_pred = einops.rearrange(fused_preds, 'l r c h w -> (l r) c h w')
        embed_pred = self._embed(embed_pred)
        embed_target = self._embed(targets)

        embed_pred = einops.rearrange(
            embed_pred,
            '(l r) out_channels 1 1 -> l r out_channels',
            l=len(preds),
        )
        embed_target = einops.rearrange(
            embed_target,
            'r out_channels 1 1 -> r out_channels',
        )
        similarity = torch.einsum(
            'l r c, r c -> l r',
            embed_pred,
            embed_target,
        )
        similarity = F.softmax(similarity / self._tau, dim=1)

        fused_pred = torch.einsum(
            'l r c h w, l r -> r c h w',
            fused_preds,
            similarity,
        )
        return super().forward(fused_pred, targets, *args, **kwargs)