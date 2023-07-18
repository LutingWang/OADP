__all__ = [
    'L12DLoss',
    'MSE2DLoss',
    'FGDLoss',
    'FGFILoss',
]

import einops
import torch
import torch.nn.functional as F

from .base import LossRegistry
from .functional import FunctionalLoss, L1Loss, MSELoss


class _2DMixin(FunctionalLoss):

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        _, _, h, w = pred.shape
        if pred.shape != target.shape:
            target = F.adaptive_avg_pool2d(target, (h, w))
        if mask is not None and pred.shape != mask.shape:
            mask = F.adaptive_avg_pool2d(mask, (h, w))
        return super().forward(pred, target, mask, *args, **kwargs)


@LossRegistry.register()
class L12DLoss(_2DMixin, L1Loss):
    pass


@LossRegistry.register()
class MSE2DLoss(_2DMixin, MSELoss):
    pass


@LossRegistry.register()
class FGFILoss(MSE2DLoss):

    def forward(  # type: ignore[override]
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        n, _, h, w = pred.shape
        target = F.adaptive_avg_pool2d(target, (h, w))
        assert mask.shape == (n, 1, h, w) and mask.dtype == torch.bool

        pred = einops.rearrange(pred, 'n c h w -> n h w c')
        target = einops.rearrange(target, 'n c h w -> n h w c')
        mask = einops.rearrange(mask, 'n 1 h w -> n h w')

        return super().forward(  # type: ignore[misc]
            pred[mask],
            target[mask],
            mask=None,
            *args,
            **kwargs,
        )


@LossRegistry.register()
class FGDLoss(MSE2DLoss):

    def forward(  # type: ignore[override]
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        attn_spatial: torch.Tensor,
        attn_channel: torch.Tensor,
        mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        _, _, h, w = pred.shape
        attn_spatial = F.adaptive_avg_pool2d(attn_spatial, (h, w))
        attn_channel = F.adaptive_avg_pool2d(attn_channel, (h, w))
        mask = attn_spatial * attn_channel * mask
        return super().forward(  # type: ignore[misc]
            pred,
            target,
            mask=mask,
            *args,
            **kwargs,
        )


# @LOSSES.register()
# class LabelEncLoss(MSE2DLoss):

#     def __init__(
#         self,
#         *args,
#         num_channels: int,
#         weight: float = 1.0,
#         **kwargs,
#     ):
#         weight = LinearScheduler(
#             start_value=0,
#             end_value=weight,
#             start_iter=30000,
#             end_iter=30000,
#         )
#         super().__init__(*args, weight=weight, **kwargs)

#         norm_cfg = dict(type='GN', num_groups=1, affine=False)
#         self._adapt = Sequential(
#             ConvModule(num_channels, num_channels, 3, 1, 1),
#             ConvModule(num_channels, num_channels, 3, 1, 1),
#             ConvModule(
#                 num_channels,
#                 num_channels,
#                 3,
#                 1,
#                 1,
#                 norm_cfg=norm_cfg,
#                 act_cfg=None,
#             ),
#         )
#         self._target_norm: nn.Module = NORM_LAYERS.build(
#             norm_cfg,
#             default_args=dict(num_features=num_channels),
#         )

#     def forward(  # type: ignore[override]
#         self,
#         preds: list[torch.Tensor],
#         targets: list[torch.Tensor],
#         *args,
#         **kwargs,
#     ) -> list[torch.Tensor]:
#         losses = []
#         for pred, target in zip(preds, targets):
#             pred = self._adapt(pred)
#             target = self._target_norm(target.detach())
#             loss = super().forward(pred, target, *args, **kwargs)
#             losses.append(loss)
#         return losses

# @LOSSES.register()
# class DevLoss(MSE2DLoss):
#     def __init__(self, pos_share: float = 1, *args, **kwargs):
#         assert 0 <= pos_share <= 1
#         super().__init__(*args, **kwargs)
#         self.pos_share = pos_share

#     def pos_mask(self, mask: torch.Tensor) -> torch.Tensor:
#         pos = mask.sum().item()
#         total = mask.numel()
#         pos_frac = self.pos_share * total / pos if pos > 0 else 0
#         neg_frac = (
#             (1 - self.pos_share)  * total / (total - pos)
#             if pos < total else 0
#         )
#         weight = torch.zeros_like(mask, dtype=torch.float)
#         weight.masked_fill_(mask, pos_frac)
#         weight.masked_fill_(~mask, neg_frac)
#         return weight

#     def iou_mask(
#         self, mask: torch.Tensor, pos_thresh: float = 0.5, index: float = 2,
#     ) -> torch.Tensor:
#         """Adjust IoU mask to satisfy `pos_share` requirement.

#         Adjust IoU mask to retrieve `weight` w such that
#         $$
#             w = a * mask^index + b, \\quad a, b \\in \\R^+
#             w_p = w * (w > pos_thresh)
#             sum(w) = mask.numel()
#             sum(w_p) = sum(w) * self.pos_share
#         $$

#         The equations are sometimes unsatisfiable.
#         The algorithms goes briefly as follows:
#         1. estimate b' to balance pos and neg
#         2. normalize sum of weight to `mask.numel()`
#         """
#         assert mask.requires_grad == False
#         n_total = mask.numel()
#         pos_mask = mask > pos_thresh
#         n_pos = pos_mask.sum()
#         mask = mask ** index
#         total = mask.sum()
#         pos = mask[pos_mask].sum()
#         base = (
#             (pos / self.pos_share - total)
#             / (n_total - n_pos / self.pos_share)
#         )
#         base.clamp_(1e-5, 1)
#         mask += base
#         weight = mask * n_total / mask.sum()
#         return weight

#     def forward(
#         self, pred: torch.Tensor, target: torch.Tensor,
#         mask: torch.Tensor, *args, **kwargs,
#     ):
#         mask = mask.unsqueeze(1)  # bs x 1 x h x w
#         if mask.dtype == torch.bool:
#             mask = self.pos_mask(mask)
#         elif mask.dtype == torch.float:
#             mask = self.iou_mask(mask)
#         else:
#             mask = torch.ones_like(pred)

#         if mask.shape != pred.shape:
#             mask = F.interpolate(mask, size=pred.shape[-2:])

#         return super().forward(pred, target, *args, mask=mask, **kwargs)
