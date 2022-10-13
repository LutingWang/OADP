__all__ = [
    'PostBlock',
    'PostFPN',
]

from typing import Any, Dict, List, Optional, Tuple

import einops
import einops.layers.torch
import todd.reproduction
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from timm.models.layers import DropPath

from mmcv.cnn import build_activation_layer, constant_init, normal_init
from mmdet.models.utils import DyReLU
from mmdet.models.necks.dyhead import DyDCNv2


class PostBlock(BaseModule):

    def __init__(
        self,
        *args,
        channels: int,
        spatial_conv: Optional[Dict[str, Any]] = None,
        task_attn: Optional[Dict[str, Any]] = None,
        zero_init_offset=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if spatial_conv is not None:
            # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
            self._offset_and_mask_dim = 3 * 3 * 3
            self._offset_dim = 2 * 3 * 3
            self._spatial_conv_offset = nn.Conv2d(
                channels, self._offset_and_mask_dim, 3, padding=1,
            )
            self._spatial_conv = DyDCNv2(channels, channels)
        else:
            self._spatial_conv_offset = None
            self._spatial_conv = None

        if task_attn is not None:
            self._task_attn = DyReLU(channels)
        else:
            self._task_attn = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)

        if zero_init_offset and self._spatial_conv_offset is not None:
            constant_init(self._spatial_conv_offset, 0)

    def forward(
        self,
        v: torch.Tensor,
    ) -> torch.Tensor:

        if self._spatial_conv is not None:
            assert self._spatial_conv_offset is not None
            offset_and_mask = self._spatial_conv_offset(v)
            offset = offset_and_mask[:, :self._offset_dim, :, :]
            mask = offset_and_mask[:, self._offset_dim:, :, :].sigmoid()
            v = self._spatial_conv(v, offset, mask)

        if self._task_attn is not None:
            v = self._task_attn(v)

        return v


class PostFPN(BaseModule):

    def __init__(
        self,
        *args,
        refine_level: int,
        num_blocks: int,
        block: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._refine_level = refine_level
        self._blocks = ModuleList(
            PostBlock(**block)
            for _ in range(num_blocks)
        )

    @todd.reproduction.set_seed_temp('PostFPN')
    def init_weights(self):
        return super().init_weights()

    def _gather(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        bsf = feats[self._refine_level]
        bsf = bsf.new_empty((len(feats),) + bsf.shape)
        for i in range(self._refine_level):
            bsf[i] = F.adaptive_max_pool2d(feats[i], bsf.shape[-2:])
        bsf[self._refine_level] = feats[self._refine_level]
        for i in range(self._refine_level + 1, len(feats)):
            bsf[i] = F.interpolate(feats[i], bsf.shape[-2:], mode='nearest')
        return einops.reduce(bsf, 'l b c h w -> b c h w', reduction='mean')

    def _scatter(self, feats: Tuple[torch.Tensor, ...], bsf: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feats = list(feats)
        for i in range(self._refine_level):
            feats[i] = feats[i] + F.interpolate(bsf, feats[i].shape[-2:], mode='nearest')
        feats[self._refine_level] = feats[self._refine_level] + bsf
        for i in range(self._refine_level + 1, len(feats)):
            feats[i] = feats[i] + F.adaptive_max_pool2d(bsf, feats[i].shape[-2:])
        return tuple(feats)

    def forward(
        self,
        feats: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        bsf = self._gather(feats)

        for block in self._blocks:
            bsf = block(bsf)

        feats = self._scatter(feats, bsf)
        return feats
