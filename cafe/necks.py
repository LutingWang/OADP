__all__ = [
    'PostBlock',
    'PostFPN',
]

from typing import Any, Dict, List, Optional, Tuple

import einops
import einops.layers.torch
import todd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from timm.models.layers import DropPath

from mmcv.cnn import build_activation_layer, constant_init, normal_init
from mmdet.models.utils import DyReLU
from mmdet.models.necks.dyhead import DyDCNv2


class CrossAttn(BaseModule):
    def __init__(
        self,
        *args,
        v_channels: int,
        l_channels: int,
        num_heads: int,
        head_dims: int,
        avg_factor: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform',
            ),
            **kwargs,
        )

        self._num_heads = num_heads
        self._head_dim = head_dims
        self._scale = head_dims ** (-0.5)

        hidden_dims = num_heads * head_dims
        rearrange = einops.layers.torch.Rearrange(
            'b n (h d) -> b h n d',
            h=num_heads,
            d=head_dims,
        )
        self._q_proj = nn.Sequential(
            einops.layers.torch.Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(v_channels),
            nn.Linear(v_channels, hidden_dims),
            rearrange,
        )
        self._k_proj = nn.Sequential(
            nn.Linear(l_channels, hidden_dims),
            rearrange,
        )
        # self._k_proj = rearrange
        self._v_proj = nn.Sequential(
            nn.Linear(l_channels, hidden_dims),
            rearrange,
        )
        # self._v_proj = rearrange
        self._o_proj = nn.Sequential(
            einops.layers.torch.Rearrange('b h n d -> b n (h d)'),
            nn.Linear(hidden_dims, v_channels),
        )

        self._gamma = nn.Parameter(torch.ones(v_channels) / avg_factor, requires_grad=True)
        self._dropout = nn.Dropout(dropout)
        self._drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self,
        v: torch.Tensor,
        l: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self._q_proj(v) * self._scale
        keys = self._k_proj(l)
        values = self._v_proj(l)
        weights: torch.Tensor = torch.einsum(
            'b h v d, b h l d -> b h v l',
            queries,
            keys,
        )

        weights = weights.softmax(dim=-1)
        weights = self._dropout(weights)
        outputs = torch.einsum('b h v l, b h l d -> b h v d', weights, values)
        outputs = self._o_proj(outputs)

        delta_v = einops.rearrange(outputs, 'b (h w) c -> b c h w', h=v.shape[2], w=v.shape[3])
        delta_v = torch.einsum('b c h w, c -> b c h w', delta_v, self._gamma)
        delta_v = self._drop_path(delta_v)

        return v + delta_v


class PostBlock(BaseModule):

    def __init__(
        self,
        *args,
        spatial_conv: Optional[Dict[str, Any]] = None,
        task_attn: Optional[Dict[str, Any]] = None,
        cross_attn: Optional[Dict[str, Any]] = None,
        zero_init_offset=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if spatial_conv is not None:
            spatial_conv = spatial_conv.copy()
            channels = spatial_conv.pop('channels')
            # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
            self._offset_and_mask_dim = 3 * 3 * 3
            self._offset_dim = 2 * 3 * 3
            self._spatial_conv_offset = nn.Conv2d(
                channels,
                self._offset_and_mask_dim,
                3,
                padding=1,
            )
            self._spatial_conv = DyDCNv2(
                channels,
                channels,
                **spatial_conv,
            )
        else:
            self._spatial_conv_offset = None
            self._spatial_conv = None

        if task_attn is not None:
            self._task_attn = DyReLU(**task_attn)
        else:
            self._task_attn = None

        if cross_attn is not None:
            self._cross_attn = CrossAttn(
                **cross_attn,
            )
        else:
            self._cross_attn = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)

        if zero_init_offset and self._spatial_conv_offset is not None:
            constant_init(self._spatial_conv_offset, 0)

    def forward(
        self,
        v: torch.Tensor,
        l: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self._spatial_conv is not None:
            assert self._spatial_conv_offset is not None
            offset_and_mask = self._spatial_conv_offset(v)
            offset = offset_and_mask[:, :self._offset_dim, :, :]
            mask = offset_and_mask[:, self._offset_dim:, :, :].sigmoid()
            v = self._spatial_conv(v, offset, mask)

        if self._task_attn is not None:
            v = self._task_attn(v)

        if self._cross_attn is not None:
            v = self._cross_attn(v, l)

        return v


class PostFPN(BaseModule):

    def __init__(
        self,
        *args,
        refine_level: int,
        num_blocks: int,
        block: Dict[str, Any],
        warmup: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._refine_level = refine_level
        self._blocks = ModuleList(
            PostBlock(**block)
            for _ in range(num_blocks)
        )
        self._warmup = todd.schedulers.WarmupScheduler(iter_=warmup)

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
            feats[i] = feats[i] + F.interpolate(bsf, feats[i].shape[-2:], mode='nearest') * self._warmup
        feats[self._refine_level] = feats[self._refine_level] + bsf * self._warmup
        for i in range(self._refine_level + 1, len(feats)):
            feats[i] = feats[i] + F.adaptive_max_pool2d(bsf, feats[i].shape[-2:]) * self._warmup
        return tuple(feats)

    def forward(
        self,
        feats: Tuple[torch.Tensor, ...],
        text_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        bsf = self._gather(feats)

        for block in self._blocks:
            bsf = block(bsf, text_feats)

        feats = self._scatter(feats, bsf)
        return feats
