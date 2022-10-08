__all__ = [
    'PLVBlock',
    'PreFPN',
    'GLIPBlock',
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


class PLVBlock(BaseModule):
    def __init__(
        self, *args, in_channels: int, out_channels: int, **kwargs,
    ):
        super().__init__(
            *args,
            init_cfg=dict(
                type='Xavier', layer='Conv2d',
                distribution='uniform',
            ),
            **kwargs,
        )
        self._v_proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self,
        v: torch.Tensor,
        l: torch.Tensor,
        *,
        v_weights: Optional[torch.Tensor] = None,
        l_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if l_weights is not None:
            l = torch.einsum('b n c, b n -> b n c', l, l_weights.sigmoid())
        l = einops.reduce(l, 'b n c -> b c', reduction='mean')

        v = self._v_proj(v)
        delta_v: torch.Tensor = torch.einsum('b c h w, b c -> b c h w', v, l)
        delta_v = delta_v.relu()
        return v + delta_v


class PreFPN(BaseModule):
    def __init__(
        self,
        *args,
        in_channels: List[int],
        out_channels: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._plvs = ModuleList([
            PLVBlock(in_channels=in_channel, out_channels=out_channels)
            for in_channel in in_channels
        ])

    @todd.reproduction.set_seed_temp('PreFPN')
    def init_weights(self):
        return super().init_weights()

    def forward(
        self,
        feats: Tuple[torch.Tensor, ...],
        class_embeddings: torch.Tensor,
        class_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        feats = tuple(
            plv(feat, class_embeddings, l_weights=class_weights)
            for plv, feat in zip(self._plvs, feats)
        )
        return feats


class GLIPBlock(BaseModule):
    def __init__(
        self,
        *args,
        num_heads: int,
        head_dims: int,
        channels: int,
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
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dims),
            rearrange,
        )
        self._k_proj = nn.Sequential(
            nn.Linear(channels, hidden_dims),
            rearrange,
        )
        self._v_proj = nn.Sequential(
            nn.Linear(channels, hidden_dims),
            rearrange,
        )
        self._o_proj = nn.Sequential(
            einops.layers.torch.Rearrange('b h n d -> b n (h d)'),
            nn.Linear(hidden_dims, channels),
        )

        self._gamma = nn.Parameter(torch.ones(channels) / avg_factor, requires_grad=True)
        self._dropout = nn.Dropout(dropout)
        self._drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self,
        v: torch.Tensor,
        l: torch.Tensor,
        *,
        v_weights: Optional[torch.Tensor] = None,
        l_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self._q_proj(v) * self._scale
        keys = self._k_proj(l)
        values = self._v_proj(l)
        weights: torch.Tensor = torch.einsum(
            'b h v d, b h l d -> b h v l',
            queries,
            keys,
        )

        masks = einops.reduce(
            weights, 'b h (vh vw) l -> b l vh vw',
            h=self._num_heads,
            vh=v.shape[2],
            vw=v.shape[3],
            reduction='mean',
        )

        if l_weights is not None:
            l_weights = einops.rearrange(l_weights, 'b l -> b 1 1 l')
            weights = weights + l_weights
        weights = weights.softmax(dim=-1)
        weights = self._dropout(weights)
        outputs = torch.einsum('b h v l, b h l d -> b h v d', weights, values)
        outputs = self._o_proj(outputs)

        delta_v = einops.rearrange(outputs, 'b (h w) c -> b c h w', h=v.shape[2], w=v.shape[3])
        delta_v = torch.einsum('b c h w, c -> b c h w', delta_v, self._gamma)
        delta_v = self._drop_path(delta_v)

        return v + delta_v, masks


class DyHeadBlock(BaseModule):

    def __init__(
        self,
        *args,
        channels: int,
        spatial_conv: Optional[Dict[str, Any]] = None,
        scale_attn: Optional[Dict[str, Any]] = None,
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

        if scale_attn is not None:
            self._scale_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, 1, 1),
                nn.ReLU(inplace=True),
                build_activation_layer(dict(type='HSigmoid', bias=3.0, divisor=6.0)),
            )
        else:
            self._scale_attn = None

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
        l: torch.Tensor,
        *,
        v_weights: Optional[torch.Tensor] = None,
        l_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self._spatial_conv is not None:
            assert self._spatial_conv_offset is not None
            offset_and_mask: torch.Tensor = self._spatial_conv_offset(v)
            offset = offset_and_mask[:, :self._offset_dim, :, :]
            mask = offset_and_mask[:, self._offset_dim:, :, :].sigmoid()
            v = self._spatial_conv(v, offset, mask)

        if self._scale_attn is not None:
            v = v * self._scale_attn(v)

        if self._task_attn is not None:
            v = self._task_attn(v)

        return v + l.mean() * 0


class PostBlock(BaseModule):

    def __init__(
        self,
        *args,
        channels: int,
        glip_block: Optional[Dict[str, Any]] = None,
        dyhead_block: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert glip_block is not None or dyhead_block is not None

        if glip_block is not None:
            self._glip_block = GLIPBlock(
                channels=channels,
                **glip_block,
            )
        else:
            self._glip_block = None

        if dyhead_block is not None:
            self._dyhead_block = DyHeadBlock(
                channels=channels,
                **dyhead_block,
            )
        else:
            self._dyhead_block = None

    def forward(
        self,
        v: torch.Tensor,
        *args, **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._glip_block is not None:
            v, masks = self._glip_block(v, *args, **kwargs)
        else:
            masks = None

        if self._dyhead_block is not None:
            v = self._dyhead_block(v, *args, **kwargs)

        return v, masks


class PostFPN(BaseModule):

    def __init__(
        self,
        *args,
        refine_level: int,
        num_blocks: int,
        channels: int,
        block: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._refine_level = refine_level
        self._blocks = ModuleList(
            PostBlock(channels=channels, **block)
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
        class_embeddings: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], List[Optional[torch.Tensor]]]:
        bsf = self._gather(feats)

        masks = []
        for block in self._blocks:
            bsf, mask = block(bsf, class_embeddings, l_weights=class_weights)
            masks.append(mask)

        feats = self._scatter(feats, bsf)
        return feats, masks
