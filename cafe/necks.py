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

from .patches import DyHeadBlock


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
        feats: Tuple[torch.Tensor],
        class_embeddings: torch.Tensor,
        class_weights: Optional[torch.Tensor],
    ):
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

        self._dyhead = DyHeadBlock(channels, channels)

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

        weights = weights.softmax(dim=-1)
        if l_weights is not None:
            weights = torch.einsum('b h v l, b l -> b h v l', weights, l_weights)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = self._dropout(weights)
        outputs = torch.einsum('b h v l, b h l d -> b h v d', weights, values)
        outputs = self._o_proj(outputs)

        delta_v = einops.rearrange(outputs, 'b (h w) c -> b c h w', h=v.shape[2], w=v.shape[3])
        delta_v = torch.einsum('b c h w, c -> b c h w', delta_v, self._gamma)
        delta_v = self._drop_path(delta_v)

        v = self._dyhead(v + delta_v)
        return v, masks


class PostFPN(BaseModule):
    def __init__(
        self,
        *args,
        refine_level: int,
        channels: int,
        num_blocks: int,
        glip_block: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._refine_level = refine_level
        self._glip_blocks = ModuleList(
            GLIPBlock(
                channels=channels,
                avg_factor=num_blocks,
                **glip_block,
            ) for l in range(num_blocks)
        )

    @todd.reproduction.set_seed_temp('PostFPN')
    def init_weights(self):
        return super().init_weights()

    def _gather(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        bsf = feats[self._refine_level]
        bsf = bsf.new_empty((len(feats),) + bsf.shape)
        for i in range(self._refine_level):
            bsf[i] = F.adaptive_max_pool2d(feats[i], bsf.shape[-2:])
        bsf[self._refine_level] = feats[self._refine_level]
        for i in range(self._refine_level + 1, len(feats)):
            bsf[i] = F.interpolate(feats[i], bsf.shape[-2:], mode='nearest')
        return einops.reduce(bsf, 'l b c h w -> b c h w', reduction='mean')

    def _scatter(self, feats: Tuple[torch.Tensor], bsf: torch.Tensor) -> List[torch.Tensor]:
        feats = list(feats)
        for i in range(self._refine_level):
            feats[i] = feats[i] + F.interpolate(bsf, feats[i].shape[-2:], mode='nearest')
        feats[self._refine_level] = feats[self._refine_level] + bsf
        for i in range(self._refine_level + 1, len(feats)):
            feats[i] = feats[i] + F.adaptive_max_pool2d(bsf, feats[i].shape[-2:])
        return feats

    def forward(
        self,
        feats: Tuple[torch.Tensor],
        class_embeddings: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bsf = self._gather(feats)

        masks = []
        for glip_block in self._glip_blocks:
            bsf, mask = glip_block(bsf, class_embeddings, l_weights=class_weights)
            masks.append(mask)

        feats = self._scatter(feats, bsf)
        return feats, masks
