__all__ = [
    'Hooks',
    'ExpandedCLIP',
    'load_default',
]

from typing import Self

import clip.model
import einops
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torch import nn


class Hooks:
    _y: torch.Tensor | None
    _masks: torch.Tensor | None

    def __init__(self) -> None:
        self._y = None
        self._masks = None

    def visual_forward_pre(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor]:
        images, masks = inputs
        masks = (masks - 1) * 100.
        masks = einops.rearrange(masks, 'b h w -> b (h w)')
        zeros = masks.new_zeros(masks.shape[0], 1)
        self._masks = torch.cat([masks, zeros], dim=-1)
        return (images, )

    def visual_transformer_forward_pre(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor],
    ) -> None:
        x, = inputs
        self._y = x[[0]]

    def visual_transformer_r_forward_pre(
        self,
        module: clip.model.ResidualAttentionBlock,
        inputs: tuple[torch.Tensor],
    ) -> None:
        assert self._y is not None
        x, = inputs
        y = self._y

        masks = einops.repeat(
            self._masks,
            'b v -> (b h) 1 v',
            h=module.attn.num_heads,
        )
        x = module.ln_1(torch.cat([x[1:], y]))
        y = y + module.attn(
            x[[-1]],
            x,
            x,
            need_weights=False,
            attn_mask=masks,
        )[0]
        y = y + module.mlp(module.ln_2(y))
        self._y = y

    def visual_transformer_forward(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert self._y is not None
        y = self._y
        self._y = None
        return y

    def visual_forward(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        self._masks = None


class ExpandedCLIP(nn.Module):

    def __init__(self, *args, model: clip.model.CLIP, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model

    @property
    def model(self) -> clip.model.CLIP:
        return self._model

    @classmethod
    def wrap(cls, model: clip.model.CLIP, upsample: int = 2) -> Self:
        model = model.requires_grad_(False)
        model = model.eval()

        visual = model.visual
        positional_embedding = visual.interpolate_positional_embedding(
            (visual.grid * 2, ) * 2,
        )
        visual.positional_embedding = nn.Parameter(positional_embedding, False)
        visual.grid *= upsample

        conv1 = visual.conv1
        conv1.stride = tuple(s // upsample for s in conv1.stride)
        conv1.padding = ((visual.patch_size - 1) // 2, ) * 2

        hooks = Hooks()
        visual.register_forward_pre_hook(hooks.visual_forward_pre)
        transformer = visual.transformer
        transformer.register_forward_pre_hook(
            hooks.visual_transformer_forward_pre
        )
        transformer.register_forward_hook(hooks.visual_transformer_forward)
        r: clip.model.ResidualAttentionBlock
        for r in transformer.resblocks:
            r.register_forward_pre_hook(hooks.visual_transformer_r_forward_pre)

        return cls(model=model)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        images = images.type(self._model.dtype)
        masks = masks.type(self._model.dtype)
        embeddings = self._model.visual(images, masks)
        embeddings = F.normalize(embeddings)
        return embeddings


def load_default() -> tuple[ExpandedCLIP, tf.Compose]:
    model, transforms = clip.load_default(False)
    return ExpandedCLIP.wrap(model), transforms
