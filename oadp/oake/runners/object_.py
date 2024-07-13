__all__ = [
    'ObjectValidator',
]

import math
from typing import Any, cast

import clip
import clip.model
import einops
import todd
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from todd.runners import Memo
from torch import nn

from ..datasets import ObjectBatch
from ..registries import OADPRunnerRegistry
from .base import BaseValidator


class Hooks:

    def __init__(self) -> None:
        self._y: torch.Tensor | None = None
        self._attn_mask: torch.Tensor | None = None

    def visual_forward_pre(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor]:
        attn_mask = einops.rearrange(inputs[-1], 'b 1 h w -> b (h w)')
        zeros = attn_mask.new_zeros(attn_mask.shape[0], 1)
        attn_mask = torch.cat([attn_mask, zeros], dim=-1)
        attn_mask *= -100
        self._attn_mask = attn_mask
        return inputs[:-1]

    def transformer_forward_pre(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor],
    ) -> None:
        x, = inputs
        self._y = x[[0]]

    def residual_attention_block_forward_pre(
        self,
        module: clip.model.ResidualAttentionBlock,
        inputs: tuple[torch.Tensor],
    ) -> None:
        assert self._y is not None
        x, = inputs
        y = self._y

        attn_mask = einops.repeat(
            self._attn_mask,
            'b v -> (b h) 1 v',
            h=module.attn.num_heads,
        )
        x = module.ln_1(torch.cat([x[1:], y]))
        y = y + module.attn(
            x[[-1]],
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
        )[0]
        y = y + module.mlp(module.ln_2(y))
        self._y = y

    def transformer_forward(
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
        self._attn_mask = None


@OADPRunnerRegistry.register_()
class ObjectValidator(BaseValidator[ObjectBatch]):

    def __init__(self, *args, mini_batch_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mini_batch_size = mini_batch_size

    def _build(self, *args, **kwargs) -> None:
        super()._build(*args, clip_=todd.Config(adaptive=False), **kwargs)

    def _build_model(
        self,
        *args,
        model: clip.model.CLIP,
        upsample: int = 2,
        **kwargs,
    ) -> None:
        visual = model.visual
        positional_embedding = visual.interpolate_positional_embedding(
            (visual.grid * 2, ) * 2
        )
        visual.positional_embedding = nn.Parameter(positional_embedding)
        visual.grid *= upsample

        conv1 = visual.conv1
        conv1.stride = tuple(s // upsample for s in conv1.stride)
        conv1.padding = ((visual.patch_size - 1) // 2, ) * 2

        hooks = Hooks()
        visual.register_forward_pre_hook(hooks.visual_forward_pre)
        transformer = visual.transformer
        transformer.register_forward_pre_hook(hooks.transformer_forward_pre)
        transformer.register_forward_hook(hooks.transformer_forward)
        for r in transformer.resblocks:
            r = cast(clip.model.ResidualAttentionBlock, r)
            r.register_forward_pre_hook(
                hooks.residual_attention_block_forward_pre
            )

        super()._build_model(*args, model=model, **kwargs)

    def _run_iter(self, batch: Any, memo: Memo, *args, **kwargs) -> Memo:
        objects = cast(ObjectBatch, batch).objects
        masks = cast(ObjectBatch, batch).masks
        bboxes = cast(ObjectBatch, batch).bboxes
        objectness = cast(ObjectBatch, batch).objectness
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            objects = objects.cuda()
            masks = masks.cuda()
            bboxes = bboxes.cuda()
            objectness = objectness.cuda()
        embeddings = []
        for i in range(math.ceil(objects.shape[0] / self._mini_batch_size)):
            indices = slice(
                i * self._mini_batch_size,
                (i + 1) * self._mini_batch_size,
            )
            o = objects[indices].type(self._model.dtype)
            m = masks[indices].type(self._model.dtype)
            embedding = self._model.visual(o, m)
            embedding = F.normalize(embedding)
            embeddings.append(embedding)
        memo['output'] = dict(
            embeddings=torch.cat(embeddings).half(),
            bboxes=bboxes.half(),
            objectness=objectness.half()
        )
        return memo
