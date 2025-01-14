__all__ = [
    'CLIPViT',
    'ExpandedCLIPViT',
    'clip_vit',
]

from typing import Callable

import einops
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as tf_v2
from PIL import Image
from todd.datasets import CLIP_MEAN, CLIP_STD
from todd.models.modules import CLIPViT as BaseCLIPViT
from todd.models.modules.clip import CLIPBlock
from torch import nn

from ..registries import OAKEModelRegistry
from .expanders import ExpandTransform


class CLIPViT(BaseCLIPViT):

    def forward(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        cls_, _ = super().forward(image, False)
        return cls_


class ExpandedCLIPBlock(CLIPBlock):

    def forward(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        obj: torch.Tensor,
        *,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.cat([x[:, 1:], obj], 1)
        masks = einops.repeat(
            masks,
            'b l -> (b nh) 1 l',
            nh=self._attention.num_heads,
        )

        norm = self._norm1(y)
        attention, _ = self._attention(
            norm[:, [-1]],
            norm,
            norm,
            need_weights=False,
            attn_mask=masks,
        )
        obj = obj + attention
        norm = self._norm2(obj)
        mlp = self._mlp(norm)
        obj = obj + mlp

        return super().forward(x), obj


class ExpandedCLIPViT(BaseCLIPViT):
    BLOCK_TYPE = ExpandedCLIPBlock

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._blocks._unpack_args = True

    def upsample(self, ratio: int = 2) -> None:
        patch_w, patch_h = self._patch_wh
        upsampled_patch_wh = (patch_w * ratio, patch_h * ratio)
        position_embedding = self._interpolate_position_embedding(
            upsampled_patch_wh,
        )
        position_embedding = einops.rearrange(
            position_embedding,
            '1 ... -> ...',
        )
        self._position_embedding = nn.Parameter(position_embedding)
        self._patch_wh = upsampled_patch_wh

        self._patch_embedding.stride = tuple(
            s // ratio for s in self._patch_embedding.stride
        )
        # TODO: Check if this is correct
        self._patch_embedding.padding = ((self._patch_size - 1) // 2,
                                         (self._patch_size - 1) // 2)

    def forward(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        image: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        masks = (masks - 1) * 100.
        masks = einops.rearrange(masks, 'b h w -> b (h w)')
        zeros = masks.new_zeros(masks.shape[0], 1)
        masks = torch.cat([masks, zeros], -1)

        x: torch.Tensor = self._patch_embedding(image)

        b, _, h, w = x.shape
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        cls_token = einops.repeat(self._cls_token, 'd -> b 1 d', b=b)
        x = torch.cat((cls_token, x), 1)

        position_embedding = self._interpolate_position_embedding(
            (w, h),
            mode='bilinear',
        )

        x = x + position_embedding
        x = self._norm_pre(x)
        _, obj = self._blocks(x, x[:, [0]], masks=masks)
        obj = self._norm(obj)

        if self._projector is not None:
            obj = obj @ self._projector

        obj = F.normalize(obj, dim=-1)

        obj = einops.rearrange(obj, 'b 1 c -> b c')
        return obj


Transform = Callable[[Image.Image | torch.Tensor], torch.Tensor]


@OAKEModelRegistry.register_()
def clip_vit(
    expand_mask_size: int | None,
    adaptive: bool,
) -> tuple[nn.Module, Transform]:
    model: nn.Module
    if expand_mask_size is None:
        model = CLIPViT(
            patch_size=32,
            patch_wh=(7, 7),
            out_features=512,
        )
        model.load_pretrained('pretrained/clip/ViT-B-32.pt')
    else:
        model = ExpandedCLIPViT(
            patch_size=32,
            patch_wh=(7, 7),
            out_features=512,
        )
        model.load_pretrained('pretrained/clip/ViT-B-32.pt')
        model.upsample()

    if adaptive:
        transforms = tf_v2.Compose([
            tf_v2.ToImage(),
            tf_v2.ToDtype(torch.float32, True),
            tf_v2.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    else:
        transforms = tf_v2.Compose([
            tf_v2.Resize(224, tf_v2.InterpolationMode.BICUBIC),
            tf_v2.CenterCrop(224),
            tf_v2.ToImage(),
            tf_v2.ToDtype(torch.float32, True),
            tf_v2.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    if expand_mask_size is not None:
        transforms = ExpandTransform(
            transforms=transforms,
            mask_size=expand_mask_size,
        )

    return model, transforms
