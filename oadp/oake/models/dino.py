__all__ = [
    'DINOv2',
    'ExpandedDINOv2',
    'dinov2',
]

from typing import Callable
import einops
import torchvision.transforms.v2 as tf_v2
import torch
from torch import nn
from todd.datasets import IMAGENET_MEAN, IMAGENET_STD
from todd.models.modules import DINOv2 as BaseDINOv2
from todd.models.modules.dino import DINOv2Block
from PIL import Image

from ..registries import OAKEModelRegistry

from .expanders import ExpandTransform


class DINOv2(BaseDINOv2):

    def forward(  # type: ignore[override] # pylint: disable=arguments-differ
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        cls_, _ = super().forward(image, False)
        return cls_


class ExpandedDINOv2Block(DINOv2Block):

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
        obj = obj + self._scaler1(attention)
        norm = self._norm2(obj)
        mlp = self._mlp(norm)
        obj = obj + self._scaler2(mlp)

        return super().forward(x), obj


class ExpandedDINOv2(BaseDINOv2):
    BLOCK_TYPE = ExpandedDINOv2Block

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._blocks._unpack_args = True

    def forward(  # type: ignore[override] # noqa: E501 pylint: disable=arguments-differ,arguments-renamed
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
            mode='bicubic',
        )

        x = x + position_embedding
        _, obj = self._blocks(x, x[:, [0]], masks=masks)
        obj = self._norm(obj)

        obj = einops.rearrange(obj, 'b 1 c -> b c')
        return obj


Transform = Callable[[Image.Image | torch.Tensor], torch.Tensor]


@OAKEModelRegistry.register_()
def dinov2(
    expand_mask_size: int | None,
    adaptive: bool,
) -> tuple[nn.Module, Transform]:
    model_type = DINOv2 if expand_mask_size is None else ExpandedDINOv2
    model = model_type(
        patch_size=14,
        patch_wh=(37, 37),
        width=1024,
        depth=24,
        num_heads=16,
    )
    model.load_pretrained('pretrained/dino/dinov2_vitl14_pretrain.pth')

    if adaptive:
        transforms = tf_v2.Compose([
            tf_v2.ToImage(),
            tf_v2.ToDtype(torch.float32, True),
            tf_v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        transforms = tf_v2.Compose([
            tf_v2.Resize(224, tf_v2.InterpolationMode.BICUBIC),
            tf_v2.CenterCrop(224),
            tf_v2.ToImage(),
            tf_v2.ToDtype(torch.float32, True),
            tf_v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    if expand_mask_size is not None:
        transforms = ExpandTransform(
            transforms=transforms,
            mask_size=expand_mask_size,
        )

    return model, transforms
