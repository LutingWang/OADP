__all__ = [
    'BaseExpander',
    'LongestExpander',
    'ConstantExpander',
    'AdaptiveExpander',
    'ExpandTransform',
]

from abc import ABC, abstractmethod

import einops
import todd.tasks.object_detection as od
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from PIL import Image


class BaseExpander(ABC):

    @abstractmethod
    def _expand_size(self, bboxes: od.FlattenBBoxesMixin) -> torch.Tensor:
        pass

    def _expand(
        self,
        bboxes: od.FlattenBBoxesMixin,
        image_wh: tuple[int, int],
    ) -> od.FlattenBBoxesCXCYWH:
        expand_size = self._expand_size(bboxes)
        expand_size = expand_size.clamp_max(min(image_wh))
        expand_size = expand_size.clamp_min(bboxes.width)
        expand_size = expand_size.clamp_min(bboxes.height)
        expand_size = einops.repeat(expand_size, 'n -> n d', d=2)
        expanded_bboxes = torch.cat([bboxes.center, expand_size], -1)
        return bboxes.to(od.FlattenBBoxesCXCYWH).copy(expanded_bboxes)

    def _translate(
        self,
        bboxes: od.FlattenBBoxesCXCYWH,
        image_wh: tuple[int, int],
    ) -> od.FlattenBBoxesCXCYWH:
        image_wh_ = torch.tensor(image_wh)
        offset = torch.zeros_like(bboxes.lt)
        offset = torch.where(
            bboxes.lt >= 0,
            offset,
            -bboxes.lt,
        )
        offset = torch.where(
            bboxes.rb <= image_wh_,
            offset,
            image_wh_ - bboxes.rb,
        )
        offset = torch.where(
            bboxes.wh <= image_wh_,
            offset,
            torch.tensor(0.0),
        )
        return bboxes.translate(offset)

    def __call__(
        self,
        bboxes: od.FlattenBBoxesMixin,
        image_wh: tuple[int, int],
    ) -> od.FlattenBBoxesCXCYWH:
        assert not bboxes.normalized
        expanded_bboxes = self._expand(bboxes, image_wh)
        return self._translate(expanded_bboxes, image_wh)


class LongestExpander(BaseExpander):

    def _expand_size(self, bboxes: od.FlattenBBoxesMixin) -> torch.Tensor:
        expand_size, _ = bboxes.wh.max(-1)
        return expand_size


class ConstantExpander(BaseExpander):

    def __init__(self, constant: int = 224) -> None:
        self._constant = constant

    def _expand_size(self, bboxes: od.FlattenBBoxesMixin) -> torch.Tensor:
        return torch.full((len(bboxes), ), self._constant)


class AdaptiveExpander(BaseExpander):

    def __init__(self, ratio: float = 8) -> None:
        self._ratio = ratio

    def _expand_size(self, bboxes: od.FlattenBBoxesMixin) -> torch.Tensor:
        area = bboxes.area * self._ratio
        return area.sqrt()


class ExpandTransform:

    def __init__(
        self,
        *args,
        expander: BaseExpander | None = None,
        transforms: tf.Compose,
        mask_size: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if expander is None:
            expander = AdaptiveExpander()

        self._expander = expander
        self._transforms = transforms
        self._mask_size = mask_size

    def _crop(
        self,
        image: Image.Image,
        bboxes: od.FlattenBBoxesMixin,
    ) -> list[Image.Image]:
        bboxes = bboxes.to(od.FlattenBBoxesXYXY)
        return [image.crop(bbox) for bbox in bboxes.to_tensor().int().tolist()]

    def _transform(self, crops: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self._transforms(crop) for crop in crops])

    def _mask(
        self,
        bboxes: od.FlattenBBoxesMixin,
        expanded_bboxes: od.FlattenBBoxesMixin,
        image_wh: tuple[int, int],
    ) -> torch.Tensor:
        scale = torch.tensor(image_wh) / expanded_bboxes.wh
        scale = einops.repeat(scale, 'n wh -> n (two wh)', two=2)

        bboxes = bboxes.to(od.FlattenBBoxesXYXY)
        bboxes = bboxes.translate(-expanded_bboxes.lt)
        bboxes = od.FlattenBBoxesXYXY(
            bboxes.to_tensor() * scale,
            divisor=image_wh,
        )

        masks = bboxes.to_mask().float()
        masks = einops.rearrange(masks, 'b h w -> b 1 h w')

        masks = F.interpolate(
            masks,
            (self._mask_size, self._mask_size),
            mode='bilinear',
        )
        masks = einops.rearrange(masks, 'b 1 h w -> b h w')
        return masks

    def __call__(
        self,
        image: Image.Image,
        bboxes: od.FlattenBBoxesMixin,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_bboxes = self._expander(bboxes, image.size)
        crops = self._crop(image, expanded_bboxes)
        tensor = self._transform(crops)
        _, _, h, w = tensor.shape
        masks = self._mask(bboxes, expanded_bboxes, (w, h))
        return tensor, masks
