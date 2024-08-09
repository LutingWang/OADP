__all__ = [
    'VisualEmbedding',
]

import random
from typing import Iterable

import einops
import todd
import torch

from ...expanded_clip import ExpandTransform, load_default
from ..embeddings import BaseEmbedding
from ..embeddings import OADPCategoryEmbeddingRegistry
from .loaders import T, VisualLoader


@OADPCategoryEmbeddingRegistry.register_()
class VisualEmbedding(BaseEmbedding[T]):
    _loader: VisualLoader

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model, transforms = load_default()
        self._model = model
        self._expand_transform = ExpandTransform(
            transforms=transforms,
            mask_size=14,
        )

    def encode(self, category_names: Iterable[str]) -> Iterable[torch.Tensor]:
        image_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        for category_name in category_names:
            category = self._loader(category_name)
            images, masks = self._expand_transform(*category)
            assert images.shape[0] == masks.shape[0] == 1
            image_list.append(images)
            mask_list.append(masks)
        images = torch.cat(image_list)
        masks = torch.cat(mask_list)
        if todd.Store.cuda:
            images = images.cuda()
            masks = masks.cuda()
        return self._model(images, masks)
