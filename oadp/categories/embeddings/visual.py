__all__ = [
    'VisualCategoryEmbedding',
]

import torch

from ...utils import Globals
from .base import BaseCategoryEmbedding


class VisualCategoryEmbedding(BaseCategoryEmbedding):

    def __init__(self, *args, **kwargs) -> None:
        embeddings: dict[str, torch.Tensor] = torch.load(
            f'work_dirs/visual_category_embeddings/{Globals.categories.name}.pth',
            'cpu',
        )
        super().__init__(*args, embeddings=embeddings, **kwargs)
