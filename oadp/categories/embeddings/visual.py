__all__ = [
    'VisualCategoryEmbedding',
]

import torch

from ...utils import Globals
from .base import BaseCategoryEmbedding


class VisualCategoryEmbedding(BaseCategoryEmbedding):

    def __init__(self, *args, model: str, **kwargs) -> None:
        embeddings: dict[str, torch.Tensor] = torch.load(
            'work_dirs/visual_category_embeddings/'
            f'{Globals.categories.name}_{model}.pth',
            'cpu',
        )
        super().__init__(*args, embeddings=embeddings, **kwargs)
