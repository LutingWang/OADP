__all__ = [
    'TextualCategoryEmbedding',
]

import torch

from ...utils import Globals
from .base import BaseCategoryEmbedding


class TextualCategoryEmbedding(BaseCategoryEmbedding):

    def __init__(self, *args, **kwargs) -> None:
        embeddings = {
            prompt['name']: torch.cat([
                e
                for key in [
                    'definition_encoding',
                    'synonym_encoding',
                    'description_encoding',
                ]
                if (e := prompt[key]) is not None
            ])
            for prompt in torch.load(
                f'work_dirs/prompts/{Globals.categories.name}_clip.pth',
                'cpu',
            )
        }
        super().__init__(*args, embeddings=embeddings, **kwargs)
