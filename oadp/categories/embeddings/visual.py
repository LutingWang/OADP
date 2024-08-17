__all__ = [
    'VisualCategoryEmbedding',
]

import random
import torch
from torch import nn
from oadp.utils import Globals


class VisualCategoryEmbedding(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        categories = Globals.categories
        embeddings: dict[str, torch.Tensor] = torch.load(
            f'work_dirs/visual_category_embeddings/{categories.name}.pth',
            'cpu',
        )
        for category in categories.all_:
            self.register_buffer(
                self._buffer_name(category),
                embeddings[category],
            )

    def _buffer_name(self, category: str) -> str:
        return f'_embeddings_{category}'

    def get_embeddings(self, category: str) -> torch.Tensor:
        buffer_name = self._buffer_name(category)
        return self.get_buffer(buffer_name)

    def sample_embedding(self, category: str) -> torch.Tensor:
        embeddings = self.get_embeddings(category)
        i = random.randrange(embeddings.shape[0])
        return embeddings[i]

    def forward(self) -> torch.Tensor:
        categories = Globals.categories
        embeddings = torch.stack([
            self.sample_embedding(category) for category in categories.all_
        ])
        return embeddings
