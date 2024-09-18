__all__ = [
    'BaseCategoryEmbedding',
]

import random

import torch
from torch import nn
import torch.nn.functional as F

from ...utils import Globals


class BaseCategoryEmbedding(nn.Module):

    def __init__(
        self,
        *args,
        embeddings: dict[str, torch.Tensor],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        for category in Globals.categories.all_:
            self.register_buffer(
                self._buffer_name(category),
                embeddings[category],
                False,
            )

    @property
    def embedding_dim(self) -> int:
        embedding_dim, = {
            embedding.shape[1]
            for embedding in self.get_embeddings()
        }
        return embedding_dim

    def _buffer_name(self, category: str) -> str:
        return f'_embedding_{category}'

    def get_embedding(self, category: str) -> torch.Tensor:
        buffer_name = self._buffer_name(category)
        return self.get_buffer(buffer_name)

    def get_embeddings(self) -> list[torch.Tensor]:
        return [
            self.get_embedding(category)
            for category in Globals.categories.all_
        ]

    def forward(self) -> torch.Tensor:
        embeddings = self.get_embeddings()

        # embeddings = [
        #     embedding[random.randrange(embedding.shape[0])]
        #     for embedding in embeddings
        # ]
        # return torch.stack(embeddings)

        embeddings = [embedding.mean(0) for embedding in embeddings]
        return F.normalize(torch.stack(embeddings))
