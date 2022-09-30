__all__ = [
    'Classifier',
]

from mmdet.models.utils.builder import LINEAR_LAYERS
from mmdet.datasets import CocoDataset
import todd
import torch
import torch.nn as nn
import torch.nn.functional as F


@LINEAR_LAYERS.register_module()
class Classifier(todd.base.Module):

    def __init__(self, *args, pretrained: str, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ckpt = torch.load(pretrained, 'cpu')
        embeddings: torch.Tensor = ckpt['embeddings']
        name2ind = {name: i for i, name in enumerate(ckpt['names'])}
        inds = [name2ind[name] for name in CocoDataset.CLASSES]
        self.register_buffer('_embeddings', embeddings[inds])
        self._scaler = ckpt['scaler'].item()
        self._bias = ckpt['bias'].item()

        embedding_dim = embeddings.shape[1]
        self._linear = nn.Linear(in_features, embedding_dim)
        if embeddings.shape[0] == out_features - 1:
            self._bg_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        elif embeddings.shape[0] == out_features:
            self._bg_embedding = None
        else:
            assert False, (embeddings.shape[0], out_features)

    @property
    def num_classes(self) -> int:
        return self._embeddings.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self._embeddings.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = F.normalize(x)
        embeddings = self._embeddings
        if self._bg_embedding is not None:
            embeddings = torch.cat([
                embeddings,
                F.normalize(self._bg_embedding),
            ])
        return (x @ embeddings.T) * self._scaler - self._bias
