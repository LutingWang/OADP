from mmdet.models import Shared2FCBBoxHead
from mmdet.models.utils.builder import LINEAR_LAYERS
import todd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import MMDET_COCO as CLASSES


@LINEAR_LAYERS.register_module()
class Classifier(todd.base.Module):

    def __init__(self, *args, pretrained: str, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pretrained = torch.load(pretrained, 'cpu')
        embeddings: torch.Tensor = pretrained['embeddings']
        embedding_dim = embeddings.shape[1]
        self._linear = nn.Linear(in_features, embedding_dim)
        name2ind = {name: i for i, name in enumerate(pretrained['names'])}
        inds = [name2ind[name] for name in CLASSES]
        embeddings = embeddings[inds]
        assert embeddings.shape[0] == out_features - 1
        self.register_buffer('_embeddings', embeddings)
        self._bg_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        self._scaler = pretrained['scaler'].item()
        self._bias = pretrained['bias'].item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = F.normalize(x)
        embeddings = torch.cat([
            self.get_buffer('_embeddings'),
            F.normalize(self._bg_embedding),
        ])
        return (x @ embeddings.T) * self._scaler - self._bias
