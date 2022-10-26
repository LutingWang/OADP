__all__ = [
    'Classifier',
    'ViLDClassifier',
]

from typing import cast, Optional
from mmdet.models.utils.builder import LINEAR_LAYERS
from mmdet.datasets import CocoDataset
import todd
import torch
import torch.nn as nn
import torch.nn.functional as F

import mldec


@LINEAR_LAYERS.register_module()
class Classifier(todd.base.Module):

    def __init__(
        self,
        *args,
        pretrained: str,
        in_features: int,
        out_features: int,
        split: str,
        num_base_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        ckpt = torch.load(pretrained, 'cpu')
        self._scaler = cast(torch.Tensor, ckpt['scaler']).item()
        self._bias = cast(torch.Tensor, ckpt['bias']).item()

        embeddings = cast(torch.Tensor, ckpt['embeddings'])
        name2ind = {name: i for i, name in enumerate(ckpt['names'])}
        inds = [name2ind[name] for name in getattr(mldec, split)]
        embeddings = embeddings[inds]
        self.register_buffer('_embeddings', embeddings, persistent=False)

        embedding_dim = embeddings.shape[1]
        self._linear = nn.Linear(in_features, embedding_dim)
        if embeddings.shape[0] == out_features - 1:
            self._bg_embedding = nn.Parameter(torch.randn(1, embedding_dim))
            nn.init.xavier_uniform_(self._bg_embedding)
        elif embeddings.shape[0] == out_features:
            self._bg_embedding = None
        else:
            assert False, (embeddings.shape[0], out_features)

        self._num_base_classes = num_base_classes

    @property
    def embeddings(self) -> torch.Tensor:
        return cast(torch.Tensor, self._embeddings)

    @property
    def num_classes(self) -> int:
        return self.embeddings.shape[0]

    @property
    def num_base_classes(self) -> int:
        return self._num_base_classes

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = F.normalize(x)
        embeddings = self.embeddings
        if self._bg_embedding is not None:
            embeddings = torch.cat([
                embeddings,
                F.normalize(self._bg_embedding),
            ])
        y = (x @ embeddings.T) * self._scaler - self._bias
        if self._num_base_classes is not None and todd.globals_.training:
            y[:, self._num_base_classes:self.num_classes] = float('-inf')
        return y


@LINEAR_LAYERS.register_module()
class ViLDClassifier(todd.base.Module):

    def __init__(
        self,
        *args,
        pretrained: str,
        in_features: int,
        out_features: int,
        split: str,
        num_base_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        ckpt = torch.load(pretrained, 'cpu')

        embeddings = cast(torch.Tensor, ckpt['embeddings'])
        name2ind = {name: i for i, name in enumerate(ckpt['names'])}
        inds = [name2ind[name] for name in getattr(mldec, split)]
        embeddings = embeddings[inds]
        self.register_buffer('_embeddings', embeddings, persistent=False)

        embedding_dim = embeddings.shape[1]
        self._linear = nn.Linear(in_features, embedding_dim)
        if embeddings.shape[0] == out_features - 1:
            self._bg_embedding = nn.Parameter(torch.randn(1, embedding_dim))
            nn.init.xavier_uniform_(self._bg_embedding)
        elif embeddings.shape[0] == out_features:
            self._bg_embedding = None
        else:
            assert False, (embeddings.shape[0], out_features)

        self._num_base_classes = num_base_classes

    @property
    def embeddings(self) -> torch.Tensor:
        return cast(torch.Tensor, self._embeddings)

    @property
    def num_classes(self) -> int:
        return self.embeddings.shape[0]

    @property
    def num_base_classes(self) -> int:
        return self._num_base_classes

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = F.normalize(x)
        embeddings = self.embeddings
        if self._bg_embedding is not None:
            embeddings = torch.cat([
                embeddings,
                F.normalize(self._bg_embedding),
            ])
        scaler = 0.007 if todd.globals_.training else 0.01  # TODO: check this
        y = (x @ embeddings.T) / scaler
        if self._num_base_classes is not None and todd.globals_.training:
            y[:, self._num_base_classes:self.num_classes] = float('-inf')
        return y
