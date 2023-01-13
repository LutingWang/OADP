__all__ = [
    'Classifier',
    'ViLDClassifier',
]

from typing import Dict, List, Literal, Optional, cast

import todd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils.builder import LINEAR_LAYERS

from ..base import Globals


@LINEAR_LAYERS.register_module()
class BaseClassifier(todd.base.Module):

    def __init__(
        self,
        *args,
        pretrained: str,
        in_features: int,
        out_features: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        ckpt = torch.load(pretrained, 'cpu')
        embeddings: torch.Tensor = ckpt['embeddings']
        names: List[str] = ckpt['names']

        name2ind = {name: i for i, name in enumerate(names)}
        inds = [name2ind[name] for name in Globals.categories.all_]
        embeddings = embeddings[inds]

        num_embeddings, embedding_dim = embeddings.shape
        assert Globals.categories.num_all == num_embeddings

        if num_embeddings == out_features - 1:
            bg_embedding = nn.Parameter(torch.randn(1, embedding_dim) * 0.1)
            nn.init.xavier_uniform_(bg_embedding)
        elif num_embeddings == out_features:
            bg_embedding = None
        else:
            assert False, (num_embeddings, out_features)

        linear = nn.Linear(in_features, embedding_dim)

        self._ckpt = ckpt
        self.register_buffer('_embeddings', embeddings, persistent=False)
        self._bg_embedding = bg_embedding
        self._linear = linear

    @property
    def embeddings(self) -> torch.Tensor:
        return self._embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = F.normalize(x)
        embeddings = self.embeddings
        if self._bg_embedding is not None:
            embeddings = torch.cat([
                embeddings,
                F.normalize(self._bg_embedding),
            ])
        y = x @ embeddings.T
        if Globals.training:
            novel_classes = slice(
                Globals.categories.num_bases,
                Globals.categories.num_all,
            )
            y[:, novel_classes] = float('-inf')
        return y


@LINEAR_LAYERS.register_module()
class Classifier(BaseClassifier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scaler = cast(torch.Tensor, self._ckpt['scaler']).item()
        self._bias = cast(torch.Tensor, self._ckpt['bias']).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self._scaler - self._bias


@LINEAR_LAYERS.register_module()
class ViLDClassifier(BaseClassifier):

    def __init__(
        self,
        *args,
        scaler: Optional[Dict[Literal['train', 'val'], float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if scaler is None:
            scaler = dict(train=0.007, val=0.01)  # inverse of DetPro
        self._scaler = scaler

    @property
    def scaler(self) -> float:
        return self._scaler['train' if Globals.training else 'val']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) / self.scaler
