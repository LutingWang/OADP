__all__ = [
    'BaseClassifier',
    'Classifier',
    'ViLDClassifier',
]

from typing import TypedDict

import todd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils.builder import LINEAR_LAYERS

from ..base import Globals
from .utils import NormalizedLinear


@LINEAR_LAYERS.register_module()
class BaseClassifier(todd.Module):

    def __init__(
        self,
        *args,
        prompts: str,
        in_features: int,
        out_features: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        prompts_ = torch.load(prompts, 'cpu')
        names: list[str] = prompts_['names']
        embeddings: torch.Tensor = prompts_['embeddings']
        indices = [names.index(name) for name in Globals.categories.all_]
        embeddings = embeddings[indices]

        if out_features == Globals.categories.num_all + 1:
            # with background embedding
            bg_embedding = nn.Parameter(torch.zeros(1, embeddings.shape[1]))
            nn.init.xavier_uniform_(bg_embedding)
        elif out_features == Globals.categories.num_all:
            bg_embedding = None
        else:
            raise RuntimeError(str(out_features))

        self._prompts = prompts_
        self.register_buffer('_embeddings', embeddings, persistent=False)
        self._bg_embedding = bg_embedding
        self._linear = NormalizedLinear(in_features, embeddings.shape[1])

    @property
    def embeddings(self) -> torch.Tensor:
        embeddings: torch.Tensor = self._embeddings
        if self._bg_embedding is None:
            return embeddings
        bg_embedding = F.normalize(self._bg_embedding)
        return torch.cat([embeddings, bg_embedding])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        y = x @ self.embeddings.T
        if Globals.training:
            novel_categories = slice(
                Globals.categories.num_bases,
                Globals.categories.num_all,
            )
            y[:, novel_categories] = float('-inf')
        return y


@LINEAR_LAYERS.register_module()
class Classifier(BaseClassifier):
    # named `Classifier` to monkey patch mmdet detectors

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        scaler: torch.Tensor = self._prompts['scaler']
        bias: torch.Tensor = self._prompts['bias']
        self._scaler = scaler.item()
        self._bias = bias.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self._scaler - self._bias


class ViLDScaler(TypedDict):
    train: float
    val: float


@LINEAR_LAYERS.register_module()
class ViLDClassifier(BaseClassifier):

    def __init__(
        self,
        *args,
        scaler: ViLDScaler | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if scaler is None:
            scaler = dict(train=0.007, val=0.01)  # inverse of DetPro
        self._scaler = scaler

    @property
    def scaler(self) -> float:
        return (
            self._scaler['train'] if Globals.training else self._scaler['val']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) / self.scaler
