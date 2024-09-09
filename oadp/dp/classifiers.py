__all__ = [
    'BaseClassifier',
    'Classifier',
    'ViLDClassifier',
]

from typing import TypedDict

import torch
import torch.nn.functional as F
from mmdet.registry import MODELS
from torch import nn

from ..categories.embeddings import TextualCategoryEmbedding
from ..utils import Globals
from .utils import NormalizedLinear


@MODELS.register_module()
class BaseClassifier(nn.Module):

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


@MODELS.register_module()
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


@MODELS.register_module()
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


@MODELS.register_module()
class FewShotClassifier(nn.Module):

    def __init__(
        self,
        *args,
        hidden_features: int = 512,
        in_features: int,
        out_features: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if out_features == Globals.categories.num_all + 1:
            # with background embedding
            bg_embedding = nn.Parameter(torch.zeros(1, hidden_features))
            nn.init.xavier_uniform_(bg_embedding)
        elif out_features == Globals.categories.num_all:
            bg_embedding = None
        else:
            raise RuntimeError(str(out_features))

        self._bg_embedding = bg_embedding
        self._linear = NormalizedLinear(in_features, hidden_features)

        self._scaler = dict(train=0.007, val=0.01)  # inverse of DetPro

    @property
    def embeddings(self) -> torch.Tensor:
        embeddings = Globals.visual_embeddings
        if self._bg_embedding is None:
            return embeddings
        bg_embedding = F.normalize(self._bg_embedding)
        return torch.cat([embeddings, bg_embedding])

    @property
    def scaler(self) -> float:
        return (
            self._scaler['train'] if Globals.training else self._scaler['val']
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        y = x @ self.embeddings.T
        return y / self.scaler


class Scaler(TypedDict):
    train: float
    val: float


@MODELS.register_module()
class OVClassifier(nn.Module):

    def __init__(
        self,
        *args,
        scaler: Scaler,
        in_features: int,
        out_features: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._scaler = scaler

        textual_category_embedding = TextualCategoryEmbedding()
        self._textual_category_embedding = textual_category_embedding

        embedding_dim = textual_category_embedding.embedding_dim
        self._linear = nn.Linear(in_features, embedding_dim)

        if out_features == Globals.categories.num_all + 1:
            bg_embedding = nn.Parameter(torch.zeros(1, embedding_dim))
            nn.init.xavier_uniform_(bg_embedding)
        elif out_features == Globals.categories.num_all:
            bg_embedding = None
        else:
            raise RuntimeError(
                f"Unexpected {out_features=} given "
                f"{Globals.categories.num_all=}",
            )
        self._bg_embedding = bg_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x)
        x = F.normalize(x)

        embeddings: torch.Tensor = self._textual_category_embedding()
        if self._bg_embedding is not None:
            # TODO: is normalization needed for t5?
            bg_embedding = F.normalize(self._bg_embedding)
            embeddings = torch.cat([embeddings, bg_embedding])

        logits = x @ embeddings.T

        if Globals.training:
            logits = logits / self._scaler['train']
            novel_categories = slice(
                Globals.categories.num_bases,
                Globals.categories.num_all,
            )
            logits[:, novel_categories] = float('-inf')
        else:
            logits = logits / self._scaler['val']

        return logits
