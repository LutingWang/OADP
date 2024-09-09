__all__ = [
    'BasePrompter',
]

from abc import ABC, abstractmethod
from typing import Any

import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item

from ..models import BaseModel
from ..registries import PromptModelRegistry


class BasePrompter(BuildPreHookMixin, ABC):

    def __init__(self, *args, model: BaseModel, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if todd.Store.cuda:
            model = model.cuda()
        self._model = model

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.model = PromptModelRegistry.build_or_return(config.model)
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.model_build_pre_hook(config, registry, item)
        return config

    @abstractmethod
    def load(self) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def _prompt(self, category: dict[str, Any]) -> dict[str, Any]:
        pass

    @torch.no_grad()
    def __call__(self, category: dict[str, Any]) -> dict[str, Any]:
        return category | self._prompt(category)
