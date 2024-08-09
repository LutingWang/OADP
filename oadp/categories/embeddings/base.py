__all__ = [
    'BaseEmbedding',
]

import pathlib
from abc import ABC, abstractmethod
import random
from typing import Any, Generic, Iterable, TypeVar

import todd
import torch
from todd.bases.registries import BuildPreHookMixin, Item
from torch import nn

from ..constants import Categories
from ..loaders import BaseLoader, OADPCategoryLoaderRegistry

T = TypeVar('T')


class BaseEmbedding(BuildPreHookMixin, nn.Module, Generic[T], ABC):

    def __init__(
        self,
        *args,
        loader: BaseLoader[T],
        cache_dir: pathlib.Path | Any = 'work_dirs/cache',
        cache_limit: int = 10_000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._loader = loader

        if not isinstance(cache_dir, pathlib.Path):
            cache_dir = pathlib.Path(cache_dir)
        cache_dir = cache_dir / self.__class__.__name__
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_dir

        self._cache_limit = cache_limit

    @classmethod
    def loader_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.loader = OADPCategoryLoaderRegistry.build_or_return(
            config.loader,
        )
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.loader_build_pre_hook(config, registry, item)
        return config

    def _load_cache(
        self,
        prefix: str,
        category_name: str,
    ) -> torch.Tensor | pathlib.Path:
        cache_dir = self._cache_dir / prefix / category_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f'{random.randrange(self._cache_limit)}.pth'
        if not cache.exists():
            return cache
        try:
            embedding: torch.Tensor = torch.load(cache, 'cpu')
        except Exception as e:
            todd.logger.warning("Invalidating cache %s: %s", cache, e)
            cache.unlink()
            return cache
        if todd.Store.cuda:
            embedding = embedding.cuda()
        return embedding

    @abstractmethod
    def encode(self, category_names: Iterable[str]) -> Iterable[torch.Tensor]:
        pass

    def forward(self, categories: Categories) -> torch.Tensor:
        cache = {
            category_name: self._load_cache(categories.name, category_name)
            for category_name in categories.all_
        }
        category_names = [
            category_name for category_name, embedding in cache.items()
            if not isinstance(embedding, torch.Tensor)
        ]
        embeddings = ([] if len(category_names) == 0 else
                      self.encode(category_names))
        for category_name, embedding in zip(category_names, embeddings):
            todd.logger.debug("Caching %s", cache[category_name])
            torch.save(embedding, cache[category_name])
            cache[category_name] = embedding
        embeddings = [
            cache[category_name] for category_name in categories.all_
        ]
        return torch.stack(embeddings).float()
