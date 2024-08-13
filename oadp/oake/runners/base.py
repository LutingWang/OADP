__all__ = [
    'BaseValidator',
]

import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

import clip.model
import todd
import torchvision.transforms as tf
from todd.bases.registries import Item
from torch import nn

from ..registries import OAKERunnerRegistry

if TYPE_CHECKING:
    from ..datasets import BaseDataset

T = TypeVar('T')


@OAKERunnerRegistry.register_()
class BaseValidator(todd.runners.Validator[nn.Module], ABC):
    _dataset: 'BaseDataset[T]'

    def __init__(self, *args, transforms: tf.Compose, **kwargs) -> None:
        self._transforms = transforms
        super().__init__(*args, **kwargs)

    @property
    def transforms(self) -> tf.Compose:
        return self._transforms

    @property
    def output_dir(self) -> pathlib.Path:
        return self._work_dir / 'output'

    def output_path(self, id_: str) -> pathlib.Path:
        return self.output_dir / f'{id_}.pth'

    @classmethod
    @abstractmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        pass

    def _init_work_dir(self, *args, **kwargs) -> None:
        super()._init_work_dir(*args, **kwargs)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_dataset(self, *args, **kwargs) -> None:
        self._dataset.bind(self)

    def _init(self, *args, **kwargs) -> None:
        super()._init(*args, **kwargs)
        self._init_dataset(*args, **kwargs)
