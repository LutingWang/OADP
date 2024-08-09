# mypy: disable-error-code="override"
# pylint: disable=arguments-differ

import pathlib
from typing import Generic, TypeVar

import clip
import clip.model
import todd
import torchvision.transforms as tf
from todd.bases.registries import Item

from ..datasets import BaseDataset
from ..registries import OAKEDatasetRegistry, OAKERunnerRegistry

from torch import nn

T = TypeVar('T', bound=nn.Module)


@OAKERunnerRegistry.register_()
class BaseValidator(todd.runners.Validator[T]):

    def __init__(self, *args, output_dir: pathlib.Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._output_dir = output_dir

    def output_path(self, key: str) -> pathlib.Path:
        return self._output_dir / f'{key}.pth'

    @classmethod
    def output_dir_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        output_dir: pathlib.Path = (
            config.work_dir / 'output' / config.output_dir.task_name
        )
        output_dir.mkdir(exist_ok=True, parents=True)
        config.output_dir = output_dir
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = super().build_pre_hook(config, registry, item)
        config = cls.output_dir_build_pre_hook(config, registry, item)
        return config
