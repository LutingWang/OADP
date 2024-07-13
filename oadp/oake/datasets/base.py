__all__ = [
    'BaseDataset',
]

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import todd
import torch.distributed
import torch.utils.data.distributed
import torchvision.transforms as tf
from PIL import Image
from todd.bases.registries import BuildPreHookMixin
from todd.runners.utils import RunnerHolderMixin
from torchvision.datasets import CocoDetection

if TYPE_CHECKING:
    from ..runners import BaseValidator

T = TypeVar('T')


class BaseDataset(RunnerHolderMixin, BuildPreHookMixin, ABC, Generic[T]):
    runner: 'BaseValidator[T]'

    def __init__(
        self,
        *args,
        access_layer: todd.Config,
        keys: todd.Config,
        transforms: tf.Compose,
        check: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dataset = CocoDetection(  # TODO: refactor
            os.path.join(access_layer.data_root, access_layer.task_name),
            keys.annotation_file,
        )
        self._access_layer = access_layer
        self._transforms = transforms
        self._check = check

    def __len__(self) -> int:
        return len(self._dataset)

    def _load_image(self, id_: int) -> Image.Image:  # TODO: refactor
        return self._dataset._load_image(id_)

    def __getitem__(self, index: int) -> T | None:
        id_ = self._dataset.ids[index]
        output_path = self.runner.output_path(id_)
        if output_path.exists():
            if not self._check:
                return None
            try:
                torch.load(output_path, 'cpu')
                return None
            except Exception:
                todd.logger.info("Fixing %s", output_path)
        image = self._load_image(id_)
        return self._preprocess(id_, image)

    @abstractmethod
    def _preprocess(
        self,
        id_: int,
        image: Image.Image,
    ) -> T:
        pass
