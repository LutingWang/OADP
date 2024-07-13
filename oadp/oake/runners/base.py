# type: ignore[override]
# pylint: disable=arguments-differ

import pathlib
from typing import Generic, TypeVar

import clip
import clip.model
import todd
import torchvision.transforms as tf

from ..datasets import BaseDataset
from ..registries import OADPDatasetRegistry, OADPRunnerRegistry

T = TypeVar('T')


@OADPRunnerRegistry.register_()
class BaseValidator(todd.runners.Validator, Generic[T]):

    def output_path(self, id_: int) -> pathlib.Path:
        return self._output_dir / f'{id_:012d}.pth'

    def _build_output_dir(
        self,
        *args,
        output_dir: todd.Config,
        **kwargs,
    ) -> None:
        self._output_dir: pathlib.Path = (
            self._work_dir / 'output' / output_dir.task_name
        )
        self._output_dir.mkdir(exist_ok=True, parents=True)

    def _build(self, *args, clip_: todd.Config, **kwargs) -> None:
        model, transforms = clip.load_default(**clip_)
        model.requires_grad_(False)
        super()._build(*args, model=model, transforms=transforms, **kwargs)
        self._build_output_dir(*args, **kwargs)

    def _build_dataloader(
        self,
        *args,
        dataloader: todd.Config,
        **kwargs,
    ) -> None:
        if todd.Store.DRY_RUN:
            dataloader.batch_size = 1
            dataloader.num_workers = 0
        super()._build_dataloader(*args, dataloader=dataloader, **kwargs)

    def _build_model(
        self,
        *args,
        model: clip.model.CLIP,
        map_model: todd.Config | None = None,
        **kwargs,
    ) -> None:
        if map_model is None:
            map_model = todd.Config()
        self._model = self._strategy.map_model(model, map_model)

    def _build_dataset(
        self,
        *args,
        dataset: todd.Config,
        transforms: tf.Compose,
        **kwargs,
    ) -> None:
        self._dataset: BaseDataset = OADPDatasetRegistry.build(
            dataset,
            transforms=transforms,
            runner=self,
        )
