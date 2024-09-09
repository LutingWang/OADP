__all__ = [
    'ObjectValidator',
]

from typing import TYPE_CHECKING, TypedDict

import todd
import todd.tasks.object_detection as od
import torch
from todd.bases.registries import Item
from todd.runners import Memo

from ..models import ExpandTransform
from ..registries import OAKEModelRegistry, OAKERunnerRegistry
from ..runners import BaseValidator

if TYPE_CHECKING:
    from .datasets import Batch, ObjectDataset


class Output(TypedDict):
    tensors: torch.Tensor
    bboxes: od.FlattenBBoxesXYWH
    categories: torch.Tensor


@OAKERunnerRegistry.register_()
class ObjectValidator(BaseValidator):
    _dataset: 'ObjectDataset'

    def __init__(
        self,
        *args,
        expand_transform: ExpandTransform,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._expand_transform = expand_transform

        categories = self._dataset.categories
        torch.save(categories, self._work_dir / 'categories.pth')

    @property
    def expand_transform(self) -> ExpandTransform:
        return self._expand_transform

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.model, config.expand_transform = OAKEModelRegistry.build(
            config.model,
        )
        config.transforms = None
        return config

    def _run_iter(
        self,
        batch: 'Batch | None',
        memo: Memo,
        *args,
        **kwargs,
    ) -> Memo:
        if batch is None:
            memo['output'] = None
        crops = batch['crops']
        masks = batch['masks']
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            crops = crops.cuda()
            masks = masks.cuda()
        tensors: torch.Tensor = self.model(crops, masks)
        memo['output'] = dict(
            tensors=tensors.half(),
            bboxes=batch['bboxes'],
            categories=batch['categories'],
        )
        return memo
