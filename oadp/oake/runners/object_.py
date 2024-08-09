__all__ = [
    'ObjectValidator',
]

import math
from typing import Any, TypeVar, cast

import clip
import clip.model
import einops
import todd
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from todd.runners import Memo
from torch import nn
from todd.bases.registries import Item
from todd.utils import NestedCollectionUtils

from oadp.expanded_clip import ExpandedCLIP
from oadp.expanded_clip import load_default
from oadp.expanded_clip import ExpandTransform
from ..datasets import ObjectDataset
from ..datasets.object_ import T

from ..registries import OAKERunnerRegistry, OAKEDatasetRegistry
from .base import BaseValidator

ModuleType = TypeVar('ModuleType', bound=nn.Module)


@OAKERunnerRegistry.register_()
class ObjectValidator(BaseValidator[ModuleType]):
    _model: ExpandedCLIP
    _dataset: ObjectDataset

    def __init__(self, *args, mini_batch_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mini_batch_size = mini_batch_size

    @classmethod
    def dataset_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        expand_transform = config.pop('expand_transform')
        config.dataset = OAKEDatasetRegistry.build(
            config.dataset,
            expand_transform=expand_transform,
        )
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        expanded_clip_model, expanded_clip_transforms = load_default()
        config.model = expanded_clip_model

        expand_transform = ExpandTransform(
            transforms=expanded_clip_transforms,
            mask_size=expanded_clip_model.model.visual.grid,
        )
        config.expand_transform = expand_transform

        config = super().build_pre_hook(config, registry, item)
        return config

    def _init_dataset(self, *args, **kwargs) -> None:
        self._dataset.bind(self)

    def _init(self, *args, **kwargs) -> None:
        super()._init(*args, **kwargs)
        self._init_dataset(*args, **kwargs)

    def _run_iter(self, batch: T, memo: Memo, *args, **kwargs) -> Memo:
        crops = batch.crops
        masks = batch.masks
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            crops = crops.cuda()
            masks = masks.cuda()
        tensors = []
        for i in range(
            math.ceil(crops.shape[0] / self._mini_batch_size),
        ):
            indices = slice(
                i * self._mini_batch_size,
                (i + 1) * self._mini_batch_size,
            )
            tensor = self._model(crops[indices], masks[indices])
            tensors.append(tensor)
        memo['output'] = dict(
            tensors=torch.cat(tensors).half(),
            proposals=batch.proposals.half(),
            objectness=batch.objectness.half()
        )
        breakpoint()
        return memo
