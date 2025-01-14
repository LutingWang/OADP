__all__ = [
    'BlockValidator',
]

from typing import TypedDict

import todd
import todd.tasks.object_detection as od
import torch
from todd.bases.registries import Item
from todd.runners import Memo

from ..registries import OAKEModelRegistry, OAKERunnerRegistry
from ..runners import BaseValidator
from .datasets import Batch


class Output(TypedDict):
    embeddings: torch.Tensor
    bboxes: od.FlattenBBoxesXYXY


@OAKERunnerRegistry.register_()
class BlockValidator(BaseValidator):

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.model, config.transforms = OAKEModelRegistry.build(config.model)
        return config

    def _run_iter(self, batch: Batch, memo: Memo, *args, **kwargs) -> Memo:
        blocks = batch['blocks']
        if todd.Store.cuda:
            blocks = blocks.cuda()
        embeddings: torch.Tensor = self.model(blocks)
        memo['output'] = dict(
            embeddings=embeddings.half(),
            bboxes=batch['bboxes'],
        )
        return memo
