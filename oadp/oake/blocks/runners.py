__all__ = [
    'BlockValidator',
]

from typing import cast

import clip.model
import todd
import torch.nn.functional as F
import torchvision.transforms as tf
from todd.runners import Memo
from todd.bases.registries import Item

from ..registries import OAKERunnerRegistry
from ..runners import BaseValidator
from .datasets import Batch


@OAKERunnerRegistry.register_()
class BlockValidator(BaseValidator):

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.model, config.transforms = clip.load_default(False)
        return config

    def _run_iter(self, batch: Batch, memo: Memo, *args, **kwargs) -> Memo:
        blocks = batch['blocks']
        bboxes = batch['bboxes']
        if todd.Store.cuda:
            blocks = blocks.cuda()
            bboxes = bboxes.cuda()
        module = cast(clip.model.CLIP, self._strategy.module)
        embeddings = module.encode_image(blocks)
        embeddings = F.normalize(embeddings)
        memo['output'] = dict(
            embeddings=embeddings.half(),
            bboxes=bboxes.half(),
        )
        return memo
