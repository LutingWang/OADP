__all__ = [
    'GlobalValidator',
]

from typing import cast

import clip.model
import einops
import todd
import torch.nn.functional as F
from todd.runners import Memo
from todd.bases.registries import Item

from .datasets import Batch
from ..registries import OAKERunnerRegistry
from ..runners import BaseValidator


@OAKERunnerRegistry.register_()
class GlobalValidator(BaseValidator):

    @classmethod
    def model_build_pre_hook(
        cls,
        config: todd.Config,
        registry: todd.RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.model, config.transforms = clip.load_default(True)
        return config

    def _run_iter(self, batch: Batch, memo: Memo, *args, **kwargs) -> Memo:
        image = batch['image']
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            image = image.cuda()
        module = cast(clip.model.CLIP, self._strategy.module)
        image = einops.rearrange(image, 'c h w -> 1 c h w')
        embedding = module.encode_image(image)
        embedding = F.normalize(embedding)
        embedding = einops.rearrange(embedding, '1 c -> c')
        memo['output'] = embedding.half()
        return memo
