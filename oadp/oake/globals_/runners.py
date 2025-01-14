__all__ = [
    'GlobalValidator',
]

import einops
import todd
import torch
from todd.bases.registries import Item
from todd.runners import Memo

from ..registries import OAKEModelRegistry, OAKERunnerRegistry
from ..runners import BaseValidator
from .datasets import Batch


@OAKERunnerRegistry.register_()
class GlobalValidator(BaseValidator):

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
        image = batch['image']
        if todd.Store.cuda:
            image = image.cuda()
        image = einops.rearrange(image, 'c h w -> 1 c h w')
        embedding: torch.Tensor = self.model(image)
        embedding = einops.rearrange(embedding, '1 c -> c')
        memo['output'] = embedding.half()
        return memo
