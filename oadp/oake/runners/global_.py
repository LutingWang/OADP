from typing import Any, cast

import todd
import torch.nn.functional as F
from todd.runners import Memo

from ..datasets.global_ import T
from ..registries import OAKERunnerRegistry
from .base import BaseValidator
from typing import TypeVar
from torch import nn

ModuleType = TypeVar('ModuleType', bound=nn.Module)


@OAKERunnerRegistry.register_()
class GlobalValidator(BaseValidator[ModuleType]):

    def _build(self, *args, **kwargs) -> None:
        super()._build(*args, clip_=todd.Config(adaptive=True), **kwargs)

    def _run_iter(self, batch: T, memo: Memo, *args, **kwargs) -> Memo:
        image = batch.image
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            image = image.cuda()
        embedding = self._model.encode_image(image)
        embedding = F.normalize(embedding)
        memo['output'] = embedding.squeeze(0).half()
        return memo
