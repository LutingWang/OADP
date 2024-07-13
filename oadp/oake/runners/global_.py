from typing import Any, cast

import todd
import torch.nn.functional as F
from todd.runners import Memo

from ..datasets import GlobalBatch
from ..registries import OADPRunnerRegistry
from .base import BaseValidator


@OADPRunnerRegistry.register_()
class GlobalValidator(BaseValidator[GlobalBatch]):

    def _build(self, *args, **kwargs) -> None:
        super()._build(*args, clip_=todd.Config(adaptive=True), **kwargs)

    def _run_iter(self, batch: Any, memo: Memo, *args, **kwargs) -> Memo:
        image = cast(GlobalBatch, batch).image
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            image = image.cuda()
        embedding = self._model.encode_image(image)
        embedding = F.normalize(embedding)
        memo['output'] = embedding.squeeze(0).half()
        return memo
