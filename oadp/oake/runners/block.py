from typing import Any, cast

import todd
import torch
import torch.cuda
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from todd.runners import Memo

from ..datasets import BlockBatch
from ..registries import OADPRunnerRegistry
from .base import BaseValidator


@OADPRunnerRegistry.register_()
class BlockValidator(BaseValidator[BlockBatch]):

    def _build(self, *args, **kwargs) -> None:
        super()._build(*args, clip_=todd.Config(adaptive=False), **kwargs)

    def _run_iter(self, batch: Any, memo: Memo) -> torch.Tensor:
        blocks = cast(BlockBatch, batch).blocks
        bboxes = cast(BlockBatch, batch).bboxes
        if todd.Store.cuda:
            blocks = blocks.cuda()
            bboxes = bboxes.cuda()
        embeddings = self._model.encode_image(blocks)
        embeddings = F.normalize(embeddings)
        memo['output'] = dict(
            embeddings=embeddings.half(),
            bboxes=bboxes.half(),
        )
        return memo
