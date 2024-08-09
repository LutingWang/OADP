from typing import Any, TypeVar, cast

import todd
import torch
import torch.cuda
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from todd.runners import Memo

from ..datasets.block import T
from ..registries import OAKERunnerRegistry
from .base import BaseValidator
from torch import nn

ModuleType = TypeVar('ModuleType', bound=nn.Module)


@OAKERunnerRegistry.register_()
class BlockValidator(BaseValidator[ModuleType]):

    def _build(self, *args, **kwargs) -> None:
        super()._build(*args, clip_=todd.Config(adaptive=False), **kwargs)

    def _run_iter(self, batch: Any, memo: Memo) -> torch.Tensor:
        blocks = cast(T, batch).blocks
        bboxes = cast(T, batch).bboxes
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
