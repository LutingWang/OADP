__all__ = [
    'OAKECallback',
]

from typing import Any, TypedDict

import torch
from todd.runners import Memo
from todd.runners.callbacks import BaseCallback
from torch import nn

from .base import BaseValidator
from .registries import OAKECallbackRegistry
from ..datasets import Batch


@OAKECallbackRegistry.register_()
class OAKECallback(BaseCallback[nn.Module]):
    runner: BaseValidator

    def should_continue(self, batch: Batch | None, memo: Memo) -> bool:
        return super().should_continue(batch, memo) or batch is None

    def after_run_iter(self, batch: Batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        torch.save(memo['output'], self.runner.output_path(batch['id_']))
