from typing import Any

import todd
import torch
from todd.runners import Memo
from todd.runners.callbacks import BaseCallback as BaseCallback_

from .base import BaseValidator
from .registries import OAKECallbackRegistry
from torch import nn


@OAKECallbackRegistry.register_()
class BaseCallback(BaseCallback_[nn.Module]):
    runner: BaseValidator[nn.Module]

    def should_continue(self, batch: Any, memo: Memo) -> bool:
        return super().should_continue(batch, memo) or batch is None

    def after_run_iter(self, batch: Any, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        torch.save(memo['output'], self.runner.output_path(batch.key))
