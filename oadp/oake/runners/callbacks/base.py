from typing import Any

import todd
import torch
from todd.runners import Memo

from ..base import BaseValidator
from ..registries import OADPCallbackRegistry


@OADPCallbackRegistry.register_()
class BaseCallback(todd.runners.callbacks.BaseCallback):
    runner: BaseValidator[Any]

    def should_continue(self, batch: Any, memo: Memo) -> bool:
        return super().should_continue(batch, memo) or batch is None

    def after_run_iter(self, batch: Any, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        id_: int = batch.id_
        torch.save(memo['output'], self.runner.output_path(id_))
