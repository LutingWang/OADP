__all__ = [
    'BlockBBoxHead',
]

import todd
import torch
from mmdet.models import HEADS, Shared2FCBBoxHead
from todd.losses import LossRegistry as LR

from .utils import MultilabelTopKRecall


@HEADS.register_module()
class BlockBBoxHead(Shared2FCBBoxHead):

    def __init__(
        self,
        *args,
        topk: int,
        loss: todd.Config,
        with_reg: bool = False,
        **kwargs,
    ) -> None:
        # block head does not need for regression
        super().__init__(*args, with_reg=False, **kwargs)

        self._multilabel_topk_recall = MultilabelTopKRecall(k=topk)
        self._loss = LR.build(loss)

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return dict(
            loss_block=self._loss(logits.sigmoid(), targets),
            recall_block=self._multilabel_topk_recall(logits, targets),
        )
