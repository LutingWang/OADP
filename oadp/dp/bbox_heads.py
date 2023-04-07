__all__ = [
    'BlockMixin',
    'ObjectMixin',
]

import todd
import torch
from mmdet.models import (
    HEADS,
    BBoxHead,
    Shared2FCBBoxHead,
    Shared4Conv1FCBBoxHead,
)
from todd.losses import LossRegistry as LR

from .classifiers import Classifier
from .utils import MultilabelTopKRecall


class NotWithRegMixin(BBoxHead):
    """Override the `with_reg` argument to ``False``."""

    def __init__(self, *args, with_reg: bool = False, **kwargs) -> None:
        super().__init__(*args, with_reg=False, **kwargs)


class BlockMixin(NotWithRegMixin):

    def __init__(self, *args, topk: int, loss: todd.Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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


class ObjectMixin(NotWithRegMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # `_bg_embedding` does not get trained, and will not be used during
        # inference.
        classifier: Classifier = self.fc_cls
        bg_embedding = classifier._bg_embedding
        assert bg_embedding is not None
        bg_embedding.requires_grad_(False)

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, None]:
        logits, _ = super().forward(*args, **kwargs)
        logits[:, -1] = float('-inf')  # disable `_bg_embedding`
        return logits, None


@HEADS.register_module()
class Shared2FCBlockBBoxHead(BlockMixin, Shared2FCBBoxHead):
    pass


@HEADS.register_module()
class Shared4Conv1FCObjectBBoxHead(ObjectMixin, Shared4Conv1FCBBoxHead):
    pass
