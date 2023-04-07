__all__ = [
    'MultilabelTopKRecall',
    'NormalizedLinear',
]

import sklearn.metrics
import todd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelTopKRecall(todd.Module):

    def __init__(self, *args, k: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._k = k

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the multilabel topk recall.

        Args:
            logits: :math:`bs \\times K`, float.
            targets: :math:`bs \\times K`, bool.

        Returns:
            One element tensor representing the recall.
        """
        _, indices = logits.topk(self._k)
        preds = torch.zeros_like(targets).scatter(1, indices, 1)
        # labels showing up at least once
        labels, = torch.where(targets.sum(0))
        recall = sklearn.metrics.recall_score(
            targets.cpu().numpy(),
            preds.cpu().numpy(),
            labels=labels.cpu().numpy(),
            average='macro',
            zero_division=0,
        )
        return logits.new_tensor(recall * 100)


class NormalizedLinear(nn.Linear):

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = super().forward(*args, **kwargs)
        return F.normalize(x)
