__all__ = [
    'AsymmetricLoss',
    'RKDLoss',
]

import todd
import torch
from todd.models import LossRegistry
from todd.models.losses import BaseLoss, MSELoss

from ..utils import force_fp32


@LossRegistry.register_()
class AsymmetricLoss(BaseLoss):

    def __init__(
        self,
        *args,
        gamma_neg: float = 4,
        gamma_pos: float = 1,
        clip: float = 0.05,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._gamma_neg = gamma_neg
        self._gamma_pos = gamma_pos
        self._clip = clip
        self._eps = eps
        self.fp16_enabled = False

    @force_fp32(apply_to=('x', ))
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""Forward asymmetric loss.

        Args:
            x: :math:`n \times k`, probability distribution
            y: :math:`n \times k`, binary ground truth of type bool

        Returns:
            One element tensor representing loss.
        """
        comp_x = 1 - x

        # Asymmetric Clipping
        if self._clip > 0:
            comp_x = (comp_x + self._clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = y * x.clamp(min=self._eps).log()
        loss_neg = ~y * comp_x.clamp(min=self._eps).log()
        loss = loss_pos + loss_neg

        # Asymmetric Focusing
        if self._gamma_neg > 0 or self._gamma_pos > 0:
            with torch.no_grad():
                pt0 = x * y
                pt1 = comp_x * ~y  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self._gamma_pos * y + self._gamma_neg * ~y
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return self._reduce(-loss, *args, **kwargs)


@LossRegistry.register_()
class RKDLoss(MSELoss):

    def get_relations(self, feats: torch.Tensor) -> torch.Tensor:
        r"""Get relations between each pair of feats.

        Args:
            feats: :math:`\star \times c`

        Returns:
            :math:`\prod \star \times \prod \star`
        """
        feats = feats.view(-1, feats.shape[-1])
        relations = torch.einsum('m c, n c -> m n', feats, feats)
        return relations

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""Compute RKD loss.

        Args:
            preds: :math:`\star \times c`
            targets: :math:`\star \times d`

        Returns:
            loss: 1
        """
        assert pred.shape[:-1] == target.shape[:-1]
        pred_relations = self.get_relations(pred)
        target_relations = self.get_relations(target)
        return super().forward(
            pred_relations,
            target_relations,
            *args,
            **kwargs,
        )
