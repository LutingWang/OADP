__all__ = [
    'AsymmetricLoss',
]

import todd
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32


@todd.losses.LossRegistry.register()
class AsymmetricLoss(todd.losses.BaseLoss):

    def __init__(
        self,
        *args,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.fp16_enabled = False

    @force_fp32(apply_to=('x', ))
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """"
        Args:
            x: n x k, probability distribution
            y: n x k, binary ground truth of type bool

        Returns:
            loss: 1
        """
        comp_x = 1 - x

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            comp_x = (comp_x + self.clip).clamp(max=1)

        # Basic CE calculation
        loss_pos = y * torch.log(x.clamp(min=self.eps))
        loss_neg = ~y * torch.log(comp_x.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = x * y
            pt1 = comp_x * ~y  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * ~y
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return self.reduce(-loss, **kwargs)


@todd.losses.LossRegistry.register()
class RKDLoss(todd.losses.MSELoss):

    def get_relations(self, feats: torch.Tensor) -> torch.Tensor:
        """Get relations between each pair of feats.

        Args:
            feats: * x c

        Returns:
            relations: prod(*) x prod(*)
        """
        feats = feats.reshape(-1, feats.shape[-1])
        feats = F.normalize(feats)
        relations = torch.einsum('m c, n c -> m n', feats, feats)
        return relations

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute RKD loss.

        Args:
            preds: * x c
            targets: * x d

        Returns:
            loss: 1
        """
        assert preds.shape[:-1] == targets.shape[:-1]
        pred_relations = self.get_relations(preds)
        target_relations = self.get_relations(targets)
        return super().forward(
            pred_relations,
            target_relations,
            *args,
            **kwargs,
        )
