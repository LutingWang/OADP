import einops
import torch
import torch.nn as nn

from .base import AdaptRegistry, BaseAdapt


@AdaptRegistry.register()
class Decouple(BaseAdapt):

    def __init__(
        self,
        num: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num = num
        self._layer = nn.Linear(in_features, out_features * num, bias)

    def forward(self, feat: torch.Tensor, id_: torch.Tensor) -> torch.Tensor:
        """Decouple features.

        Args:
            feat: n x dim
            pos: n

        Returns:
            decoupled_feat n x dim
        """
        feat = self._layer(feat)  # n x (num x dim)
        feat = einops.rearrange(
            feat,
            'n (num dim) -> n num dim',
            num=self._num,
        )
        feat = feat[torch.arange(id_.shape[0]), id_.long()]  # n x dim
        return feat
