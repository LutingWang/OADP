from mmdet.models import Shared2FCBBoxHead
from mmdet.models.utils.builder import LINEAR_LAYERS
import todd
import torch
import torch.nn as nn


@LINEAR_LAYERS.register_module()
class Classifier(todd.base.Module):

    def __init__(self, *args, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        weight = torch.load()
        assert weight.shape == (out_features, in_features)
        self.register_buffer('_weight', weight)
        self._scaler = nn.Parameter(torch.tensor(20.0))
        self._bias = nn.Parameter(torch.tensor(4.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self._weight.T) * self._scaler - self._bias


class CustomBBoxHead(Shared2FCBBoxHead):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc_cls = Classifier()
