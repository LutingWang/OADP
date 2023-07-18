import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Module, Sequential
from .base import AdaptRegistry, BaseAdapt
from typing import List

class Bottleneck(Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        downsample: bool = False,
    ):
        super().__init__()
        stride = 2 if downsample else 1
        self._downsample = None if in_channels == out_channels else nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self._conv1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self._conv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self._conv3 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
        )

    def init_weights(self):
        assert not self.is_init
        for layer in self.children():
            assert isinstance(layer, nn.Conv2d)
            nn.init.kaiming_uniform_(layer.weight, a=1)
        self._is_init = True

    def forward(self, x):
        identity = x

        out = self._conv1(x)
        out = F.relu_(out)
        out = self._conv2(out)
        out = F.relu_(out)
        out = self._conv3(out)

        if self._downsample is not None:
            identity = self._downsample(x)

        out += identity
        out = F.relu_(out)
        return out


class ResBlock(Sequential):

    def __init__(
        self,
        *,
        base_channels: int,
        expansion: int = 2,
        **kwargs,
    ):
        layer1 = Bottleneck(
            base_channels * expansion,
            base_channels,
            base_channels * expansion**2,
            downsample=True,
        )
        layer2 = Bottleneck(
            base_channels * expansion**2,
            base_channels,
            base_channels * expansion**2,
        )
        super().__init__(layer1, layer2, **kwargs)


@AdaptRegistry.register()
class LabelEncAdapt(BaseAdapt):

    def __init__(
        self,
        *args,
        num_classes: int = 80,
        base_channels: int = 32,
        expansion: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._stage1 = nn.Conv2d(
            num_classes,
            base_channels * 4,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        base_channels *= 2
        self._stage2 = Bottleneck(
            base_channels * 2,
            base_channels,
            base_channels * 4,
            downsample=True,
        )
        base_channels *= 2
        self._stage3 = ResBlock(
            base_channels=base_channels,
            expansion=expansion,
        )
        base_channels *= 2
        self._stage4 = ResBlock(
            base_channels=base_channels,
            expansion=expansion,
        )
        base_channels *= 2
        self._stage5 = Bottleneck(
            base_channels * 2,
            base_channels,
            base_channels * 4,
            downsample=True,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self._stage1(x)
        x2 = self._stage2(x1)
        x3 = self._stage3(x2)
        x4 = self._stage4(x3)
        x5 = self._stage5(x4)
        outs = [x2, x3, x4, x5]
        return outs
