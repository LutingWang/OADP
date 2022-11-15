__all__ = [
    'debug',
]

import mmcv.cnn
import todd
import torch.nn as nn


class Debug(todd.BaseDebug):

    def init_cpu(self, **kwargs) -> None:
        super().init_cpu(**kwargs)
        mmcv.cnn.NORM_LAYERS.register_module(
            name='SyncBN',
            force=True,
            module=nn.BatchNorm2d,
        )


debug = Debug()
