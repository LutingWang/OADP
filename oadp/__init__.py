import os

os.environ.setdefault('MPS', 'False')  # isort: skip

import todd

from .base import *
from .dp import *
from .oake import *

if todd.Store.CPU:
    import torch
    from mmcv.cnn import NORM_LAYERS
    NORM_LAYERS.register_module(
        name='SyncBN',
        force=True,
        module=torch.nn.BatchNorm2d,
    )
