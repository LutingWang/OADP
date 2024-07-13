import todd

from .base import *
from .dp import *
from .oake import *

if todd.Store.cpu:  # pylint: disable=using-constant-test
    import torch
    from mmengine import MODELS
    MODELS.register_module(
        name='SyncBN',
        force=True,
        module=torch.nn.BatchNorm2d,
    )
