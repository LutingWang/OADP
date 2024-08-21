import todd
import torch
from mmengine import MODELS

if todd.Store.cpu:
    MODELS.register_module(
        name='SyncBN',
        force=True,
        module=torch.nn.BatchNorm2d,
    )
