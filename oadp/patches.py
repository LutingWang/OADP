import mmcv
import todd
import torch
from mmengine import MODELS

with todd.utils.set_temp(mmcv, '__version__', '2.1.0'):
    import mmdet  # noqa: F401 pylint: disable=unused-import

if todd.Store.cpu:
    MODELS.register_module(
        name='SyncBN',
        force=True,
        module=torch.nn.BatchNorm2d,
    )
