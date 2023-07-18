__all__ = [
    'BaseAdapt',
    'AdaptRegistry',
]

import inspect
import itertools

import einops.layers.torch
import torch.nn as nn

from ..base import Module, Registry


class BaseAdapt(Module):
    pass


class AdaptRegistry(Registry):
    pass


for _, class_ in itertools.chain(
    inspect.getmembers(nn, inspect.isclass),
    inspect.getmembers(einops.layers.torch, inspect.isclass),
):
    if issubclass(class_, nn.Module):
        AdaptRegistry.register()(class_)

try:
    import mmcv.cnn

    for k, v in itertools.chain(
        mmcv.cnn.CONV_LAYERS.module_dict.items(),
        mmcv.cnn.PLUGIN_LAYERS.module_dict.items(),
    ):
        AdaptRegistry.register(keys=(f'mmcv_{k}', ))(v)
except Exception:
    pass
