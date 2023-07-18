__all__ = [
    'Formatter',
    'SGRFormatter',
    'logger',
    'get_',
    'has_',
    'set_',
    'del_',
    'exec_',
    'map_',
    'set_temp',
    'Module',
    'ModuleList',
    'ModuleDict',
    'Sequential',
    'get_rank',
    'get_local_rank',
    'get_world_size',
]

import builtins
import contextlib
import enum
import importlib.util
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Generator

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from packaging import version
from PIL import Image

from .enums import SGR


class Formatter(logging.Formatter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            "[%(asctime)s %(process)d:%(thread)d]"
            "[%(filename)s:%(lineno)d %(name)s %(funcName)s]"
            " %(levelname)s: %(message)s",
            *args,
            **kwargs,
        )


class SGRFormatter(Formatter):

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if record.levelno == logging.DEBUG:
            return SGR.format(s, SGR.FAINT)
        if record.levelno == logging.WARNING:
            return SGR.format(s, SGR.BOLD, SGR.FG_YELLOW)
        if record.levelno == logging.ERROR:
            return SGR.format(s, SGR.BOLD, SGR.FG_RED)
        if record.levelno == logging.CRITICAL:
            return SGR.format(s, SGR.BOLD, SGR.BLINK_SLOW, SGR.FG_RED)
        return s


# override logging settings from other packages
try:
    import lvis  # noqa: F401
except ImportError:
    pass
logging.basicConfig(force=True)

logger = logging.getLogger('todd')
logger.propagate = False
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(SGRFormatter())
logger.addHandler(handler)

try:
    import ipdb
    logger.info("`ipdb` is installed. Using it for debugging.")
    builtins.breakpoint = ipdb.set_trace
except ImportError:
    pass

if torch.__version__ < '1.7.0':
    logger.warning("Monkey patching `torch.maximum` and `torch.minimum`.", )
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min

if version.parse(torchvision.__version__) < version.parse('0.9.0'):
    logger.warning(
        "Monkey patching `torchvision.transforms.InterpolationMode`.",
    )

    class InterpolationMode(enum.Enum):
        BICUBIC = Image.BICUBIC

    transforms.InterpolationMode = InterpolationMode


def get_(obj, attr: str, default=...):
    try:
        return eval('__o' + attr, dict(__o=obj))
    except Exception:
        if default is not ...:
            return default
        raise


def has_(obj, name: str) -> bool:
    default = object()
    return get_(obj, name, default) is not default


def set_(obj, attr: str, value) -> None:
    locals_: dict[str, Any] = dict()
    exec(f'__o{attr} = __v', dict(__o=obj, __v=value), locals_)
    if len(locals_) != 0:
        raise ValueError(f"{attr} is invalid. Consider prepending a dot.")


def del_(obj, attr: str) -> None:
    exec(f'del __o{attr}', dict(__o=obj))


def exec_(source: str, **kwargs):
    locals_: dict[str, Any] = dict()
    exec(source, kwargs, locals_)
    return locals_


def map_(data, f: Callable[[Any], Any]):
    if isinstance(data, (list, tuple, set)):
        return data.__class__(map_(v, f) for v in data)
    if isinstance(data, dict):
        return data.__class__({k: map_(v, f) for k, v in data.items()})
    return f(data)


@contextlib.contextmanager
def set_temp(obj, name: str, value) -> Generator[None, None, None]:
    if has_(obj, name):
        prev = get_(obj, name)
        set_(obj, name, value)
        yield
        set_(obj, name, prev)
    else:
        set_(obj, name, value)
        yield
        del_(obj, name)


if importlib.util.find_spec('mmcv.runner') and not TYPE_CHECKING:
    from mmcv.runner import BaseModule as Module
    from mmcv.runner import ModuleDict, ModuleList, Sequential
else:
    from torch.nn import Module, ModuleDict, ModuleList, Sequential


def get_rank(*args, **kwargs) -> int:
    if dist.is_initialized():
        return dist.get_rank(*args, **kwargs)
    return 0


def get_local_rank(*args, **kwargs) -> int:
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return get_rank(*args, **kwargs)


def get_world_size(*args, **kwargs) -> int:
    if dist.is_initialized():
        return dist.get_world_size(*args, **kwargs)
    return 1
