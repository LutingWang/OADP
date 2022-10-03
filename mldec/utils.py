__all__ = [
    'odps_init',
    'k8s_init',
    'all_gather',
]

import os
from typing import Sequence, List, Tuple

import torch
import torch.distributed

import todd


def odps_init(kwargs: todd.base.Config) -> None:
    logger = todd.base.get_logger()
    logger.debug("ODPS initializing.")
    kwargs.setdefault('LOCAL_RANK', '0')
    os.environ.update(kwargs)
    if not os.path.lexists('data'):
        os.symlink('/data/oss_bucket_0', 'data')
    if not os.path.lexists('pretrained'):
        os.symlink('/data/oss_bucket_0/ckpts', 'pretrained')
    if not os.path.lexists('work_dirs'):
        os.symlink('/data/oss_bucket_0/work_dirs', 'work_dirs')
    logger.debug(f"ODPS initialization done with {os.listdir('.')}.")


def k8s_init(kwargs: todd.base.Config) -> None:
    logger = todd.base.get_logger()
    logger.debug("ODPS initializing.")
    # kwargs.setdefault('LOCAL_RANK', '0')
    os.environ.update(kwargs)
    root = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/dingzihan/openset/data'
    if not os.path.lexists('data'):
        os.symlink(f'{root}', 'data')
    if not os.path.lexists('pretrained'):
        os.symlink(f'{root}/ckpts', 'pretrained')
    if not os.path.lexists('work_dirs'):
        os.symlink(f'{root}/work_dirs', 'work_dirs')
    logger.debug(f"ODPS initialization done with {os.listdir('.')}.")


def all_gather(tensors: Tuple[torch.Tensor]) -> List[torch.Tensor]:
    tensor = torch.cat(tensors)
    tensors = [torch.zeros_like(tensor) for _ in range(todd.base.get_world_size())]
    torch.distributed.all_gather(tensors, tensor)
    return tensors
