__all__ = [
    'odps_init',
    'Store',
]

import os
from typing import Mapping

import todd

ODPS_PATHS = dict(
    data='/data/oss_bucket_0',
    pretrained='/data/oss_bucket_0/ckpts',
    work_dirs='/data/oss_bucket_0/work_dirs',
)


class Store(metaclass=todd.StoreMeta):
    ODPS: bool
    NUM_CLASSES: int
    NUM_BASE_CLASSES: int
    TRAINING: bool


def odps_init(kwargs: Mapping[str, str]) -> None:
    Store.ODPS = True

    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.update({k: v for k, v in kwargs.items()})

    for k, v in ODPS_PATHS.items():
        if not os.path.lexists(k):
            os.symlink(v, k)

    todd.get_logger().debug(f"ODPS initialized with {os.listdir('.')}.")
