__all__ = [
    'odps_init',
]

import os

import todd

from .debug import debug

ODPS_PATHS = dict(
    data='/data/oss_bucket_0',
    pretrained='/data/oss_bucket_0/ckpts',
    work_dirs='/data/oss_bucket_0/work_dirs',
)


def odps_init(kwargs: todd.Config) -> None:
    kwargs.setdefault('LOCAL_RANK', '0')
    os.environ.update({k: str(v) for k, v in kwargs.items()})

    debug.ODPS = True

    for k, v in ODPS_PATHS.items():
        if not os.path.lexists(k):
            os.symlink(v, k)

    todd.get_logger().debug(f"ODPS initialized with {os.listdir('.')}.")
