__all__ = [
    'log',
    'oake_collate_fn',
]

import argparse
import pathlib
import sys
from typing import Any, cast

import todd
from todd.patches.torch import get_rank

from ..runners import BaseValidator


def log(
    runner: BaseValidator,
    args: argparse.Namespace,
    config: todd.Config,
) -> None:
    if get_rank() != 0:
        return

    runner.logger.info("Command\n" + ' '.join(sys.argv))
    runner.logger.info(f"Args\n{vars(args)}")
    runner.logger.info(f"Config\n{config.dumps()}")

    config_name = cast(pathlib.Path, args.config).name
    config.dump(runner.work_dir / config_name)


@todd.registries.CollateRegistry.register_()
def oake_collate_fn(batch: tuple[Any]) -> Any:
    assert len(batch) == 1
    return batch[0]
