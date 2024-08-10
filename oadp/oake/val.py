import argparse
import os
import pathlib
from typing import cast

import todd
from todd.patches.py import DictAction

from .registries import OAKERunnerRegistry
from .runners import BaseValidator
from .utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--override', action=DictAction)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = todd.configs.PyConfig.load(args.config)
    if args.override is not None:
        config.override(args.override)
    if not todd.Store.TRAIN_WITH_VAL_DATASET:
        validate(args, config, config.trainer)
    validate(args, config, config.validator)


def validate(
    args: argparse.Namespace,
    config: todd.Config,
    validator_config: todd.Config,
) -> None:
    runner: BaseValidator = OAKERunnerRegistry.build(
        validator_config,
        name=args.name,
    )
    log(runner, args, config)
    runner.run()


if __name__ == '__main__':
    main()
