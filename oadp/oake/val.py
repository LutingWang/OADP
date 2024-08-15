import argparse
import importlib
import pathlib
from typing import TypeVar

import todd
from todd.configs import PyConfig
from todd.patches.py import DictAction
from torch import nn

from .registries import OAKERunnerRegistry
from .runners import BaseValidator
from .utils import log

T = TypeVar('T', bound=nn.Module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--override', action=DictAction)
    parser.add_argument('--auto-fix', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = PyConfig.load(
        args.config,
        **args.config_options,
        auto_fix=args.auto_fix,
    )
    if args.override is not None:
        config.override(args.override)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    if not todd.Store.TRAIN_WITH_VAL_DATASET:
        validate(args, config, True)
    validate(args, config, False)


def validate(
    args: argparse.Namespace,
    config: todd.Config,
    train: bool,
) -> None:
    runner: BaseValidator[T] = OAKERunnerRegistry.build(
        config.trainer if train else config.validator,
        name=f'{args.name}/{"train" if train else "val"}',
    )
    log(runner, args, config)
    runner.run()


if __name__ == '__main__':
    main()
