import argparse
import importlib
import pathlib

from mmengine.config import DictAction
from todd.configs import PyConfig

from ..categories import Categories
from ..utils import Globals
from .runners import DPRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--config-options', action=DictAction, default=dict())
    parser.add_argument('--override', action=DictAction, default=dict())
    parser.add_argument('--visual')
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--load-model-from', required=True, nargs='+')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = PyConfig.load(args.config, **args.config_options)
    config.override(args.override)

    for custom_import in config.get('custom_imports', []):
        importlib.import_module(custom_import)

    Globals.categories = Categories.get(config.categories)

    # runner = DPRunnerRegistry.build(
    #     config,
    #     name=f'{args.name}_test',
    #     visual=args.visual,
    #     autocast=args.autocast,
    # )
    # log(trainer, args, config)
    # trainer.strategy.load_model_from(args.load_model_from, strict=False)
    runner = DPRunner.from_cfg(
        config,
        name=f'{args.name}_test',
        visual=args.visual,
        autocast=args.autocast,
        load_model_from=args.load_model_from[0],
    )
    # runner.run()
    runner.test()


if __name__ == '__main__':
    main()
