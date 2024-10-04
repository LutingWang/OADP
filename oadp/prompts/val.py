import argparse
import pathlib
from typing import Any

import todd
import torch
import torch.distributed as dist
from todd.patches.py_ import DictAction
from todd.patches.torch import get_local_rank, get_rank, get_world_size
from tqdm import tqdm

from .prompters import BasePrompter
from .registries import PrompterRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prompt')
    parser.add_argument('name')
    parser.add_argument('--config', action=DictAction, required=True)
    parser.add_argument('--model', action=DictAction, required=True)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    dist.init_process_group()
    torch.cuda.set_device(get_local_rank() % torch.cuda.device_count())

    prompter: BasePrompter = PrompterRegistry.build(
        args.config,
        model=args.model,
    )

    categories = prompter.load()
    if todd.Store.DRY_RUN:
        categories = categories[:get_world_size() * 2 + 1]

    num_categories = len(categories)

    categories = categories[get_rank()::get_world_size()]
    categories = list(map(prompter, tqdm(categories)))

    gathered_categories: list[list[dict[str, Any]]] = \
        [[] for _ in range(get_world_size())]
    dist.all_gather_object(gathered_categories, categories)

    if get_rank() == 0:
        categories = [
            gathered_categories[i % get_world_size()][i // get_world_size()]
            for i in range(num_categories)
        ]
        if todd.Store.DRY_RUN:
            print(categories)

        work_dir = pathlib.Path('work_dirs/prompts')
        if todd.Store.DRY_RUN:
            work_dir = work_dir / 'dry_run'
        work_dir.mkdir(parents=True, exist_ok=True)

        torch.save(categories, work_dir / f'{args.name}.pth')


if __name__ == '__main__':
    main()
