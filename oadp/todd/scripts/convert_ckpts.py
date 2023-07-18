import argparse
import re
from enum import Enum, auto
from typing import Iterable, Mapping, MutableMapping

import torch
from mmcv import DictAction


class MatchMode(Enum):
    PREFIX = auto()
    REGEX = auto()
    EXACT = auto()

    def match(self, key: str, pattern: str) -> bool:
        if self == MatchMode.PREFIX:
            return key.startswith(pattern)
        if self == MatchMode.REGEX:
            return re.match(pattern, key) is not None
        if self == MatchMode.EXACT:
            return key == pattern
        raise ValueError(f"Unknown match mode: {self}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Checkpoints")
    parser.add_argument('ckpt')
    for mode in MatchMode:
        parser.add_argument(
            f'--rename-{mode.name.lower()}',
            nargs='+',
            action=DictAction,
        )
        parser.add_argument(f'--remove-{mode.name.lower()}', nargs='+')
    parser.add_argument('--out')
    args = parser.parse_args()
    return args


def rename_exact(
    state_dict: MutableMapping[str, torch.Tensor],
    mapping: Mapping[str, str],
):
    for k, v in mapping.items():
        state_dict[v] = state_dict[k]
        del state_dict[k]


def rename(
    state_dict: MutableMapping[str, torch.Tensor],
    mapping: Mapping[str, str],
    mode: MatchMode,
):
    if mode == MatchMode.EXACT:
        rename_exact(state_dict, mapping)
        return
    exact_mapping = {}
    for k, v in state_dict.items():
        patterns = {pattern for pattern in mapping if mode.match(k, pattern)}
        if len(patterns) == 0:
            continue
        longest_pattern = max(patterns, key=len)
        exact_mapping[k] = k.replace(longest_pattern, mapping[longest_pattern])
    rename_exact(state_dict, exact_mapping)


def remove_exact(
    state_dict: MutableMapping[str, torch.Tensor],
    patterns: Iterable[str],
):
    for k in patterns:
        del state_dict[k]


def remove(
    state_dict: MutableMapping[str, torch.Tensor],
    patterns: Iterable[str],
    mode: MatchMode,
):
    if mode == MatchMode.EXACT:
        remove_exact(state_dict, patterns)
        return
    exact_patterns = set()
    for k in state_dict:
        if any(mode.match(k, pattern) for pattern in patterns):
            exact_patterns.add(k)
    remove_exact(state_dict, exact_patterns)


def main():
    args = parse_args()

    ckpt = torch.load(args.ckpt, 'cpu')
    state_dict = ckpt
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    if args.out is None:
        args.out = args.ckpt + '.converted'

    for mode in MatchMode:
        mapping = getattr(args, f'rename_{mode.name.lower()}')
        if mapping is not None:
            rename(state_dict, mapping, mode)

    for mode in MatchMode:
        patterns = getattr(args, f'remove_{mode.name.lower()}')
        if patterns is not None:
            remove(state_dict, patterns, mode)

    torch.save(state_dict, args.out)


if __name__ == '__main__':
    main()
