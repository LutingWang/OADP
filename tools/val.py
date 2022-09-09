import argparse
import sys
import pathlib

import sklearn  # must before torch
import todd
import torch.cuda
import torch.distributed

sys.path.insert(0, '')
import mldec.runner
import clip
from mldec.debug import debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('name')
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--config-options', action=todd.base.DictAction)
    args = parser.parse_args()
    return args


def main() -> float:
    args = parse_args()
    if args.odps is not None:
        mldec.runner.odps_init(args.odps)
    runner = mldec.runner.Runner(args.name, args.config, args.config_options)

    debug.init(config=runner.config)
    if not debug.CPU:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.base.get_local_rank())

    clip_model, clip_transform = clip.load('pretrained/clip/RN50.pt', 'cpu')
    runner.build_val_dataloader(clip_transform)
    runner.build_model(clip_model)
    runner.load_checkpoint(40)
    return runner.val()

if __name__ == '__main__':
    main()
