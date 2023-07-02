import todd
import torch
import argparse
import pathlib

from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

from .inference import single_gpu_infer
from .plot_utils import plot_single_img

def build_model(config: todd.Config, checkpoint: str) -> torch.nn.Module:
    config.pop('train_cfg', None)
    config.pop('pretrained', None)
    backbone: todd.Config = config.backbone
    backbone.pop('init_cfg', None)
    model = build_detector(config)
    load_checkpoint(model, checkpoint, map_location='cpu')
    return model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=pathlib.Path)
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img', help='image')
    
    parser.add_argument('--override', action=todd.DictAction)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config: todd.Config = args.config
    if args.override is not None:
        config.override(args.override)
    
    name: pathlib.Path = args.name
    work_dir = 'work_dirs' / name
    work_dir.mkdir(parents=True, exist_ok=True)

    from oadp.base import Globals
    from oadp.base import coco, lvis  # noqa: F401
    Globals.categories = eval(config.categories)
    
    if todd.Store.CUDA:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(todd.get_local_rank())
    
    model = build_model(config.model, args.checkpoint)
    results = single_gpu_infer(model, args.img, config)
    plot_single_img(args.img, results, 0.15, str(work_dir / 'result.png'), Globals.categories.all_)
    
if __name__ == "__main__":
    main()