import argparse
import pprint

import todd
import torch
import torch.distributed
import torch.utils.data
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import build_ddp, build_dp

from ..base import Globals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--override', action=todd.DictAction)
    parser.add_argument('--metrics', nargs='+', default=['bbox'])
    args = parser.parse_args()
    return args


def build_model(config: todd.Config, checkpoint: str) -> torch.nn.Module:
    config.pop('train_cfg', None)
    config.pop('pretrained', None)
    backbone: todd.Config = config.backbone
    backbone.pop('init_cfg', None)
    model = build_detector(config)
    load_checkpoint(model, checkpoint, map_location='cpu')
    return model


def main() -> None:
    args = parse_args()
    config: todd.Config = args.config
    if args.override is not None:
        config.override(args.override)

    config_validator_dataloader: todd.Config = config.validator.dataloader
    if todd.Store.DRY_RUN:
        config_validator_dataloader.workers_per_gpu = 0

    from ..base import coco, lvis  # noqa: F401
    Globals.categories = eval(config.categories)

    if todd.Store.CUDA:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(todd.get_local_rank())

    dataset = build_dataset(
        config_validator_dataloader.dataset,
        dict(test_mode=True),
    )
    config_validator_dataloader.dataset = dataset
    dataloader = build_dataloader(
        dist=todd.Store.CUDA,
        shuffle=False,
        **config_validator_dataloader,
    )

    if todd.Store.CPU:
        model = build_model(config.model, args.checkpoint)
        model = build_dp(model, 'cpu', device_ids=[0])
        outputs = single_gpu_test(model, dataloader)
    elif todd.Store.CUDA:
        model = build_model(config.model, args.checkpoint)
        if config.validator.fp16:
            wrap_fp16_model(model)
        model = build_ddp(
            model,
            'cuda',
            device_ids=[todd.get_local_rank()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, dataloader)
    else:
        raise NotImplementedError

    if todd.get_rank() == 0:
        metric = dataset.evaluate(outputs, args.metrics)
        todd.logger.info('\n' + pprint.pformat(metric))


if __name__ == '__main__':
    main()
