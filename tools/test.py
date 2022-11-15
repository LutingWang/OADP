import argparse
import os
import sys

import todd
import torch
from mmcv import Config, DictAction
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (
    build_dataloader,
    build_dataset,
    replace_ImageToTensor,
)
from mmdet.models import build_detector
from mmdet.utils import (
    build_ddp,
    build_dp,
    compat_cfg,
    get_device,
    setup_multi_processes,
)

sys.path.insert(0, '')
import oadp  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model',
    )
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)',
    )
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.',
    )
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    if args.odps is not None:
        oadp.odps_init(args.odps)

    cfg = todd.base.Config.load(args.config)
    cfg = Config(cfg, filename=args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    oadp.debug.init()
    if oadp.debug.CPU:
        cfg.fp16 = None
    if oadp.debug.DRY_RUN:
        cfg.data.workers_per_gpu = 0
        cfg.log_config.interval = 1

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=distributed,
        shuffle=False,
    )

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline,
            )
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    rank, _ = get_dist_info()
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()
    if rank == 0:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        eval_kwargs.pop('interval', None)
        metric = dataset.evaluate(outputs, **eval_kwargs)
        print(metric)


if __name__ == '__main__':
    main()
