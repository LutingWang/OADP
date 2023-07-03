import argparse
import pathlib
from datetime import datetime

import mmcv
import mmdet
import todd
import torch
import torch.distributed
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import BaseDetector, build_detector
from mmdet.utils import collect_env, get_root_logger

from ..base import Globals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=pathlib.Path)
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('--override', action=todd.DictAction)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    name: pathlib.Path = args.name
    config: todd.Config = args.config
    if args.override is not None:
        config.override(args.override)

    if todd.Store.DRY_RUN:
        name = 'dry_run' / name
    work_dir = 'work_dirs' / name
    work_dir.mkdir(parents=True, exist_ok=True)
    config.dump(work_dir / 'config.py')

    config_trainer: todd.Config = config.trainer
    config_validator: todd.Config = config.validator
    config_trainer_dataloader: todd.Config = config_trainer.dataloader
    config_validator_dataloader: todd.Config = config_validator.dataloader

    if todd.Store.TRAIN_WITH_VAL_DATASET:
        config_trainer_dataloader.dataset.update(
            ann_file=config_validator_dataloader.dataset.ann_file,
            img_prefix=config_validator_dataloader.dataset.img_prefix,
        )
    if todd.Store.DRY_RUN:
        config_trainer.log_config.interval = 1
        config_trainer.checkpoint_config.interval = 6
        config_trainer.evaluation.interval = 3
        dataloader = dict(samples_per_gpu=1, workers_per_gpu=0)
        config_trainer_dataloader.update(dataloader)
        config_validator_dataloader.update(dataloader)

    from ..base import coco, lvis  # noqa: F401
    Globals.categories = eval(config.categories)

    if todd.Store.CUDA:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(todd.get_local_rank())

    if todd.Store.CPU:
        config_trainer.fp16 = None
        config_trainer.gpu_ids = [None]
        config_trainer.device = 'cpu'
    elif todd.Store.CUDA:
        config_trainer.gpu_ids = range(todd.get_world_size())
        config_trainer.device = 'cuda'
    else:
        raise NotImplementedError

    timestamp = datetime.now().astimezone().isoformat()
    timestamp = timestamp.replace(':', '-').replace('+', '-').replace('.', '_')
    logger = get_root_logger(
        log_file=work_dir / f'{timestamp}.log',
        log_level=config_trainer.log_level,
    )

    env_info = '\n'.join([f'{k}: {v}' for k, v in collect_env().items()])
    logger.info("Environment Info:\n" + env_info)
    logger.info("Config:\n" + config.dumps())

    todd.reproduction.init_seed(config_trainer.seed)
    model: BaseDetector = build_detector(config.model)
    model.init_weights()
    dataset = build_dataset(config_trainer_dataloader.dataset)

    train_dataloader = config_trainer_dataloader.copy()
    train_dataset = train_dataloader.pop('dataset')
    val_dataloader = config_validator_dataloader.copy()
    val_dataset = val_dataloader.pop('dataset')
    trainer = dict(config_trainer)
    trainer.pop('dataloader')
    trainer['data'] = dict(
        train_dataloader=train_dataloader,
        train=train_dataset,
        val_dataloader=val_dataloader,
        val=val_dataset,
        test_dataloader=val_dataloader,
        test=val_dataset,
    )
    trainer['checkpoint_config']['meta'] = dict(
        mmdet_version=mmdet.__version__,
        classes=dataset.CLASSES,
    )
    trainer['work_dir'] = str(work_dir)
    train_detector(
        model,
        [dataset],
        mmcv.Config(trainer),
        distributed=todd.Store.CUDA,
        validate=True,
        timestamp=timestamp,
        meta=dict(env_info=env_info),
    )


if __name__ == '__main__':
    main()
