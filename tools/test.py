import argparse
import pprint
import sys

import todd
import torch
import torch.distributed
import torch.utils.data
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import CustomDataset, build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import build_ddp, build_dp

sys.path.insert(0, '')
import oadp  # noqa: E402


class Validator(todd.utils.Validator):
    _model: torch.nn.Module

    def __init__(self, *args, fp16: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if fp16 and todd.Store.CUDA:
            wrap_fp16_model(self._model)
        if todd.Store.CPU:
            self._model = build_dp(self._model, 'cpu', device_ids=[0])
            self._test = single_gpu_test
        elif todd.Store.CUDA:
            self._model = build_ddp(
                self._model,
                'cuda',
                device_ids=[todd.get_local_rank()],
                broadcast_buffers=False,
            )
            self._test = multi_gpu_test
        else:
            raise NotImplementedError

        oadp.base.Globals.logger = self._logger

    def _build_dataloader(
        self,
        config: todd.Config,
    ) -> torch.utils.data.DataLoader:
        if todd.Store.DRY_RUN:
            config.workers_per_gpu = 0
        config.dataset = build_dataset(config.dataset, dict(test_mode=True))
        dataloader = build_dataloader(
            dist=todd.Store.CUDA,
            shuffle=False,
            **config,
        )
        return dataloader

    def _run_iter(self, i: int, batch, memo: todd.utils.Memo) -> torch.Tensor:
        pass

    def _run(self, memo: todd.utils.Memo) -> None:
        memo['outputs'] = self._test(self._model, self._dataloader)

    def _after_run(self, memo: todd.utils.Memo) -> None:
        super()._after_run(memo)
        if todd.get_rank() == 0:
            dataset: CustomDataset = self._dataloader.dataset
            metric = dataset.evaluate(memo['outputs'])
            self._logger.info('\n' + pprint.pformat(metric))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model',
    )
    parser.add_argument('name')
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--override', action=todd.DictAction)
    parser.add_argument('--odps', action=todd.DictAction)
    args = parser.parse_args()
    return args


def build_model(config: todd.Config) -> torch.nn.Module:
    config = config.copy()
    config.pop('train_cfg', None)
    config.pop('pretrained', None)
    backbone: todd.Config = config.backbone
    backbone.pop('init_cfg', None)
    return build_detector(config)


def main() -> None:
    if todd.Store.CUDA:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(todd.get_local_rank())

    args = parse_args()

    if args.odps is not None:
        oadp.odps_init(args.odps)

    config: todd.Config = args.config

    if args.override is not None:
        config.override(args.override)

    oadp.base.Globals.categories = getattr(oadp.base, config.categories)

    model = build_model(config.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    validator = Validator(
        name=args.name,
        model=model,
        log=todd.Config(),
        load_state_dict=todd.Config(),
        state_dict=todd.Config(),
        **config.validator,
    )
    validator.run()


if __name__ == '__main__':
    main()
