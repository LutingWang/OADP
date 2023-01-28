import argparse
from typing import Any, NamedTuple

import einops
import nni
import numpy as np
import todd
import torch
import torch.distributed
import torch.utils.data
import torch.utils.data.distributed
from mmdet.core import bbox2result, multiclass_nms
from mmdet.datasets import build_dataset

from ..base import Globals


class Batch(NamedTuple):
    image_id: str
    bboxes: torch.Tensor
    bbox_logits: torch.Tensor
    object_logits: torch.Tensor
    objectness: torch.Tensor


class Dataset(todd.datasets.PthDataset):

    def __init__(self, root: str) -> None:
        access_layer = todd.datasets.PthAccessLayer(root)
        super().__init__(access_layer=access_layer)

    def __getitem__(self, index: int) -> Batch:
        key = self._keys[index]
        item = self._access_layer[key]
        return Batch(key, **item)


class Model(todd.Module):

    def __init__(
        self,
        *args,
        nms: todd.Config,
        bboxes: todd.Config,
        objects: todd.Config,
        objectness: todd.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._nms = nms
        self._bboxes = bboxes
        self._objects = objects
        self._objectness = objectness

    def _classify(
        self,
        scores: torch.Tensor,
        config: todd.Config,
    ) -> torch.Tensor:
        base_categories = slice(Globals.categories.num_bases)
        novel_categories = slice(
            Globals.categories.num_bases,
            Globals.categories.num_all,
        )
        scores[:, base_categories] *= config.base_scaler
        scores[:, novel_categories] *= config.novel_scaler
        scores = scores.softmax(-1)
        scores[:, base_categories] = \
            scores[:, base_categories]**config.base_gamma
        scores[:, novel_categories] = \
            scores[:, novel_categories]**config.novel_gamma
        return scores

    def forward(
        self,
        bboxes: torch.Tensor,
        bbox_logits: torch.Tensor,
        object_logits: torch.Tensor,
        objectness: torch.Tensor,
    ) -> list[np.ndarray]:
        bbox_scores = self._classify(bbox_logits, self._bboxes)
        object_scores = self._classify(object_logits, self._objects)

        objectness = einops.rearrange(objectness, 'n -> n 1')
        objectness = objectness**self._objectness.gamma

        ensemble_score = (bbox_scores * object_scores * objectness).float()

        return bbox2result(
            *multiclass_nms(bboxes.float(), ensemble_score, **self._nms),
            Globals.categories.num_all,
        )


class Validator(todd.utils.Validator):

    def __init__(self, *args, evaluator: todd.Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._evaluator = evaluator

    def _build_dataloader(
        self,
        config: todd.Config,
    ) -> torch.utils.data.DataLoader:
        config.dataset = Dataset(**config.dataset)
        if todd.Store.CUDA:
            config.sampler = torch.utils.data.distributed.DistributedSampler(
                config.dataset,
                shuffle=False,
            )
        return torch.utils.data.DataLoader(batch_size=None, **config)

    def _before_run(self) -> todd.utils.Memo:
        memo = super()._before_run()
        memo['results'] = dict()
        return memo

    def _run_iter(self, batch: Batch, memo: todd.utils.Memo) -> torch.Tensor:
        bboxes = batch.bboxes
        bbox_logits = batch.bbox_logits
        object_logits = batch.object_logits
        objectness = batch.objectness
        if todd.Store.CPU:
            bboxes = bboxes.float()
            bbox_logits = bbox_logits.float()
            object_logits = object_logits.float()
            objectness = objectness.float()
        if todd.Store.CUDA:
            bboxes = bboxes.cuda()
            bbox_logits = bbox_logits.cuda()
            object_logits = object_logits.cuda()
            objectness = objectness.cuda()
        memo['results'][batch.image_id] = self._model(
            bboxes,
            bbox_logits,
            object_logits,
            objectness,
        )
        return torch.tensor(0.0)

    def _after_run(self, memo: todd.Config):
        super()._after_run(memo)

        if todd.Store.CUDA:
            results_list: list[dict[str, Any]] = \
                [dict()] * todd.get_world_size()
            torch.distributed.all_gather_object(results_list, memo['results'])
            if todd.get_rank() != 0:
                return
            results = {k: v for d in results_list for k, v in d.items()}
        else:
            results = memo['results']

        evaluator = build_dataset(
            self._evaluator,
            default_args=dict(test_mode=True),
        )
        result_list = [
            results[f'{img_id:012d}'] for img_id in evaluator.img_ids
        ]
        result = evaluator.evaluate(result_list)
        mAP = result['COCO_48_bbox_mAP_50']
        nni.report_final_result(mAP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('root')
    args = parser.parse_args()
    return args


def main() -> None:
    params = nni.get_next_parameter()
    params = []
    if len(params) == 0:
        params = dict(
            bbox_base_scaler=1,
            bbox_novel_scaler=1,
            bbox_base_gamma=2 / 3,
            bbox_novel_gamma=1 / 3,
            object_base_scaler=1,
            object_novel_scaler=1,
            object_base_gamma=1 / 3,
            object_novel_gamma=2 / 3,
            objectness_gamma=0,
        )
    print(params)

    args = parse_args()
    config: todd.Config = args.config

    from ..base import coco, lvis  # noqa: F401
    Globals.categories = eval(config.categories)

    if todd.Store.CUDA:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(todd.get_local_rank())

    model = Model(
        nms=todd.Config(
            score_thr=config.model.test_cfg.rcnn.score_thr,
            nms_cfg=config.model.test_cfg.rcnn.nms,
            max_num=config.model.test_cfg.rcnn.max_per_img,
        ),
        bboxes=todd.Config(
            base_scaler=params['bbox_base_scaler'],
            novel_scaler=params['bbox_novel_scaler'],
            base_gamma=params['bbox_base_gamma'],
            novel_gamma=params['bbox_novel_gamma'],
        ),
        objects=todd.Config(
            base_scaler=params['object_base_scaler'],
            novel_scaler=params['object_novel_scaler'],
            base_gamma=params['object_base_gamma'],
            novel_gamma=params['object_novel_gamma'],
        ),
        objectness=todd.Config(gamma=params['objectness_gamma']),
    )

    validator = Validator(
        name=args.name,
        model=model,
        dataloader=todd.Config(
            num_workers=1,
            dataset=dict(root=args.root),
        ),
        log=todd.Config(interval=64),
        load_state_dict=todd.Config(),
        state_dict=todd.Config(),
        evaluator=config.validator.dataloader.dataset,
    )
    validator.run()


if __name__ == '__main__':
    main()
