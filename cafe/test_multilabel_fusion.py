import argparse
import functools
import os
import pathlib
import sys
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple
import einops

import nni
import todd
import torch
import torch.nn as nn
import torch.distributed
import torch.utils.data
import torch.utils.data.distributed
from mmdet.datasets import build_dataset
from mmdet.core import multiclass_nms, bbox2result

from mldec import debug, BaseRunner

sys.path.insert(0, '')
import cafe
import mldec


class Batch(NamedTuple):
    image_ids: List[int]
    bboxes: torch.Tensor
    bbox_scores: torch.Tensor
    image_scores: torch.Tensor
    objectness: torch.Tensor
    multilabel_logits: Optional[torch.Tensor]
    patch_logits: Optional[torch.Tensor]
    patch_bboxes: Optional[torch.Tensor]


class Dataset(todd.datasets.PthDataset):

    def __init__(self, root: str) -> None:
        super().__init__(
            access_layer=dict(
                type='PthAccessLayer',
                data_root=root,
            ),
        )
        self._patch_access_layer = todd.datasets.PthAccessLayer(
            data_root='data/coco/embeddings',
            task_name='val',
        )

    def __getitem__(self, index: int) -> Batch:
        key = self._keys[index]
        item: Dict[str, Any] = self._access_layer[key]
        item.update(
            patch_bboxes=self._patch_access_layer[key]['bboxes'],
            image_ids=key,
        )
        return Batch(**item)


class Model(todd.Module):

    def __init__(
        self,
        *args,
        classifier: str,
        base_ensemble_mask: float,
        novel_ensemble_mask: float,
        nms_cfg: Dict[str, Any],
        bbox_cfg: Dict[str, Any],
        image_cfg: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._ensemble_mask = torch.empty(66, device='cuda')
        self._ensemble_mask[:48] = base_ensemble_mask
        self._ensemble_mask[48:-1] = novel_ensemble_mask
        self._nms_cfg = nms_cfg
        self._bbox_cfg = bbox_cfg
        self._image_cfg = image_cfg

    def _classify(
        self,
        scores: torch.Tensor,
        patch_relevance: torch.Tensor,
        objectness: torch.Tensor,
        batch: Batch,
        cfg: Dict[str, Any],
    ) -> torch.Tensor:
        scores *= cfg['score_scaler']
        scores = scores.softmax(-1)

        scores = (
            scores ** cfg['score_gamma']
            * objectness ** cfg['objectness_gamma']
        )
        return scores

    def forward(self, batch: Batch) -> Dict[int, Any]:
        patch_relevance = todd.BBoxesXYXY(batch.bboxes).intersections(
            todd.BBoxesXYXY(batch.patch_bboxes),
        )
        patch_relevance[patch_relevance <= 0] = float('-inf')

        objectness = einops.rearrange(batch.objectness, 'b n -> b n 1')

        bbox_scores = self._classify(batch.bbox_scores, patch_relevance, objectness, batch, self._bbox_cfg)
        image_scores = self._classify(batch.image_scores, patch_relevance, objectness, batch, self._image_cfg)

        ensemble_score = (
            bbox_scores ** self._ensemble_mask
            * image_scores ** (1 - self._ensemble_mask)
        )

        ensemble_score = ensemble_score.float()
        return {
            image_id: bbox2result(
                *multiclass_nms(
                    batch.bboxes[i], ensemble_score[i],
                    **self._nms_cfg,
                ),
                65,
            )
            for i, image_id in enumerate(batch.image_ids)
        }


class Runner(BaseRunner):

    def _build_dataloader(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> Tuple[
        Dataset,
        Optional[torch.utils.data.distributed.DistributedSampler],
        torch.utils.data.DataLoader,
    ]:
        assert config is not None
        dataset = Dataset(**config.dataset)
        if todd.get_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False,
            )
        else:
            sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
        )
        return dataset, sampler, dataloader

    def _build_model(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> nn.Module:
        assert config is not None
        return Model(**config)

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        memo['result_dict'] = dict()
        return memo

    def _run_iter(
        self,
        *args,
        i: int,
        batch: Batch,
        memo: Dict[str, Any],
        **kwargs,
    ) -> None:
        if not debug.CPU:
            batch = Batch(*[
                field.cuda().float() if isinstance(field, torch.Tensor) else field
                for field in batch
            ])
        result_dict = self._model(batch)
        memo['result_dict'].update(result_dict)

    def _after_run_iter(
        self,
        *args,
        i: int,
        batch: Batch,
        memo: Dict[str, Any],
        log: bool = False,
        **kwargs,
    ) -> Optional[bool]:
        if log and todd.get_rank() == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )
        if log and mldec.debug.DRY_RUN:
            return True

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs) -> float:
        super()._after_run(*args, memo=memo, **kwargs)

        if todd.get_world_size() > 1:
            result_dicts = [None] * todd.get_world_size()
            torch.distributed.all_gather_object(result_dicts, memo['result_dict'])
            if todd.get_rank() != 0:
                return
        else:
            result_dicts = [memo['result_dict']]

        result_dict = functools.reduce(
            lambda a, b: {**a, **b},
            result_dicts,
        )
        evaluator = build_dataset(
            self._config.evaluator,
            default_args=dict(test_mode=True),
        )
        results = [
            result_dict[f'{img_id:012d}']
            for img_id in evaluator.img_ids]
        result = evaluator.evaluate(results)
        mAP = result['coco 17_bbox_mAP_50']
        nni.report_final_result(mAP)
        return mAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('root', type=pathlib.Path)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    params = nni.get_next_parameter()
    if len(params) == 0:
        params = dict(
            base_ensemble_mask=2 / 3,
            novel_ensemble_mask=1 / 3,
            bbox_score_scaler=1,
            bbox_score_gamma=1,
            bbox_objectness_gamma=0,
            image_score_scaler=1,
            image_score_gamma=1,
            image_objectness_gamma=0,
        )
    print(params)

    args = parse_args()

    config = todd.Config.load(args.config)
    config = todd.Config(
        val=dict(
            dataloader=dict(
                batch_size=1,
                num_workers=1,
                dataset=dict(
                    root=args.root,
                ),
            )
        ),
        model=dict(
            classifier=args.root / 'classifier',
            base_ensemble_mask=params['base_ensemble_mask'],
            novel_ensemble_mask=params['novel_ensemble_mask'],
            nms_cfg=dict(
                score_thr=config.model.test_cfg.rcnn.score_thr,
                nms_cfg=config.model.test_cfg.rcnn.nms,
                max_num=config.model.test_cfg.rcnn.max_per_img,
            ),
            bbox_cfg=dict(
                score_scaler=params['bbox_score_scaler'],
                multilabel_logit_scaler=params['bbox_multilabel_logit_scaler'],
                score_gamma=params['bbox_score_gamma'],
                objectness_gamma=params['bbox_objectness_gamma'],
                multilabel_score_gamma=params['bbox_multilabel_score_gamma'],
            ),
            image_cfg=dict(
                score_scaler=params['image_score_scaler'],
                multilabel_logit_scaler=params['image_multilabel_logit_scaler'],
                score_gamma=params['image_score_gamma'],
                objectness_gamma=params['image_objectness_gamma'],
                multilabel_score_gamma=params['image_multilabel_score_gamma'],
            ),
        ),
        evaluator=config.data.test,
        logger=dict(
            interval=16,
        ),
    )

    mldec.debug.init()
    # if not debug.CPU:
    #     torch.distributed.init_process_group(backend='nccl')
    #     torch.cuda.set_device(todd.base.get_local_rank())

    runner = Runner(name=args.name, config=config)
    runner.run()
