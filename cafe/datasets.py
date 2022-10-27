_base_ = [
    'DebugMixin',
    'CocoDataset',
    'CocoDataset4817',
    'LoadCLIPFeatures',
]

import contextlib
import io
import json
import logging
import random
from typing import Any, Dict, Optional, Tuple

from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
from mmdet.core import BitmapMasks
from mmdet.datasets import PIPELINES, DATASETS, CocoDataset as _CocoDataset, LVISV1Dataset as _LVISV1Dataset, CustomDataset
from mmdet.datasets.pipelines import LoadAnnotations as _LoadAnnotations
from mmdet.datasets.api_wrappers import COCOeval
import numpy as np
import torch
import todd

import mldec
from mldec import debug


class DebugMixin(CustomDataset):

    def __len__(self) -> int:
        if debug.DRY_RUN:
            return 3
        return super().__len__()

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if debug.DRY_RUN:
            data_infos = data_infos[:len(self)]
        return data_infos

    def load_proposals(self, *args, **kwargs):
        proposals = super().load_proposals(*args, **kwargs)
        if debug.DRY_RUN:
            proposals = proposals[:len(self)]
        return proposals

    def evaluate(self, *args, **kwargs):
        kwargs.pop('gpu_collect', None)
        kwargs.pop('tmpdir', None)
        return super().evaluate(*args, **kwargs)


@DATASETS.register_module(force=True)
class CocoDataset(DebugMixin, _CocoDataset):

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if debug.DRY_RUN:
            self.coco.dataset['images'] = \
                self.coco.dataset['images'][:len(self)]
            self.img_ids = [img['id'] for img in self.coco.dataset['images']]
            self.coco.dataset['annotations'] = [
                ann for ann in self.coco.dataset['annotations']
                if ann['image_id'] in self.img_ids
            ]
            self.coco.imgs = {
                img['id']: img
                for img in self.coco.dataset['images']
            }
        return data_infos


@DATASETS.register_module()
class CocoDataset4817(CocoDataset):
    CLASSES = mldec.COCO_48_17

    def summarize(self, cocoEval: COCOeval, logger=None, split_name: Optional[str] = None) -> dict:
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        eval_results = {
            'bbox_' + metric: round(cocoEval.stats[i], 4)
            for i, metric in enumerate([
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
            ])
        }
        eval_results['bbox_mAP_copypaste'] = ' '.join(f'{ap:.4f}' for ap in eval_results.values())
        if split_name is not None:
            print_log(f'Evaluate split *{split_name}*', logger=logger)
            eval_results = {f'{split_name}_{k}': v for k, v in eval_results.items()}
        print_log('\n' + redirect_string.getvalue(), logger=logger)
        return eval_results

    def evaluate(
        self,
        results,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        iou_thrs: Optional[Tuple[float]] = None,
        max_dets: Optional[Tuple[int]] = (100, 300, 1000),
        gpu_collect=False,
    ) -> dict:
        if metric == 'proposal_fast':
            return super().evaluate(
                results,
                metric,
                logger,
                jsonfile_prefix,
                False,
                iou_thrs=iou_thrs,
            )
        predictions = self._det2json(results)

        cocoGt = self.coco
        try:
            cocoDt = self.coco.loadRes(predictions)
        except IndexError:
            print_log(
                'The testing results of the whole dataset is empty.',
                logger=logger, level=logging.ERROR,
            )
            return {}
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        if iou_thrs is not None:
            cocoEval.params.iouThrs = np.array(iou_thrs)
        if max_dets is not None:
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.evaluate()
        cocoEval.accumulate()

        eval_results = self.summarize(cocoEval, logger)

        precision: np.ndarray = cocoEval.eval['precision']  # Thresh x Recall x K x Area x MaxDets
        recall: np.ndarray = cocoEval.eval['recall']  # Thresh x K x Area x MaxDets
        assert len(self.cat_ids) == precision.shape[2] == recall.shape[1], f"{len(self.cat_ids)}, {precision.shape}, {recall.shape}"

        cocoEval.eval['precision'] = precision[:, :, :48, :, :]
        cocoEval.eval['recall'] = recall[:, :48, :, :]
        eval_results.update(self.summarize(cocoEval, logger, split_name='coco 48'))

        cocoEval.eval['precision'] = precision[:, :, 48:, :, :]
        cocoEval.eval['recall'] = recall[:, 48:, :, :]
        eval_results.update(self.summarize(cocoEval, logger, split_name='coco 17'))

        return eval_results


@DATASETS.register_module()
class CocoDataset48Ext(CocoDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            classes=todd.globals_.COCO_48_EXT,
            **kwargs,
        )


@DATASETS.register_module(force=True)
class LVISV1Dataset(DebugMixin, _LVISV1Dataset):

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if debug.DRY_RUN:
            self.coco.dataset['images'] = \
                self.coco.dataset['images'][:len(self)]
            self.img_ids = [img['id'] for img in self.coco.dataset['images']]
            self.coco.dataset['annotations'] = [
                ann for ann in self.coco.dataset['annotations']
                if ann['image_id'] in self.img_ids
            ]
            self.coco.imgs = {
                img['id']: img
                for img in self.coco.dataset['images']
            }
        return data_infos


@DATASETS.register_module()
class LVISV1Dataset866337(LVISV1Dataset):
    CLASSES = mldec.LVIS


@PIPELINES.register_module()
class LoadCLIPFeatures:

    def __init__(
        self,
        task_name: str,
        images: Optional[Dict[str, Any]] = None,
        regions: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._task_name = task_name

        if images is not None:
            self._images = todd.datasets.ACCESS_LAYERS.build(
                images,
                default_args=dict(task_name=task_name),
            )
        else:
            self._images = None

        if regions is not None:
            self._regions_as_proposals = regions.pop('as_proposals', False)
            self._regions = todd.datasets.ACCESS_LAYERS.build(
                regions,
                default_args=dict(task_name=task_name),
            )
        else:
            self._regions = None

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        key = f'{results["img_info"]["id"]:012d}'
        if debug.CPU:
            key = '000000000139'

        if self._images is not None:
            image = self._images[key]
            results['clip_image'] = image['image'].squeeze(0)

            clip_patches = todd.base.BBoxesXYWH(image['bboxes'])
            if 'gt_bboxes' in results:
                gt_bboxes = todd.base.BBoxesXYXY(results['gt_bboxes'])
                patch_ids, bbox_ids = torch.where(clip_patches.intersections(gt_bboxes) > 0)
                labels = results['gt_labels'][bbox_ids]
                valid_inds = labels < todd.globals_.num_classes
                results['clip_patch_labels'] = np.zeros((len(clip_patches), todd.globals_.num_classes), dtype=bool)
                results['clip_patch_labels'][patch_ids[valid_inds], labels[valid_inds]] = True

            results['clip_patch_feats'] = image['patches']
            results['clip_patches'] = todd.base.BBoxesXYXY(clip_patches).to_tensor().float().numpy()
            results['bbox_fields'].append('clip_patches')

        if self._regions is not None:
            regions = self._regions[key]
            clip_bbox_feats = regions['patches']
            clip_bboxes = regions['bboxes']
            inds = (clip_bboxes[:, 2] > clip_bboxes[:, 0] + 4) & (clip_bboxes[:, 3] > clip_bboxes[:, 1] + 4)  # TODO: update with todd
            results['clip_bbox_feats'] = clip_bbox_feats[inds]
            results['clip_bboxes'] = clip_bboxes[inds].float().numpy()
            results['bbox_fields'].append('clip_bboxes')
            if self._regions_as_proposals:
                results['proposals'] = results['clip_bboxes']
                results['bbox_fields'].append('proposals')

        return results


@PIPELINES.register_module()
class LoadDetproFeatures:

    def __init__(
        self,
        images: Optional[Dict[str, Any]] = None,
        regions: Optional[Dict[str, Any]] = None,
    ) -> None:
        if images is not None:
            self._images = todd.datasets.ACCESS_LAYERS.build(
                images,
            )
        else:
            self._images = None

        if regions is not None:
            self._regions = todd.datasets.ACCESS_LAYERS.build(
                regions,
            )
        else:
            self._regions = None

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if self._images is not None:
            key = results['img_info']['filename']\
                .replace('.jpg', '')\
                .replace('train2017/', 'train/')\
                .replace('val2017/', 'val/')
            image = self._images[key]
            results['clip_image'] = image['image'].squeeze(0)

        if self._regions is not None:
            key = results['img_info']['filename']\
                .replace('.jpg', '')\
                .replace('train2017/', '')\
                .replace('val2017/', '')
            if debug.DRY_RUN:
                key = '000000000030'

            clip_patches = self._regions[key]
            clip_bboxes = results.pop('proposals')

            results['bbox_fields'].remove('proposals')
            results['bbox_fields'].append('clip_bboxes')

            inds = (clip_bboxes[:, 2] > clip_bboxes[:, 0] + 32) & (clip_bboxes[:, 3] > clip_bboxes[:, 1] + 32)  # TODO: update with todd
            results['clip_bbox_feats'] = clip_patches[inds]
            results['clip_bboxes'] = clip_bboxes[inds]

        return results
