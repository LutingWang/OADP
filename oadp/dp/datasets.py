__all__ = [
    'DebugMixin',
    'OV_LVIS',
    'LoadCLIPFeatures',
    'PackInputs'
]

import contextlib
import io
from typing import Any, Mapping, cast
import os.path as osp
import tempfile
import numpy as np
import todd
import torch
from lvis import LVIS
from mmcv.transforms import to_tensor
from mmengine.fileio import load
from mmdet.datasets import (
    CocoDataset,
    BaseDetDataset,
    LVISV1Dataset,
)
from mmdet.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.evaluation.metrics import CocoMetric
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.datasets.transforms import PackDetInputs
from todd.datasets import AccessLayerRegistry as ALR

from ..base import Globals, coco, lvis


class DebugMixin(BaseDetDataset):

    def __len__(self) -> int:
        if todd.Store.DRY_RUN:
            return 3
        return super().__len__()

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if not todd.Store.DRY_RUN:
            return data_infos

        coco: COCO | LVIS = getattr(self, 'coco')
        dataset = cast(dict[str, Any], coco.dataset)
        images = dataset['images'][:len(self)]
        image_ids = [img['id'] for img in images]
        id2image = {img['id']: img for img in images}
        annotations = [
            ann for ann in dataset['annotations']
            if ann['image_id'] in image_ids
        ]

        self.img_ids = image_ids
        coco.imgs = id2image
        dataset.update(
            images=images,
            annotations=annotations,
        )
        return data_infos[:len(self)]

    def load_proposals(self, *args, **kwargs):
        proposals = super().load_proposals(*args, **kwargs)
        if todd.Store.DRY_RUN:
            proposals = proposals[:len(self)]
        return proposals


@DATASETS.register_module(name='CocoDataset', force=True)
class CocoDataset_(DebugMixin, CocoDataset):
    pass


@DATASETS.register_module(name='LVISV1Dataset', force=True)
class LVISV1Dataset_(DebugMixin, LVISV1Dataset):
    pass


@METRICS.register_module()
class OVCOCOMetric(CocoMetric):

    def summarize(self, cocoEval: COCOeval, prefix: str) -> dict[str, Any]:
        string_io = io.StringIO()
        with contextlib.redirect_stdout(string_io):
            cocoEval.summarize()
        todd.logger.info(f'Evaluate *{prefix}*\n{string_io.getvalue()}')

        stats = {
            s: f'{cocoEval.stats[i]:.04f}'
            for i, s in enumerate(['', '50', '75', 's', 'm', 'l'])
        }
        stats['copypaste'] = ' '.join(stats.values())
        return {f'{prefix}_bbox_mAP_{k}': v for k, v in stats.items()}

    def compute_metrics(self, results, *args, **kwargs) -> dict[str, float]:
        _, preds = zip(*results)
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix
        
        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()
        
        result_files = self.results2json(preds, outfile_prefix)
        predictions = load(result_files['bbox'])
        coco_dt = self._coco_api.loadRes(predictions)

        coco_eval = COCOeval(self._coco_api, coco_dt, 'bbox')
        coco_eval.params.catIds = self.cat_ids
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.maxDets = [100, 300, 1000]

        coco_eval.evaluate()
        coco_eval.accumulate()

        # iou_thrs x recall x k x area x max_dets
        precision: np.ndarray = coco_eval.eval['precision']
        # iou_thrs x k x area x max_dets
        recall: np.ndarray = coco_eval.eval['recall']
        assert len(self.cat_ids) == precision.shape[2] == recall.shape[1], (
            f"{len(self.cat_ids)}, {precision.shape}, {recall.shape}"
        )

        all_ = self.summarize(
            coco_eval, f'COCO_{coco.num_bases}_{coco.num_novels}'
        )

        coco_eval.eval['precision'] = precision[:, :, :coco.num_bases, :, :]
        coco_eval.eval['recall'] = recall[:, :coco.num_bases, :, :]
        bases = self.summarize(coco_eval, f'COCO_{coco.num_bases}')

        coco_eval.eval['precision'] = precision[:, :, coco.num_bases:, :, :]
        coco_eval.eval['recall'] = recall[:, coco.num_bases:, :, :]
        novels = self.summarize(coco_eval, f'COCO_{coco.num_novels}')

        return all_ | bases | novels


@DATASETS.register_module()
class OV_LVIS(LVISV1Dataset_):
    CLASSES = lvis.all_


@TRANSFORMS.register_module()
class LoadCLIPFeatures:

    def __init__(
        self,
        default: todd.Config,
        globals_: todd.Config | None = None,
        blocks: todd.Config | None = None,
        objects: todd.Config | None = None,
    ) -> None:
        assert (
            globals_ is not None or blocks is not None or objects is not None
        )
        if todd.Store.TRAIN_WITH_VAL_DATASET:
            task_name: str = default.task_name
            default.task_name = task_name.replace('train', 'val')
        self._globals: Mapping[str, torch.Tensor] | None = (
            None if globals_ is None else ALR.build(globals_, default)
        )
        self._blocks: Mapping[str, dict[str, torch.Tensor]] | None = (
            None if blocks is None else ALR.build(blocks, default)
        )
        self._objects: Mapping[str, dict[str, torch.Tensor]] | None = (
            None if objects is None else ALR.build(objects, default)
        )

        if todd.Store.DRY_RUN:
            keys = [
                set(mapping.keys())
                for mapping in [self._globals, self._blocks, self._objects]
                if mapping is not None
            ]
            self.__key = set.intersection(*keys).pop()

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        key = (
            self.__key
            if todd.Store.DRY_RUN else f'{results["img_id"]:012d}'
        )

        if self._globals is not None:
            global_ = self._globals[key]
            results['clip_global'] = global_.squeeze(0)

        if self._blocks is not None:
            blocks = self._blocks[key]
            block_bboxes = blocks['bboxes']
            if 'gt_bboxes' in results:
                num_all = Globals.categories.num_all
                gt_bboxes = results['gt_bboxes']
                gt_labels = results['gt_bboxes_labels']
                indices = gt_labels < num_all  # filter out pseudo labels
                gt_bboxes = gt_bboxes[indices]
                gt_labels = gt_labels[indices]
                block_ids, gt_ids = torch.where(
                    todd.BBoxesXYXY(block_bboxes)
                    & todd.BBoxesXYXY(gt_bboxes.tensor) > 0
                )
                block_labels = np.zeros(
                    (block_bboxes.shape[0], num_all),
                    dtype=bool,
                )
                block_labels[block_ids, gt_labels[gt_ids]] = True
                results['block_labels'] = to_tensor(block_labels)
            results['clip_blocks'] = blocks['embeddings']
            results['block_bboxes'] = block_bboxes.float()

        if self._objects is not None:
            objects = self._objects[key]
            object_bboxes = objects['bboxes']
            indices = todd.BBoxesXYXY(object_bboxes).indices(min_wh=(4, 4))
            results['clip_objects'] = objects['embeddings'][indices]
            results['object_bboxes'] = object_bboxes[indices].float()

        return results

@TRANSFORMS.register_module()
class PackInputs(PackDetInputs):
    def __init__(self, extra_keys, 
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        super().__init__(meta_keys)
        self.keys = extra_keys
    
    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        for key in self.keys:
            packed_results[key] = results[key]
        return packed_results

@MODELS.register_module()
class DataPreprocessor(DetDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:
        pack_data = super().forward(data, training)
        for key, value in data.items():
            if key not in ['inputs', 'data_samples']:
                pack_data[key] = value
        return pack_data