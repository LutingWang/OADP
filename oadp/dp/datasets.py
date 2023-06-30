__all__ = [
    'DebugMixin',
    'OV_COCO',
    'OV_LVIS',
    'LoadCLIPFeatures',
]

import contextlib
import io
from typing import Any, Mapping, cast

import numpy as np
import todd
import torch
from lvis import LVIS
from mmdet.datasets import (
    DATASETS,
    PIPELINES,
    CocoDataset,
    CustomDataset,
    LVISV1Dataset,
)
from mmdet.datasets.api_wrappers import COCO, COCOeval
from todd.datasets import AccessLayerRegistry as ALR

from ..base import Globals, coco, lvis


class DebugMixin(CustomDataset):

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


@DATASETS.register_module()
class OV_COCO(CocoDataset_):
    CLASSES = coco.all_

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

    def evaluate(self, results, *args, **kwargs) -> dict[str, Any]:
        results = self._det2json(results)
        try:
            results = self.coco.loadRes(results)
        except IndexError:
            todd.logger.error('The testing results is empty')
            return dict()

        coco_eval = COCOeval(self.coco, results, 'bbox')
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


@PIPELINES.register_module()
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
            if todd.Store.DRY_RUN else f'{results["img_info"]["id"]:012d}'
        )
        bbox_fields: list[str] = results['bbox_fields']

        if self._globals is not None:
            global_ = self._globals[key]
            results['clip_global'] = global_.squeeze(0)

        if self._blocks is not None:
            blocks = self._blocks[key]
            block_bboxes = blocks['bboxes']
            if 'gt_bboxes' in results:
                num_all = Globals.categories.num_all
                gt_bboxes = results['gt_bboxes']
                gt_labels = results['gt_labels']
                indices = gt_labels < num_all  # filter out pseudo labels
                gt_bboxes = gt_bboxes[indices]
                gt_labels = gt_labels[indices]
                block_ids, gt_ids = torch.where(
                    todd.BBoxesXYXY(block_bboxes)
                    & todd.BBoxesXYXY(torch.tensor(gt_bboxes)) > 0
                )
                block_labels = np.zeros(
                    (block_bboxes.shape[0], num_all),
                    dtype=bool,
                )
                block_labels[block_ids, gt_labels[gt_ids]] = True
                results['block_labels'] = block_labels
            results['clip_blocks'] = blocks['embeddings']
            results['block_bboxes'] = block_bboxes.float().numpy()
            bbox_fields.append('block_bboxes')

        if self._objects is not None:
            objects = self._objects[key]
            object_bboxes = objects['bboxes']
            indices = todd.BBoxesXYXY(object_bboxes).indices(min_wh=(4, 4))
            results['clip_objects'] = objects['embeddings'][indices]
            results['object_bboxes'] = object_bboxes[indices].float().numpy()
            bbox_fields.append('object_bboxes')

        return results
