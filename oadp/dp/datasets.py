__all__ = [
    'DebugMixin',
    'OV_COCO',
]

import contextlib
import io
from typing import Any

import numpy as np
import todd
from mmdet.datasets import DATASETS, CocoDataset, CustomDataset
from mmdet.datasets.api_wrappers import COCOeval

from ..base import Globals, coco


class DebugMixin(CustomDataset):

    def __len__(self) -> int:
        if todd.utils.BaseRunner.Store.DRY_RUN:
            return 3
        return super().__len__()

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if todd.utils.BaseRunner.Store.DRY_RUN:
            data_infos = data_infos[:len(self)]
        return data_infos

    def load_proposals(self, *args, **kwargs):
        proposals = super().load_proposals(*args, **kwargs)
        if todd.utils.BaseRunner.Store.DRY_RUN:
            proposals = proposals[:len(self)]
        return proposals


@DATASETS.register_module()
class OV_COCO(DebugMixin, CocoDataset):
    CLASSES = coco.all_

    def load_annotations(self, *args, **kwargs):
        data_infos = super().load_annotations(*args, **kwargs)
        if not todd.utils.BaseRunner.Store.DRY_RUN:
            return data_infos

        images = self.coco.dataset['images'][:len(self)]
        image_ids = [img['id'] for img in images]
        id2image = {img['id']: img for img in images}
        annotations = [
            ann for ann in self.coco.dataset['annotations']
            if ann['image_id'] in image_ids
        ]

        self.img_ids = image_ids
        self.coco.imgs = id2image
        self.coco.dataset.update(
            images=images,
            annotations=annotations,
        )
        return data_infos

    def summarize(self, cocoEval: COCOeval, prefix: str) -> dict[str, Any]:
        string_io = io.StringIO()
        with contextlib.redirect_stdout(string_io):
            cocoEval.summarize()
        Globals.logger.info(f'Evaluate *{prefix}*\n{string_io.getvalue()}')

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
            Globals.logger.error('The testing results is empty')
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
