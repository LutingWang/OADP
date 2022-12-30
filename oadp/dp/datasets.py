__all__ = [
    'DebugMixin',
    'CocoDataset4817',
]

import contextlib
import io
from typing import Any, Dict, Optional, Tuple

import numpy as np
import todd
from mmdet.datasets import DATASETS
from mmdet.datasets import CocoDataset as _CocoDataset
from mmdet.datasets import CustomDataset
from mmdet.datasets.api_wrappers import COCOeval

from ..base import Categories


class DebugMixin(CustomDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._logger = todd.get_logger()

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

    def evaluate(self, *args, **kwargs):  # TODO: delete this
        kwargs.pop('gpu_collect', None)
        kwargs.pop('tmpdir', None)
        return super().evaluate(*args, **kwargs)


@DATASETS.register_module()
class CocoDataset4817(DebugMixin, _CocoDataset):
    CLASSES = Categories.COCO_48_17

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

    def summarize(
        self,
        cocoEval: COCOeval,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()

        message = '\n' + redirect_string.getvalue()
        if split is not None:
            message = f'Evaluate split *{split}*' + message
        self._logger.info(message)

        eval_results = dict(
            zip(
                ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'],
                cocoEval.stats,
            ),
        )
        eval_results = {
            f'bbox_{k}': round(v, 4)
            for k, v in eval_results.items()
        }
        eval_results['bbox_mAP_copypaste'] = ' '.join(
            map(str, eval_results.values()),
        )
        if split is not None:
            eval_results = {f'{split}_{k}': v for k, v in eval_results.items()}
        return eval_results

    def evaluate(
        self,
        results,
        iou_thrs: Optional[Tuple[float, ...]] = None,
        max_dets: Optional[Tuple[int, ...]] = (100, 300, 1000),
    ) -> dict:
        predictions = self._det2json(results)
        try:
            cocoDt = self.coco.loadRes(predictions)
        except IndexError:
            self._logger.error(
                'The testing results of the whole dataset is empty.',
            )
            return {}

        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        if iou_thrs is not None:
            cocoEval.params.iouThrs = np.array(iou_thrs)
        if max_dets is not None:
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.evaluate()
        cocoEval.accumulate()

        # iou_thrs x recall x k x area x max_dets
        precision: np.ndarray = cocoEval.eval['precision']
        # iou_thrs x k x area x max_dets
        recall: np.ndarray = cocoEval.eval['recall']
        assert len(self.cat_ids) == precision.shape[2] == recall.shape[1], (
            f"{len(self.cat_ids)}, {precision.shape}, {recall.shape}"
        )

        eval_results = self.summarize(cocoEval)

        cocoEval.eval['precision'] = precision[:, :, :48, :, :]
        cocoEval.eval['recall'] = recall[:, :48, :, :]
        eval_results.update(self.summarize(cocoEval, split='COCO_48'))

        cocoEval.eval['precision'] = precision[:, :, 48:, :, :]
        cocoEval.eval['recall'] = recall[:, 48:, :, :]
        eval_results.update(self.summarize(cocoEval, split='COCO_17'))

        return eval_results
