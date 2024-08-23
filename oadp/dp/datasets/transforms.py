__all__ = [
    'LoadOAKE',
    'LoadOAKE_COCO',
    'LoadOAKE_LVIS',
    'PackTrainInputs',
    'PackValInputs',
]

import enum
from typing import Any

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS
import todd.tasks.object_detection as od

from .access_layers import AccessLayer, COCOAccessLayer, LVISAccessLayer

from oadp.utils import Globals


class BBoxesFlag(enum.IntEnum):
    BLOCK = 2
    OBJECT = 3


class LoadOAKE:

    def __init__(self, access_layer: AccessLayer) -> None:
        self._access_layer = access_layer

    def _assign_block_labels(
        self,
        results: dict[str, Any],
        block_bboxes: od.FlattenBBoxesXYXY,
    ) -> torch.Tensor:
        num_all = Globals.categories.num_all
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_bboxes_labels']
        indices = gt_labels < num_all  # filter out pseudo labels
        gt_bboxes = gt_bboxes[indices]
        gt_labels = gt_labels[indices]
        block_ids, gt_ids = torch.where(
            block_bboxes.intersections(od.FlattenBBoxesXYXY(gt_bboxes.tensor))
            > 0
        )
        block_labels = np.zeros(
            (block_bboxes.shape[0], num_all),
            dtype=bool,
        )
        block_labels[block_ids, gt_labels[gt_ids]] = True
        return to_tensor(block_labels)

    def _append_bboxes(
        self,
        results: dict[str, Any],
        bboxes: od.FlattenBBoxesMixin,
        flag: int,
    ) -> dict[str, Any]:
        results['gt_bboxes'].tensor = torch.cat([
            results['gt_bboxes'].tensor,
            bboxes.to_tensor().float()
        ])
        results['gt_ignore_flags'] = np.concatenate([
            results['gt_ignore_flags'],
            np.ones(bboxes.shape[0]) * flag,
        ])
        return results

    def _access(self, results: dict[str, Any], key: str) -> dict[str, Any]:
        global_, blocks, objects = self._access_layer[key]

        results['clip_global'] = global_

        block_bboxes = blocks['bboxes']
        results['block_labels'] = self._assign_block_labels(
            results,
            block_bboxes,
        )
        results['clip_blocks'] = blocks['embeddings']
        results = self._append_bboxes(results, block_bboxes, BBoxesFlag.BLOCK)

        object_bboxes = objects['bboxes'].to(od.FlattenBBoxesXYXY)
        results['clip_objects'] = objects['tensors']
        results = self._append_bboxes(
            results,
            object_bboxes,
            BBoxesFlag.OBJECT,
        )

        return results


@TRANSFORMS.register_module()
class LoadOAKE_COCO(LoadOAKE):  # noqa: N801 pylint: disable=invalid-name

    def __init__(self) -> None:
        super().__init__(COCOAccessLayer())

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        return self._access(results, results['img_id'])


@TRANSFORMS.register_module()
class LoadOAKE_LVIS(LoadOAKE):  # noqa: N801 pylint: disable=invalid-name

    def __init__(self) -> None:
        super().__init__(LVISAccessLayer())

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        return self._access(results, results['img_path'])


@TRANSFORMS.register_module()
class PackTrainInputs(PackDetInputs):
    OAKE_KEYS = [
        'clip_global',
        'clip_blocks',
        'clip_objects',
        'block_labels',
    ]

    def transform(self, results: dict) -> dict:
        flags = results['gt_ignore_flags']
        bboxes = results['gt_bboxes']

        block_indices = flags == BBoxesFlag.BLOCK
        object_indices = flags == BBoxesFlag.OBJECT
        indices = ~(block_indices | object_indices)

        results.update(
            gt_bboxes=bboxes[indices],
            gt_ignore_flags=flags[indices],
        )

        packed_results = super().transform(results)
        packed_results.update(
            ((k, results[k]) for k in self.OAKE_KEYS),
            block_bboxes=bboxes[block_indices],
            object_bboxes=bboxes[object_indices],
        )
        return packed_results


@TRANSFORMS.register_module()
class PackValInputs(PackDetInputs):

    def __init__(self) -> None:
        super().__init__(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            ),
        )
