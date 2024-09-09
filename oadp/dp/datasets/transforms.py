__all__ = [
    'LoadOAKE',
    'LoadOAKE_COCO',
    'LoadOAKE_LVIS',
    'PackTrainInputs',
    'PackValInputs',
]

import enum
from typing import Any, cast

import todd.tasks.object_detection as od
import torch
from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes

from oadp.utils import Globals

from .access_layers import AccessLayer, COCOAccessLayer, LVISAccessLayer


class BBoxesFlag(enum.IntEnum):
    BLOCK = 2
    OBJECT = 3


class LoadOAKE:

    def __init__(self, access_layer: AccessLayer) -> None:
        self._access_layer = access_layer

    def _assign_block_labels(
        self,
        gt_bboxes: od.FlattenBBoxesXYXY,
        gt_labels: torch.Tensor,
        block_bboxes: od.FlattenBBoxesXYXY,
    ) -> torch.Tensor:
        num_categories = Globals.categories.num_all
        indices = gt_labels < num_categories  # filter out pseudo labels
        gt_bboxes = gt_bboxes[indices]
        gt_labels = gt_labels[indices]

        block_ids, gt_ids = torch.where(
            block_bboxes.intersections(gt_bboxes) > 0
        )
        block_labels = torch.zeros(
            (len(block_bboxes), num_categories),
            dtype=bool,
        )
        block_labels[block_ids, gt_labels[gt_ids]] = True

        return block_labels

    def _append_bboxes(
        self,
        gt_bboxes: torch.Tensor,
        gt_flags: torch.Tensor,
        bboxes: od.FlattenBBoxesXYXY,
        flag: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_bboxes = torch.cat([gt_bboxes, bboxes.to_tensor().float()])
        gt_flags = torch.cat([gt_flags,
                              torch.full((bboxes.shape[0], ), flag)])
        return gt_bboxes, gt_flags

    def _access(
        self,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        key: str,
    ) -> tuple[
        dict[str, torch.Tensor],
        od.FlattenBBoxesXYXY,
        od.FlattenBBoxesXYXY,
    ]:
        global_, blocks, objects = self._access_layer[key]

        block_bboxes = blocks['bboxes']
        block_labels = self._assign_block_labels(
            od.FlattenBBoxesXYXY(gt_bboxes),
            gt_labels,
            block_bboxes,
        )

        object_bboxes = objects['bboxes'].to(od.FlattenBBoxesXYXY)

        oake = dict(
            clip_global=global_,
            block_labels=block_labels,
            clip_blocks=blocks['embeddings'],
            clip_objects=objects['tensors'],
        )
        return oake, block_bboxes, object_bboxes

    def _load(self, results: dict[str, Any], key: str) -> dict[str, Any]:
        gt_bboxes = cast(BaseBoxes, results['gt_bboxes']).tensor
        gt_labels = results['gt_bboxes_labels']
        gt_flags = torch.from_numpy(results['gt_ignore_flags'])

        oake, block_bboxes, object_bboxes = self._access(
            gt_bboxes,
            gt_labels,
            key,
        )
        results.update(oake)

        gt_bboxes, gt_flags = self._append_bboxes(
            gt_bboxes,
            gt_flags,
            block_bboxes,
            BBoxesFlag.BLOCK,
        )
        gt_bboxes, gt_flags = self._append_bboxes(
            gt_bboxes,
            gt_flags,
            object_bboxes,
            BBoxesFlag.OBJECT,
        )

        cast(BaseBoxes, results['gt_bboxes']).tensor = gt_bboxes
        results['gt_ignore_flags'] = gt_flags.numpy()
        return results


@TRANSFORMS.register_module()
class LoadOAKE_COCO(LoadOAKE):  # noqa: N801 pylint: disable=invalid-name

    def __init__(self) -> None:
        super().__init__(COCOAccessLayer())

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        return self._load(results, results['img_id'])


@TRANSFORMS.register_module()
class LoadOAKE_LVIS(LoadOAKE):  # noqa: N801 pylint: disable=invalid-name

    def __init__(self) -> None:
        super().__init__(LVISAccessLayer())

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        return self._load(results, results['img_path'])


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
