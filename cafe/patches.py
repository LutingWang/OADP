_base_ = [
    'one_hot',
    'DyHeadBlock',
    'DebugMixin',
    'CocoDataset',
    'MaskToTensor',
]

from typing import Any, Dict, Sequence, TypeGuard, Union
import torch

from mmcv.parallel import DataContainer as DC
from mmdet.core import BitmapMasks
from mmdet.models.necks.dyhead import DyHeadBlock as _DyHeadBlock
from mmdet.datasets import PIPELINES, DATASETS, CocoDataset as _CocoDataset, CustomDataset

from mldec.debug import debug


def _is_tensor_sequence(data) -> TypeGuard[Sequence[torch.Tensor]]:
    if not isinstance(data, Sequence):
        return False
    return all(isinstance(datum, torch.Tensor) for datum in data)


def one_hot(labels: Union[torch.Tensor, Sequence[torch.Tensor]], num_classes: int) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        assert labels.ndim == 2
        y = labels.new_zeros(
            labels.shape[0],
            num_classes,
            dtype=bool,
        )
    elif _is_tensor_sequence(labels):
        assert all(label.ndim == 1 for label in labels)
        y = labels[0].new_zeros(
            len(labels),
            num_classes,
            dtype=bool,
        )
    else:
        raise TypeError(f"Unsupported type {type(labels)}")

    for i, label in enumerate(labels):
        y[i, label] = True
    return y


class DyHeadBlock(_DyHeadBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_conv_high = None
        self.spatial_conv_low = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward([x])[0]


class DebugMixin(CustomDataset):

    def __len__(self) -> int:
        if debug.DRY_RUN:
            return 6
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


@PIPELINES.register_module()
class MaskToTensor:

    def __init__(self, num_classes: int) -> None:
        self._num_classes = num_classes

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        gt_masks: BitmapMasks = results['gt_masks']
        tensor = torch.zeros((self._num_classes, gt_masks.height, gt_masks.width), dtype=torch.bool)
        for i, gt_label in enumerate(results['gt_labels']):
            tensor[gt_label] += gt_masks.masks[i]
        results['gt_masks_tensor'] = DC(tensor, stack=True, padding_value=0)
        return results
