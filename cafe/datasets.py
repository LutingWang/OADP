_base_ = [
    'DebugMixin',
    'CocoDataset',
    'MaskToTensor',
    'LoadCLIPFeatures',
]

from typing import Any, Dict

from mmcv.parallel import DataContainer as DC
from mmdet.core import BitmapMasks
from mmdet.datasets import PIPELINES, DATASETS, CocoDataset as _CocoDataset, CustomDataset
from mmdet.datasets.pipelines import LoadAnnotations as _LoadAnnotations
import torch
import todd

from mldec.debug import debug


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


@PIPELINES.register_module()
class LoadCLIPFeatures:

    def __init__(
        self,
        task_name: str,
        images: Dict[str, Any],
        regions: Dict[str, Any],
    ) -> None:
        assert task_name in ['train', 'val']
        self._task_name = task_name
        self._images = todd.datasets.ACCESS_LAYERS.build(images, default_args=dict(task_name=task_name))
        self._regions = todd.datasets.ACCESS_LAYERS.build(regions, default_args=dict(task_name=task_name))

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        key = f'{results["img_info"]["id"]:012d}'
        image = self._images[key]
        results['clip_image'] = image['image'].squeeze(0)
        key = '000000000009'
        if not debug.CPU:
            assert False
        regions = self._regions[key]
        clip_patches = torch.cat([image['patches'], regions['patches']])
        clip_bboxes = torch.cat([image['bboxes'], regions['bboxes']])
        inds = (clip_bboxes[:, 2] > clip_bboxes[:, 0]) & (clip_bboxes[:, 3] > clip_bboxes[:, 1])
        results['clip_patches'] = clip_patches[inds]
        results['clip_bboxes'] = clip_bboxes[inds].float().numpy()
        results['bbox_fields'].append('clip_bboxes')
        return results
