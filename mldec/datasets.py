__all__ = [
    'COCO_48',
    'COCO_17',
    'COCO_48_17',
    'COCO',
    'DebugMixin',
    'CocoDataset',
]

from mmdet.datasets import DATASETS, CocoDataset as _CocoDataset, CustomDataset

from .debug import debug

COCO_48 = (
    'person', 'bicycle', 'car', 'motorcycle', 'truck', 'boat', 'bench', 'bird',
    'horse', 'sheep', 'zebra', 'giraffe', 'backpack', 'handbag', 'skis',
    'kite', 'surfboard', 'bottle', 'spoon', 'bowl', 'banana', 'apple',
    'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'tv',
    'laptop', 'remote', 'microwave', 'oven', 'refrigerator', 'book', 'clock',
    'vase', 'toothbrush', 'train', 'bear', 'suitcase', 'frisbee', 'fork',
    'sandwich', 'toilet', 'mouse', 'toaster',
)
COCO_17 = (
    'bus', 'dog', 'cow', 'elephant', 'umbrella', 'tie', 'skateboard', 'cup',
    'knife', 'cake', 'couch', 'keyboard', 'sink', 'scissors', 'airplane',
    'cat', 'snowboard',
)
COCO_48_17 = COCO_48 + COCO_17

COCO = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
)


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
