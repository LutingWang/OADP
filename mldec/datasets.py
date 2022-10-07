__all__ = [
    'COCO_48',
    'COCO_17',
    'COCO_48_17',
    'COCO',
]

from mmdet.datasets import CocoDataset

COCO_48 = (
    'person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
    'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe', 'backpack',
    'handbag', 'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 'bottle',
    'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster',
    'refrigerator', 'book', 'clock', 'vase', 'toothbrush',
)

COCO_17 = (
    'airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
    'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard',
    'sink', 'scissors',
)

COCO_48_17 = COCO_48 + COCO_17

COCO = CocoDataset.CLASSES
