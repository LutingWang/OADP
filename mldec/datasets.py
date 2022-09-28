__all__ = [
    'COCO_48',
    'COCO_17',
    'COCO_48_17',
    'COCO',
]

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
