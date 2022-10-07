_base_ = [
    'multilabel.py',
]

model = dict(
    multilabel_classifier=dict(
        out_features=65,
        split='COCO_48_17',
        num_base_classes=48,
    ),
)
