_base_ = [
    'mask.py',
]

model = dict(
    roi_head=dict(
        mask_head=dict(
            num_classes=1203,
        ),
    ),
)
