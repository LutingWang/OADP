_base_ = ['base.py']

model = dict(
    type='CLIPModel',
)

val_evaluator = [
    dict(
        type='MutiLabelMetric',
        threshold=0.1,
    ),
]