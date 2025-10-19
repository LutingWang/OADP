_base_ = '../ov_distill.py'

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/lvis_v1_val.json',
))

val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=_base_.data_root +
    'annotations/lvis_v1_val.json')

test_dataloader = val_dataloader
test_evaluator = val_evaluator