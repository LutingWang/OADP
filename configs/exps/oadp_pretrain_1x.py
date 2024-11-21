_base_ = [
    '../datasets/pretrain.py',
    '../schedule/schedule_1x.py',
    '../models/oadp.py',
    '../default_runtime.py'
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

auto_scale_lr = dict(base_batch_size=64)

root = '/mnt/data1/wlt/workspace/OADP/'
work_dir = f'{root}work_dirs/pretrain'