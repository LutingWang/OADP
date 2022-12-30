interval = 2000
trainer = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    optimizer_config=dict(grad_clip=None),
    lr_config=dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=0.001,
        by_epoch=False,
        step=[3e4],
    ),
    runner=dict(type='IterBasedRunner', max_iters=4e4),
    log_config=dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
        ],
    ),
    workflow=[('train', interval)],
    checkpoint_config=dict(
        by_epoch=False,
        interval=interval,
        save_last=True,
    ),
    evaluation=dict(interval=interval),
)
