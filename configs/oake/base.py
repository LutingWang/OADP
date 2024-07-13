data_root = 'data/coco/'
annotation_root = data_root + 'annotations/'
callbacks = [
    dict(
        type='LogCallback',
        interval=50,
        collect_env=dict(),
        with_file_handler=True,
        eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
        priority=dict(init=-1),
    ),
    dict(type='OADPCallbackRegistry.BaseCallback'),
]
dataloader = dict(num_workers=2, collate_fn=dict(type='oadp_collate_fn'))
trainer = dict(
    callbacks=callbacks,
    dataset=dict(
        access_layer=dict(data_root=data_root, task_name='train2017'),
        keys=dict(
            annotation_file=annotation_root + 'instances_train2017.json',
        ),
    ),
    dataloader=dataloader,
    output_dir=dict(task_name='train2017'),
)
validator = dict(
    callbacks=callbacks,
    dataset=dict(
        access_layer=dict(data_root=data_root, task_name='val2017'),
        keys=dict(
            annotation_file=annotation_root + 'instances_val2017.json',
        ),
    ),
    dataloader=dataloader,
    output_dir=dict(task_name='val2017'),
)
