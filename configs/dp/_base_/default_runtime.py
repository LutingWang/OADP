trainer = dict(
    custom_hooks=[dict(type='NumClassCheckHook')],
    log_level='INFO',
    load_from=None,
    resume_from=None,
    opencv_num_threads=0,
    mp_start_method='fork',
    auto_scale_lr=dict(enable=False, base_batch_size=16),
    fp16=dict(loss_scale=dict(init_scale=64.0)),
)
