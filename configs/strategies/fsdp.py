_base_ = [
    'ddp.py',
]

trainer = dict(
    strategy=dict(type='FSDPStrategy'),
    wrap_model=dict(sync_module_states=True, use_orig_params=True),
)
