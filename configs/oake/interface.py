from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
auto_fix = _kwargs_['auto_fix']

_base_ = [
    '../strategies/interface.py',
]

callbacks = [
    dict(
        type='LogCallback',
        interval=50,
        collect_env=dict(),
        with_file_handler=True,
        eta=dict(type='EMA_ETA', ema=dict(decay=0.9)),
        priority=dict(init=-1),
    ),
    dict(type='OAKECallbackRegistry.OAKECallback'),
]
dataloader = dict(batch_size=None, num_workers=2)
dataset = dict(auto_fix=auto_fix)
runner = dict(callbacks=callbacks, dataloader=dataloader, dataset=dataset)

_export_ = dict(trainer=runner, validator=runner)
