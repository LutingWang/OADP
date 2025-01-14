from typing import Any

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)
dataset = _kwargs_['dataset']  # COCO, LVIS, Object365
branch = _kwargs_['branch']  # Object, Block, or Global
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
trainer = dict(
    type=f'{branch}Validator',
    callbacks=callbacks,
    dataloader=dataloader,
    dataset=dict(
        type=f'OAKEDatasetRegistry.{dataset}{branch}Dataset',
        auto_fix=auto_fix,
        split='train',
    ),
)
validator = dict(
    type=f'{branch}Validator',
    callbacks=callbacks,
    dataloader=dataloader,
    dataset=dict(
        type=f'OAKEDatasetRegistry.{dataset}{branch}Dataset',
        auto_fix=auto_fix,
        split='val',
    ),
)

_export_ = dict(trainer=trainer, validator=validator)
