from typing import Any

from todd.configs import PyConfig

_kwargs_: dict[str, Any]
_kwargs_ = dict(_kwargs_)

_kwargs_.setdefault('branch', 'Global')
_kwargs_.setdefault('strategy', 'cuda')

_base_ = [
    PyConfig.load('configs/oake/interface.py', **_kwargs_),
]

runner = dict(
    model=dict(type='ram_plus', expand_mask_size=None, adaptive=False),
)
custom_imports = [
    'oadp.oake.globals_',
]

_export_ = dict(
    trainer=runner,
    validator=runner,
    custom_imports=custom_imports,
)
