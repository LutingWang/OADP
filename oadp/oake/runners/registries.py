__all__ = [
    'OAKECallbackRegistry',
]

from todd.runners import CallbackRegistry

from ..registries import OAKERunnerRegistry


class OAKECallbackRegistry(OAKERunnerRegistry, CallbackRegistry):
    pass
