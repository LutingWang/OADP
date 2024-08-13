__all__ = [
    'OAKECallbackRegistry',
]

from ..registries import OAKERunnerRegistry
from todd.runners import CallbackRegistry


class OAKECallbackRegistry(OAKERunnerRegistry, CallbackRegistry):
    pass
