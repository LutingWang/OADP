__all__ = [
    'OAKECallbackRegistry',
]

import todd

from ..registries import OAKERegistry
from todd.runners import CallbackRegistry


class OAKECallbackRegistry(OAKERegistry, CallbackRegistry):
    pass
