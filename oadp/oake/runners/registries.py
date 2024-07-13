__all__ = [
    'OADPCallbackRegistry',
]

import todd

from ...registries import OADPRegistry


class OADPCallbackRegistry(todd.runners.CallbackRegistry, OADPRegistry):
    pass
