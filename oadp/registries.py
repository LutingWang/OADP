__all__ = [
    'OADPRegistry',
    'OAKERegistry',
    'DPRegistry',
]

import todd


class OADPRegistry(todd.Registry):
    pass


class OAKERegistry(OADPRegistry):
    pass


class DPRegistry(OADPRegistry):
    pass
