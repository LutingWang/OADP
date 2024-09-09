__all__ = [
    'OADPRegistry',
    'OAKERegistry',
    'DPRegistry',
    'PromptRegistry',
]

import todd


class OADPRegistry(todd.Registry):
    pass


class OAKERegistry(OADPRegistry):
    pass


class DPRegistry(OADPRegistry):
    pass


class PromptRegistry(OADPRegistry):
    pass
