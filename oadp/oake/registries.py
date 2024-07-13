__all__ = [
    'OADPDatasetRegistry',
    'OADPRunnerRegistry',
]

import todd

from ..registries import OADPRegistry


class OADPDatasetRegistry(todd.registries.DatasetRegistry, OADPRegistry):
    pass


class OADPRunnerRegistry(todd.registries.RunnerRegistry, OADPRegistry):
    pass
