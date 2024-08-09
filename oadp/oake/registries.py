__all__ = [
    'OAKERegistry',
    'OAKEDatasetRegistry',
    'OAKERunnerRegistry',
]

from ..registries import OADPRegistry
from todd.registries import RunnerRegistry, DatasetRegistry


class OAKERegistry(OADPRegistry):
    pass


class OAKEDatasetRegistry(OAKERegistry, DatasetRegistry):
    pass


class OAKERunnerRegistry(OAKERegistry, RunnerRegistry):
    pass
