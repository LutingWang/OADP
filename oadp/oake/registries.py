__all__ = [
    'OAKEDatasetRegistry',
    'OAKERunnerRegistry',
]

from ..registries import OAKERegistry
from todd.registries import RunnerRegistry, DatasetRegistry


class OAKEDatasetRegistry(OAKERegistry, DatasetRegistry):
    pass


class OAKERunnerRegistry(OAKERegistry, RunnerRegistry):
    pass
