__all__ = [
    'OAKEDatasetRegistry',
    'OAKERunnerRegistry',
    'OAKEModelRegistry',
]

from ..registries import OAKERegistry
from todd.registries import RunnerRegistry, DatasetRegistry, ModelRegistry


class OAKEDatasetRegistry(OAKERegistry, DatasetRegistry):
    pass


class OAKERunnerRegistry(OAKERegistry, RunnerRegistry):
    pass


class OAKEModelRegistry(OAKERegistry, ModelRegistry):
    pass
