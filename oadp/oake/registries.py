__all__ = [
    'OAKEDatasetRegistry',
    'OAKERunnerRegistry',
    'OAKEModelRegistry',
]

from todd.registries import DatasetRegistry, ModelRegistry, RunnerRegistry

from ..registries import OAKERegistry


class OAKEDatasetRegistry(OAKERegistry, DatasetRegistry):
    pass


class OAKERunnerRegistry(OAKERegistry, RunnerRegistry):
    pass


class OAKEModelRegistry(OAKERegistry, ModelRegistry):
    pass
