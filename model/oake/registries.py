__all__ = [
    'OAKEDatasetRegistry',
    'OAKERunnerRegistry',
    'OAKEModelRegistry',
]
import todd
from todd.registries import DatasetRegistry, ModelRegistry, RunnerRegistry

class OADPRegistry(todd.Registry):
    pass


class OAKERegistry(OADPRegistry):
    pass


class OAKEDatasetRegistry(OAKERegistry, DatasetRegistry):
    pass


class OAKERunnerRegistry(OAKERegistry, RunnerRegistry):
    pass


class OAKEModelRegistry(OAKERegistry, ModelRegistry):
    pass
