__all__ = [
    'DPDatasetRegistry',
]

from todd.registries import DatasetRegistry

from ..registries import DPRegistry


class DPDatasetRegistry(DPRegistry, DatasetRegistry):
    pass
