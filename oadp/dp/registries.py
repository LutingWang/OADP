__all__ = [
    'DPDatasetRegistry',
    'DPTransformRegistry',
]

from todd.registries import DatasetRegistry
from todd.registries import TransformRegistry

from ..registries import DPRegistry


class DPDatasetRegistry(DPRegistry, DatasetRegistry):
    pass


class DPTransformRegistry(DPRegistry, TransformRegistry):
    pass
