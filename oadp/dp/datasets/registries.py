__all__ = [
    'DPAccessLayerRegistry',
]

from todd.datasets import AccessLayerRegistry

from ..registries import DPDatasetRegistry


class DPAccessLayerRegistry(DPDatasetRegistry, AccessLayerRegistry):
    pass
