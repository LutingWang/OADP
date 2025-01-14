# mypy: disable-error-code="misc"

__all__ = [
    'COCOGlobalDataset',
]

from todd.datasets import COCODataset

from ..registries import OAKEDatasetRegistry
from .datasets import GlobalDataset


@OAKEDatasetRegistry.register_()
class COCOGlobalDataset(GlobalDataset, COCODataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)
