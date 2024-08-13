__all__ = [
    'COCOGlobalDataset',
]

from todd.datasets import COCODataset
from todd.datasets.coco import T

from .datasets import GlobalDataset
from ..registries import OAKEDatasetRegistry


@OAKEDatasetRegistry.register_()
class COCOGlobalDataset(GlobalDataset, COCODataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)
