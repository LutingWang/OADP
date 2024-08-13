__all__ = [
    'COCOBlockDataset',
]

from todd.datasets import COCODataset

from .datasets import BlockDatasetMixin
from ..registries import OAKEDatasetRegistry
from .datasets import Batch


@OAKEDatasetRegistry.register_()
class COCOBlockDataset(BlockDatasetMixin, COCODataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)

    def _getitem(self, index: int) -> Batch:
        key, image = self._access(index)
        bboxes, blocks = self._partition(image)
        return Batch(id_=key, bboxes=bboxes, blocks=blocks)
