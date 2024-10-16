__all__ = [
    'V3DetBlockDataset',
]

from todd.datasets import V3DetDataset

from ..utils import V3DetMixin
from ..registries import OAKEDatasetRegistry
from .datasets import Batch, BlockDatasetMixin


@OAKEDatasetRegistry.register_()
class V3DetBlockDataset(V3DetMixin, BlockDatasetMixin, V3DetDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)

    def _getitem(self, index: int) -> Batch:
        key, image = self._access(index)
        bboxes, blocks = self._partition(image)
        return Batch(id_=key.replace('/', '_'), bboxes=bboxes, blocks=blocks)
