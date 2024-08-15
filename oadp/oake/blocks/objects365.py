from ..globals_.objects365 import Objects365V2Dataset
from ..registries import OAKEDatasetRegistry
from .datasets import Batch, BlockDatasetMixin


@OAKEDatasetRegistry.register_()
class Objects365V2BlockDataset(BlockDatasetMixin, Objects365V2Dataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)

    def _getitem(self, index: int) -> Batch:
        key, image = self._access(index)
        bboxes, blocks = self._partition(image)
        return Batch(id_=key.replace('/', '_'), bboxes=bboxes, blocks=blocks)
