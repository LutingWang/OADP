# mypy: disable-error-code=misc

__all__ = [
    'Objects365GlobalDataset',
]

from todd.datasets import Objects365Dataset

from ..registries import OAKEDatasetRegistry
from .datasets import GlobalDataset, Batch


@OAKEDatasetRegistry.register_()
class Objects365GlobalDataset(GlobalDataset, Objects365Dataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)

    def _getitem(self, index: int) -> Batch | None:
        item = super()._getitem(index)
        item['id_'] = item['id_'].replace('/', '_')
        return item
