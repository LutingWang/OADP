# mypy: disable-error-code="misc"

__all__ = [
    'V3DetGlobalDataset',
]

from todd.datasets import V3DetDataset

from ..utils import V3DetMixin
from ..registries import OAKEDatasetRegistry
from .datasets import Batch, GlobalDataset


@OAKEDatasetRegistry.register_()
class V3DetGlobalDataset(V3DetMixin, GlobalDataset, V3DetDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)

    def _getitem(self, index: int) -> Batch | None:
        item = super()._getitem(index)
        item['id_'] = item['id_'].replace('/', '_')
        return item
