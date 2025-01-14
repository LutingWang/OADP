__all__ = [
    'V3DetMixin',
]

from todd.datasets import V3DetDataset
from todd.datasets.access_layers import PILAccessLayer
from typing import cast
from ..datasets import BaseDataset


class V3DetMixin(BaseDataset, V3DetDataset):

    def exists(self, index: int) -> bool:
        if super().exists(index):
            return True
        # If the image doesn't exist, jump to the next one by pretending the
        # embeddings exist
        access_layer = cast(PILAccessLayer, self._access_layer)
        key = self._keys[index]
        return not access_layer._file(key).exists()
