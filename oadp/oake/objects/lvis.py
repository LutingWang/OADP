__all__ = [
    'LVISObjectDataset',
]

from typing import Any
from todd.datasets import LVISDataset
from todd.datasets.lvis import Annotations

from .datasets import ObjectDataset, Batch
from ..registries import OAKEDatasetRegistry


@OAKEDatasetRegistry.register_()
class LVISObjectDataset(ObjectDataset, LVISDataset):

    @property
    def categories(self) -> list[dict[str, Any]]:
        category_ids = self.api.get_cat_ids()
        categories = self.api.load_cats(category_ids)
        return categories

    def _getitem(self, index: int) -> Batch:
        key, image = self._access(index)
        annotations = Annotations.load(
            self._api,
            self._keys.image_ids[index],
            self._categories,
        )
        bboxes = annotations.bboxes
        categories = annotations.categories
        crops, masks = self.runner.expand_transform(image, bboxes)
        return Batch(
            id_=key.replace('/', '_'),
            bboxes=bboxes,
            categories=categories,
            crops=crops,
            masks=masks,
        )
