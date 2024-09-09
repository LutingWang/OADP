# mypy: disable-error-code="misc"

__all__ = [
    'LVISObjectDataset',
]

from typing import Any

from todd.datasets import LVISDataset
from todd.datasets.lvis import Annotations

from ..registries import OAKEDatasetRegistry
from .datasets import Batch, ObjectDataset


@OAKEDatasetRegistry.register_()
class LVISObjectDataset(ObjectDataset, LVISDataset):

    @property
    def categories(self) -> list[dict[str, Any]]:
        category_ids = self.api.get_cat_ids()
        categories = self.api.load_cats(category_ids)
        return categories

    def _getitem(self, index: int) -> Batch | None:
        key, image = self._access(index)
        annotations = Annotations.load(
            self._api,
            self._keys.image_ids[index],
            self._categories,
        )
        bboxes = annotations.bboxes
        indices = bboxes.indices(min_wh=self._min_wh)
        if not indices.any():
            return None
        bboxes = bboxes[indices]
        categories = annotations.categories[indices]
        crops, masks = self.runner.expand_transform(image, bboxes)
        return Batch(
            id_=key.replace('/', '_'),
            bboxes=bboxes,
            categories=categories,
            crops=crops,
            masks=masks,
        )