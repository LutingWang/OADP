# mypy: disable-error-code="misc"

__all__ = [
    'Objects365ObjectDataset',
]

from typing import Any

from todd.datasets import Objects365Dataset
from todd.datasets.objects365 import Annotations
from ..registries import OAKEDatasetRegistry
from .datasets import Batch, ObjectDataset


@OAKEDatasetRegistry.register_()
class Objects365ObjectDataset(ObjectDataset, Objects365Dataset):

    @property
    def categories(self) -> list[dict[str, Any]]:
        category_ids = self.api.getCatIds()
        categories = self.api.loadCats(category_ids)
        return categories

    def _getitem(self, index: int) -> Batch | None:
        key, image = self._access(index)
        annotations = Annotations.load(
            self._api,
            self._keys.image_ids[index],
            self._categories,
        )
        bboxes = annotations.bboxes
        indices = bboxes.indices(min_wh=(4, 4))
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
