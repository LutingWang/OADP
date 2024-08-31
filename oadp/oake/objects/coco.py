# mypy: disable-error-code="misc"

__all__ = [
    'COCOObjectDataset',
]

from typing import TYPE_CHECKING
from todd.datasets import COCODataset
from todd.datasets.coco import Annotations

from .datasets import ObjectDataset, Batch
from ..registries import OAKEDatasetRegistry

if TYPE_CHECKING:
    from pycocotools.coco import _Category


@OAKEDatasetRegistry.register_()
class COCOObjectDataset(ObjectDataset, COCODataset):

    @property
    def categories(self) -> list['_Category']:
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
        indices = bboxes.indices(min_wh=self._min_wh)
        if not indices.any():
            return None
        bboxes = bboxes[indices]
        categories = annotations.categories[indices]
        crops, masks = self.runner.expand_transform(image, bboxes)
        return Batch(
            id_=key,
            bboxes=bboxes,
            categories=categories,
            crops=crops,
            masks=masks,
        )
