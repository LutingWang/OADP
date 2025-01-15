# mypy: disable-error-code="misc"

__all__ = [
    'COCOObjectDataset',
]
import torch
from typing import TYPE_CHECKING

from todd.datasets import COCODataset
from todd.datasets.coco import Annotations
from todd.tasks.object_detection import BBox, FlattenBBoxesXYWH

from ..registries import OAKEDatasetRegistry
from .datasets import Batch, ObjectDataset
from ..globals_.coco import V3DetGlobalDataset

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


class BaseObjectDataset(V3DetGlobalDataset):
    def __init__(self, *args, auto_fix, split, **kwargs):
        self._min_wh = (16, 16)
        self.categories = []
        super().__init__(*args, auto_fix=auto_fix, split=split, **kwargs)


    def __getitem__(self, index: int) -> Batch | None:
        filename, image, annotations = self._getitem(index)
        if len(annotations) == 0:
            bboxes = FlattenBBoxesXYWH(torch.zeros(0, 4))
        else:
            bboxes = FlattenBBoxesXYWH(torch.tensor([ann['bbox'] for ann in annotations]))

        indices = bboxes.indices(min_wh=self._min_wh)
        if not indices.any():
            return None

        bboxes = bboxes[indices]
        crops, masks = self.runner.expand_transform(image, bboxes)
        key = filename.replace('.jpg', '').replace('/', '-')
        return Batch(
            id_=key,
            bboxes=bboxes,
            categories=[],
            crops=crops,
            masks=masks,
        )    

@OAKEDatasetRegistry.register_()
class V3DetObjectDataset(BaseObjectDataset):
    pass