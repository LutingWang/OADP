__all__ = [
    'BaseLoader',
    'COCOLoader',
]

import random
from typing import TYPE_CHECKING, NamedTuple

import todd.tasks.object_detection as od
import torch
from PIL import Image
from pycocotools.coco import COCO
from todd.datasets import COCODataset

from ..errors import CategoryNotSupportedError
from ..loaders import BaseLoader, OADPCategoryLoaderRegistry

if TYPE_CHECKING:
    from pycocotools.coco import _Annotation


class T(NamedTuple):
    image: Image.Image
    bboxes: od.FlattenBBoxesMixin


class VisualLoader(BaseLoader[T]):
    pass


@OADPCategoryLoaderRegistry.register_()
class COCOLoader(VisualLoader):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._coco = COCODataset(split='train')

    def _load_category_id(self, coco: COCO, category_name: str) -> int:
        category_ids = coco.getCatIds(catNms=[category_name])
        if len(category_ids) == 0:
            raise CategoryNotSupportedError(category_name)
        category_id, = category_ids
        return category_id

    def _load_annotations(
        self,
        coco: COCO,
        category_name: str,
        category_id: int,
    ) -> list['_Annotation']:
        annotation_ids = coco.getAnnIds(catIds=[category_id])
        if len(annotation_ids) == 0:
            raise CategoryNotSupportedError(category_name)
        return coco.loadAnns(annotation_ids)

    def _sample_annotation(
        self,
        annotations: list['_Annotation'],
    ) -> '_Annotation':
        while True:
            index = random.randint(0, len(annotations) - 1)
            annotation = annotations[index]
            _, _, w, h = annotation['bbox']
            if w >= 32 and h >= 32:
                return annotation

    def _load_image(
        self,
        coco: COCO,
        annotation: '_Annotation',
    ) -> Image.Image:
        image_id = annotation['image_id']
        image, = coco.loadImgs(image_id)
        key = image['file_name'].removesuffix(f'.{self._coco.SUFFIX}')
        pil_image = self._coco.access_layer[key]
        pil_image = pil_image.convert('RGB')
        return pil_image

    def __call__(self, category_name: str) -> T:
        coco = self._coco.coco
        category_id = self._load_category_id(coco, category_name)
        annotations = self._load_annotations(coco, category_name, category_id)
        annotation = self._sample_annotation(annotations)
        image = self._load_image(coco, annotation)
        bboxes = od.FlattenBBoxesXYWH(torch.tensor([annotation['bbox']]))
        return T(image, bboxes)
