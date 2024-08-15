__all__ = ['Objects365GlobalDataset']
import os.path as osp
import pathlib
from collections import UserList
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, TypedDict, cast
from typing_extensions import Self

import torch
from todd.datasets.coco import COCO, BaseDataset, BaseKeys, PILAccessLayer, T

from ..registries import OAKEDatasetRegistry
from .datasets import GlobalDataset

if TYPE_CHECKING:
    from todd.tasks.object_detection import BBox, FlattenBBoxesXYWH

Split = Literal['train', 'val']
objv2_ignore_list = [
    osp.join('patch16', 'objects365_v2_00908726.jpg'),
    osp.join('patch6', 'objects365_v1_00320532.jpg'),
    osp.join('patch6', 'objects365_v1_00320534.jpg'),
]


class Keys(BaseKeys):

    def __init__(self, coco: COCO, *args, **kwargs) -> None:
        self._coco = coco
        image_ids = self._coco.getImgIds()
        self.id2file = {}
        for image_id in image_ids:
            image = self._coco.loadImgs(image_id)[0]
            file_name = osp.join(
                osp.split(osp.split(image['file_name'])[0])[-1],
                osp.split(image['file_name'])[-1]
            )
            if file_name in objv2_ignore_list:
                continue
            self.id2file[image_id] = file_name
        super().__init__(image_ids, *args, **kwargs)

    def _getitem(self, image_id: int) -> str:
        return self.id2file[image_id]


class _Annotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    area: float
    bbox: list[float]
    iscrowd: int


@dataclass(frozen=True)
class Annotation:
    area: float
    is_crowd: bool
    bbox: 'BBox'
    category: int

    @classmethod
    def load(
        cls,
        annotation: '_Annotation',
        categories: Mapping[int, int],
    ) -> Self:
        return cls(
            annotation['area'],
            bool(annotation['iscrowd']),
            cast('BBox', annotation['bbox']),
            categories[annotation['category_id']],
        )


class Annotations(UserList[Annotation]):

    @classmethod
    def load(
        cls,
        coco: COCO,
        image_id: int,
        categories: Mapping[int, int],
    ) -> Self:
        annotation_ids = coco.getAnnIds([image_id])
        annotations = coco.loadAnns(annotation_ids)
        return cls(
            Annotation.load(annotation, categories)
            for annotation in annotations
        )

    @property
    def areas(self) -> torch.Tensor:
        return torch.tensor([annotation.area for annotation in self])

    @property
    def is_crowd(self) -> torch.Tensor:
        return torch.tensor([annotation.is_crowd for annotation in self])

    @property
    def bboxes(self) -> 'FlattenBBoxesXYWH':
        from todd.tasks.object_detection import FlattenBBoxesXYWH
        if len(self) > 0:
            bboxes = torch.tensor([annotation.bbox for annotation in self])
        else:
            bboxes = torch.zeros(0, 4)
        return FlattenBBoxesXYWH(bboxes)

    @property
    def categories(self) -> torch.Tensor:
        return torch.tensor([annotation.category for annotation in self])


class Objects365V2Dataset(BaseDataset[COCO, T]):
    _keys: Keys

    DATA_ROOT = pathlib.Path('data/objects365v2')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'

    def __init__(
        self,
        *args,
        split: Split,
        version: Literal[1, 2] = 2,
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:

        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                task_name=split,
                suffix=self.SUFFIX,
            )
        if annotations_file is None:
            annotations_file = (
                self.ANNOTATIONS_ROOT / f'zhiyuan_objv2_{split}_existed.json'
            )

        coco = COCO(annotations_file)

        self._categories = {
            category_id: i
            for i, category_id in enumerate(coco.getCatIds())
        }

        super().__init__(*args, api=coco, access_layer=access_layer, **kwargs)

    def build_keys(self) -> Keys:
        return Keys(self._api, self.SUFFIX)

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = self._transform(image)
        annotations = (
            Annotations.load(
                self._api,
                self._keys.image_ids[index],
                self._categories,
            ) if self._load_annotations else Annotations()
        )
        return T(
            id_=key.replace('/', '_'), image=tensor, annotations=annotations
        )


@OAKEDatasetRegistry.register_()
class Objects365GlobalDataset(GlobalDataset, Objects365V2Dataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)
