import json
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import todd
from lvis import LVIS
from mmdet.datasets.api_wrappers import COCO

from .base import Categories, coco, lvis

Data = dict[str, Any]


class Builder(ABC):

    def __init__(self, categories: Categories, root: str) -> None:
        self._categories = categories
        self._root = pathlib.Path(root)

    @abstractmethod
    def _load(self, file: pathlib.Path) -> Data:
        pass

    def _map_cat_ids(self, data: Data, cat_oid2nid: dict[int, int]) -> None:
        for cat in data['categories']:
            cat['id'] = cat_oid2nid[cat['id']]
        for ann in data['annotations']:
            ann['category_id'] = cat_oid2nid[ann['category_id']]

    def _dump(self, data: Data, file: pathlib.Path, suffix: str) -> None:
        file = file.with_stem(f'{file.stem}.{suffix}')
        todd.logger.info(f'Dumping {file}')
        with file.open('w') as f:
            json.dump(data, f, separators=(',', ':'))
        todd.logger.info(f'Dumped {file}')

    def _filter_anns(self, data: Data) -> Data:
        anns = [
            ann for ann in data['annotations']
            if ann['category_id'] < self._categories.num_bases
        ]
        return data | dict(annotations=anns)

    def _filter_imgs(self, data: Data) -> Data:
        img_ids = {ann['image_id'] for ann in data['annotations']}
        imgs = [img for img in data['images'] if img['id'] in img_ids]
        return data | dict(images=imgs)

    def build(self, filename: str, min: bool) -> None:
        file = self._root / filename
        data = self._load(file)

        cat_oid2nid = {  # nid = new id, oid = old id
            cat['id']: self._categories.all_.index(cat['name'])
            for cat in data['categories']
        }
        self._map_cat_ids(data, cat_oid2nid)
        data['categories'] = sorted(
            data['categories'], key=lambda cat: cat['id']
        )

        self._dump(data, file, str(self._categories.num_all))
        filtered_data = self._filter_anns(data)
        self._dump(filtered_data, file, str(self._categories.num_bases))
        if min:
            filtered_data = self._filter_imgs(data)
            self._dump(filtered_data, file, f'{self._categories.num_all}.min')


class COCOBuilder(Builder):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(coco, *args, **kwargs)

    def _load(self, file: pathlib.Path) -> Data:
        data = COCO(file)
        cat_ids = data.get_cat_ids(cat_names=self._categories.all_)
        ann_ids = data.get_ann_ids(cat_ids=cat_ids)
        img_ids = data.get_img_ids()
        cats = data.load_cats(cat_ids)
        anns = data.load_anns(ann_ids)
        imgs = data.load_imgs(img_ids)
        return dict(categories=cats, annotations=anns, images=imgs)


class LVISBuilder(Builder):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(lvis, *args, **kwargs)

    def _load(self, file: pathlib.Path) -> Data:
        data = LVIS(file)
        anns = data.load_anns()
        cats = data.load_cats(None)
        imgs = data.load_imgs(None)
        return dict(categories=cats, annotations=anns, images=imgs)

    def _map_cat_ids(self, data: Data, cat_oid2nid: dict[int, int]) -> None:
        super()._map_cat_ids(data, cat_oid2nid)
        for img in data['images']:
            img['neg_category_ids'] = [
                cat_oid2nid[cat_id] for cat_id in img['neg_category_ids']
            ]
            img['not_exhaustive_category_ids'] = [
                cat_oid2nid[cat_id]
                for cat_id in img['not_exhaustive_category_ids']
            ]


def main() -> None:
    coco_builder = COCOBuilder('data/coco/annotations')
    coco_builder.build('instances_val2017.json', True)
    coco_builder.build('instances_train2017.json', False)

    lvis_builder = LVISBuilder('data/lvis_v1/annotations')
    lvis_builder.build('lvis_v1_val.json', False)
    lvis_builder.build('lvis_v1_train.json', False)


if __name__ == '__main__':
    main()
