import json
import pathlib
import sys
from abc import ABC, abstractmethod
from typing import Any

import todd
from lvis import LVIS

sys.path.insert(0, '')
from oadp.categories import Categories, coco, lvis  # noqa: E402 E501 isort:skip pylint: disable=wrong-import-position
from mmdet.datasets.api_wrappers import COCO  # noqa: E402 E501 isort:skip pylint: disable=wrong-import-position,wrong-import-order

Data = dict[str, Any]


class Builder(ABC):
    CATEGORIES: Categories

    def __init__(self, root: str) -> None:
        self._root = pathlib.Path(root)

    @abstractmethod
    def _load(self, file: pathlib.Path) -> Data:
        pass

    def _map_category_ids(self, data: Data, oid2nid: dict[int, int]) -> None:
        for cat in data['categories']:
            cat['id'] = oid2nid[cat['id']]
        for ann in data['annotations']:
            ann['category_id'] = oid2nid[ann['category_id']]

    def _dump(self, data: Data, file: pathlib.Path, suffix: str) -> None:
        file = file.with_stem(f'{file.stem}.{suffix}')
        todd.logger.info("Dumping %s", file)
        with file.open('w') as f:
            json.dump(data, f, separators=(',', ':'))
        todd.logger.info("Dumped %s", file)

    def _filter_annotations(self, data: Data) -> Data:
        annotations = [
            annotation for annotation in data['annotations']
            if annotation['category_id'] < self.CATEGORIES.num_bases
        ]
        return data | dict(annotations=annotations)

    def _filter_images(self, data: Data) -> Data:
        image_ids = {
            annotation['image_id']
            for annotation in data['annotations']
        }
        images = [
            image for image in data['images'] if image['id'] in image_ids
        ]
        return data | dict(images=images)

    def build(self, filename: str, min_: bool) -> None:
        file = self._root / filename
        data = self._load(file)

        category_oid2nid = {  # nid = new id, oid = old id
            category['id']: self.CATEGORIES.all_.index(category['name'])
            for category in data['categories']
        }
        self._map_category_ids(data, category_oid2nid)
        data['categories'] = sorted(
            data['categories'],
            key=lambda category: category['id'],
        )

        self._dump(data, file, str(self.CATEGORIES.num_all))
        filtered_data = self._filter_annotations(data)
        self._dump(filtered_data, file, str(self.CATEGORIES.num_bases))
        if min_:
            filtered_data = self._filter_images(data)
            self._dump(filtered_data, file, f'{self.CATEGORIES.num_all}.min')


class COCOBuilder(Builder):
    CATEGORIES = coco

    def _load(self, file: pathlib.Path) -> Data:
        data = COCO(file)
        category_ids = data.get_cat_ids(cat_names=self.CATEGORIES.all_)
        annotation_ids = data.get_ann_ids(cat_ids=category_ids)
        image_ids = data.get_img_ids()
        categories = data.load_cats(category_ids)
        annotations = data.load_anns(annotation_ids)
        images = data.load_imgs(image_ids)
        return dict(
            categories=categories,
            annotations=annotations,
            images=images,
        )


class LVISBuilder(Builder):
    CATEGORIES = lvis

    def _load(self, file: pathlib.Path) -> Data:
        data = LVIS(file)
        annotations = data.load_anns()
        categories = data.load_cats(None)
        images = data.load_imgs(None)
        return dict(
            categories=categories,
            annotations=annotations,
            images=images,
        )

    def _map_category_ids(self, data: Data, oid2nid: dict[int, int]) -> None:
        super()._map_category_ids(data, oid2nid)
        for images in data['images']:
            images['neg_category_ids'] = [
                oid2nid[category_id]
                for category_id in images['neg_category_ids']
            ]
            images['not_exhaustive_category_ids'] = [
                oid2nid[category_id]
                for category_id in images['not_exhaustive_category_ids']
            ]


def main() -> None:
    coco_builder = COCOBuilder('data/coco/annotations')
    coco_builder.build('instances_val2017.json', True)
    coco_builder.build('instances_train2017.json', False)

    lvis_builder = LVISBuilder('data/lvis/annotations')
    lvis_builder.build('lvis_v1_val.json', False)
    lvis_builder.build('lvis_v1_minival.json', False)
    lvis_builder.build('lvis_v1_train.json', False)


if __name__ == '__main__':
    main()
