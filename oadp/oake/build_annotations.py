import json
from typing import Any

import todd

from ..base import Globals, coco

Data = dict[str, Any]


class Builder:

    def __init__(self, name: str) -> None:
        categories = Globals.categories
        name = f'{name}_{categories.num_bases}_{categories.num_novels}'
        self._name = name

    def __call__(self, file: str, filter_: bool) -> None:
        data = self._load(file)
        file = f'{file}.{self._name}'

        category_ids = self._build_categories(data)
        base_annotations = self._build_annotations(data, category_ids)
        self._dump(data, file)

        with todd.set_temp(data, '["annotations"]', base_annotations):
            self._dump(data, f'{file}.{Globals.categories.num_bases}')

        if filter_:
            self._filter_images(data)
            self._dump(data, f'{file}.filtered')

    def _load(self, file: str) -> Data:
        Globals.logger.info(f'Loading {file}')
        with open(file) as f:
            data = json.load(f)
        Globals.logger.info(f'Loaded {file}')
        return data

    def _dump(self, data: Data, file: str) -> None:
        Globals.logger.info(f'Saving {file}')
        with open(file, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        Globals.logger.info(f'Saved {file}')

    def _build_categories(self, data: Data) -> list[int]:
        dict_ = {category['name']: category for category in data['categories']}
        categories = [dict_[category] for category in Globals.categories.all_]
        ids_ = [category['id'] for category in categories]
        for i, category in enumerate(categories):
            category['id'] = i
        data['categories'] = categories
        return ids_

    def _build_annotations(
        self,
        data: Data,
        category_ids: list[int],
    ) -> list[dict[str, Any]]:
        annotations = [
            annotation for annotation in data['annotations']
            if annotation['category_id'] in category_ids
        ]
        for annotation in annotations:
            annotation['category_id'] = category_ids.index(
                annotation['category_id'],
            )
        data['annotations'] = annotations
        return [
            annotation for annotation in annotations
            if annotation['category_id'] < Globals.categories.num_bases
        ]

    def _filter_images(self, data: Data) -> None:
        image_ids = {
            annotation['image_id']
            for annotation in data['annotations']
        }
        data['images'] = [
            image for image in data['images'] if image['id'] in image_ids
        ]


class LVISBuilder(Builder):

    def _build_categories(self, data: Data) -> list[int]:
        ids = super()._build_categories(data)
        for image in data['images']:
            image['neg_category_ids'] = [
                ids.index(i) for i in image['neg_category_ids']
            ]
            image['not_exhaustive_category_ids'] = [
                ids.index(i) for i in image['not_exhaustive_category_ids']
            ]
        return ids


def main() -> None:
    Globals.categories = coco
    builder = Builder('COCO')
    builder('data/coco/annotations/instances_train2017.json', False)
    builder('data/coco/annotations/instances_val2017.json', True)


if __name__ == '__main__':
    main()
