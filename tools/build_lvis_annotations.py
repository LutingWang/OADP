import json
import sys
from typing import Optional

import todd

sys.path.insert(0, '')
import mldec

logger = todd.base.get_logger()


def build(data_file: str, split: str, num_base_classes: Optional[int] = None) -> None:
    logger.info(f'Loading {data_file}')
    with open(data_file) as f:
        data = json.load(f)
    logger.info(f'Loaded {data_file}')

    category_dict = {category['name']: category for category in data['categories']}
    categories = [category_dict[class_] for class_ in getattr(mldec, split)]

    category_ids = [category['id'] for category in categories]
    for image in data['images']:
        image['neg_category_ids'] = [
            category_ids.index(i) for i in image['neg_category_ids']
        ]
        image['not_exhaustive_category_ids'] = [
            category_ids.index(i) for i in image['not_exhaustive_category_ids']
        ]
    annotations = [
        annotation for annotation in data['annotations']
        if annotation['category_id'] in category_ids
    ]
    for annotation in annotations:
        annotation['category_id'] = category_ids.index(annotation['category_id'])
    data['annotations'] = annotations

    for i, category in enumerate(categories):
        category['id'] = i
    data['categories'] = categories

    logger.info(f'Saving {data_file}.{split}')
    with open(f'{data_file}.{split}', 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    logger.info(f'Saved {data_file}.{split}')

    if num_base_classes is None:
        return

    data['annotations'] = [
        annotation for annotation in data['annotations']
        if annotation['category_id'] < num_base_classes
    ]

    logger.info(f'Saving {data_file}.{split}.{num_base_classes}')
    with open(f'{data_file}.{split}.{num_base_classes}', 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    logger.info(f'Saved {data_file}.{split}.{num_base_classes}')


def main() -> None:
    build('data/lvis_v1/annotations/lvis_v1_train.json', 'LVIS', 866)
    build('data/lvis_v1/annotations/lvis_v1_val.json', 'LVIS', 866)


if __name__ == '__main__':
    main()
