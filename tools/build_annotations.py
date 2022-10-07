import json
import sys

import todd

sys.path.insert(0, '')
import mldec

logger = todd.base.get_logger()


def build(data_file: str, split: str) -> None:
    logger.info(f'Loading {data_file}')
    with open(data_file) as f:
        data = json.load(f)
    logger.info(f'Loaded {data_file}')

    category_dict = {category['name']: category for category in data['categories']}
    categories = [category_dict[class_] for class_ in getattr(mldec, split)]

    category_ids = [category['id'] for category in categories]
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

    logger.info(f'Saving {data_file}')
    with open(f'{data_file}.{split}', 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    logger.info(f'Saved {data_file}')


def main() -> None:
    build('data/coco/annotations/instances_train2017.json', 'COCO_48_17')
    build('data/coco/annotations/instances_val2017.json', 'COCO_48_17')


if __name__ == '__main__':
    main()
