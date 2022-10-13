import json
import sys
from typing import Optional

import todd

sys.path.insert(0, '')
import mldec

logger = todd.base.get_logger()


def filter_(data_file: str) -> None:
    logger.info(f'Loading {data_file}')
    with open(data_file) as f:
        data = json.load(f)
    logger.info(f'Loaded {data_file}')

    image_ids = {
        annotation['image_id']
        for annotation in data['annotations']
    }
    data['annotations'] = [
        annotation
        for annotation in data['annotations']
        if annotation['image_id'] in image_ids
    ]
    data['images'] = [
        image
        for image in data['images']
        if image['id'] in image_ids
    ]

    logger.info(f'Saving {data_file}.filtered')
    with open(f'{data_file}.filtered', 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    logger.info(f'Saved {data_file}.filtered')


def main() -> None:
    filter_('data/coco/annotations/instances_val2017.json.COCO_48_17')


if __name__ == '__main__':
    main()
