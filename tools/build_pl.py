import json
import sys
from typing import Optional

import todd

sys.path.insert(0, '')
import mldec

logger = todd.base.get_logger()


def build_coco_from_pl(pl_data_file: str, coco_data_file: str, coco_split: str,pl_split: str, delete:Optional[bool]=False) -> None:
    logger.info(f'Loading {pl_data_file}')
    with open(pl_data_file) as f:
        pl_data = json.load(f)
    logger.info(f'Loaded {pl_data_file}')

    logger.info(f'Loading {coco_data_file}')
    with open(coco_data_file) as f:
        coco_data = json.load(f)
    logger.info(f'Loaded {coco_data_file}')

    coco_category_dict = {category['name']: category for category in coco_data['categories']}
    coco_categories = [coco_category_dict[class_] for class_ in getattr(mldec, coco_split)]

    coco_category_ids = [category['id'] for category in coco_categories]
    coco_annotations = [
        annotation for annotation in coco_data['annotations']
        if annotation['category_id'] in coco_category_ids
    ]
    for annotation in coco_annotations:
        annotation['category_id'] = coco_category_ids.index(annotation['category_id'])
    
    for i, category in enumerate(coco_categories):
        # import ipdb;ipdb.set_trace()
        category['id'] = i
    coco_category_ids = [category['id'] for category in coco_categories]

    pl_category_dict = {category['name']: category for category in pl_data['categories']}
    # import ipdb;ipdb.set_trace()
    for i,class_ in enumerate(getattr(mldec, pl_split)):
        cat = pl_category_dict[class_] 
        assert cat['id'] == i
        if class_ not in coco_category_dict.keys():
            cat['id'] = cat['id'] + 80
            coco_categories.append(cat)

    coco_category_ids = [category['id'] for category in coco_categories]
    # import ipdb;ipdb.set_trace()
    for anno in pl_data['annotations']:
        if anno['category_id'] + 80 not in coco_category_ids:
            if delete:
                pass
            else:
                anno['category_id'] = getattr(mldec, coco_split).index(getattr(mldec, pl_split)[anno['category_id']])
                anno['is_pl'] = True
                coco_annotations.append(anno)
        else:
            anno['category_id'] = anno['category_id']+80
            # anno['category_id'] = coco_category_ids.index(anno['category_id'])
            coco_annotations.append(anno)   

    # for i, category in enumerate(coco_categories):
    #     # import ipdb;ipdb.set_trace()
    #     category['id'] = i
    # import ipdb;ipdb.set_trace()
    coco_data['categories'] = coco_categories
    coco_data['annotations'] = coco_annotations
    
    save_path = f'{coco_data_file}.{pl_split}'
    logger.info(f'Saving {save_path}')
    import ipdb;ipdb.set_trace()
    with open(f'{save_path}', 'w') as f:
        json.dump(coco_data, f, separators=(',', ':'))
    logger.info(f'Saved {save_path}')


def build_lvis_from_pl(pl_data_file: str, lvis_data_file: str, lvis_split: str,pl_split: str) -> None:
    logger.info(f'Loading {pl_data_file}')
    with open(pl_data_file) as f:
        pl_data = json.load(f)
    logger.info(f'Loaded {pl_data_file}')

    logger.info(f'Loading {lvis_data_file}')
    with open(lvis_data_file) as f:
        lvis_data = json.load(f)
    logger.info(f'Loaded {lvis_data_file}')

    lvis_category_dict = {category['name']: category for category in lvis_data['categories']}
    lvis_categories = [lvis_category_dict[class_] for class_ in getattr(mldec, lvis_split)]

    lvis_category_ids = [category['id'] for category in lvis_categories]
    lvis_annotations = [
        annotation for annotation in lvis_data['annotations']
        if annotation['category_id'] in lvis_category_ids
    ]
    for annotation in lvis_annotations:
        annotation['category_id'] = lvis_category_ids.index(annotation['category_id'])
    
    image_ids = [image['id'] for image in lvis_data['images']]
    for anno in pl_data['annotations']:
        if anno['image_id'] in image_ids:
            lvis_annotations.append(anno)

    for i, category in enumerate(lvis_categories):
        category['id'] = i
    import ipdb;ipdb.set_trace()
    lvis_data['categories'] = lvis_categories
    lvis_data['annotations'] = lvis_annotations
    
    save_path = f'Saving {lvis_data_file}.{pl_split}'
    logger.info(f'Saving {save_path}')
    with open(f'{save_path}', 'w') as f:
        json.dump(lvis_data, f, separators=(',', ':'))
    logger.info(f'Saved {save_path}')




def main() -> None:
    build_coco_from_pl('data/lvis_pl.json','data/coco/annotations/instances_val2017.json','COCO_48_17','LVIS',False)
    pass


if __name__ == '__main__':
    main()
