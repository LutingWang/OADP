import json
import random
import os
from pycocotools.coco import COCO

from tqdm import tqdm


def sample_images_for_cls(num_samples, cls_name, coco: COCO, min_area=32 * 32):
    cls_id = coco.getCatIds(catNms=[cls_name])[0]['id']
    img_ids = coco.catToImgs[cls_id]
    random.shuffle(img_ids)

    images_path = []
    cnt = 0
    for img_id in tqdm(img_ids):
        img = coco.loadImgs(img_id)[0]
        # check the area of the bbox
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        max_area = 0
        for ann in anns:
            if ann['category_id'] == cls_id:
                max_area = max(max_area, ann['area'])
        if max_area < min_area:
            continue
        else:
            img_path = os.path.join('data/coco/train2017', img['file_name'])
            images_path.append(img_path)    
            cnt += 1
            if cnt >= num_samples:
                break
    return images_path


def sample_images_for_all(ann_file, num_samples):
    coco = COCO(ann_file)
    examples_dict = {}
    for cls in coco.loadCats(coco.getCatIds()):
        images_path = sample_images_for_cls(num_samples, cls['name'], coco)
        examples_dict[cls['name']] = images_path
    with open('examples.json', 'w') as f:
        json.dump(examples_dict, f)
        
if __name__ == '__main__':
    sample_images_for_all(ann_file='data/coco/annotations/instances_train2017.json', num_samples=10)
