import pathlib
import sys
import cv2

import numpy as np
import todd
import torchvision
from tqdm import trange

sys.path.insert(0, '')
import mldec


dataset = torchvision.datasets.CocoDetection(
    root='data/coco/val2017/',
    annFile='data/coco/annotations/instances_val2017.json',
)
out = pathlib.Path('work_dirs/visual/coco_48_17')
out.mkdir(parents=True, exist_ok=True)
for i in trange(len(dataset)):
    image, target = dataset[i]
    cv_image = np.array(image)[:, :, ::-1].copy()
    bboxes = todd.BBoxesXYXY(todd.BBoxesXYWH([_['bbox'] for _ in target]))
    texts = [dataset.coco.cats[_['category_id']]['name'] for _ in target]
    colors = [(255, 0, 0) if text in mldec.COCO_48 else (0, 255, 0) for text in texts]
    todd.visuals.draw_annotations(
        cv_image,
        bboxes,
        colors,
        texts,
    )
    assert cv2.imwrite(f'{out}/{dataset.ids[i]:012d}.png', cv_image)
