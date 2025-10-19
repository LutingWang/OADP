import json
import os
from pycocotools.coco import COCO

lvis_path = 'data/grounding_data/coco/annotations/lvis_v1_val.json'

print(f"Loading annotations from {lvis_path}")
data = json.load(open(lvis_path))

images = data['images']
annotations = data['annotations']

num_images = len(images)
mid_point = num_images // 2
print(f"Splitting at image {mid_point}")
# breakpoint()
images1 = images[:mid_point]
images2 = images[mid_point:]

image_ids1 = {img['id'] for img in images1}
image_ids2 = {img['id'] for img in images2}

print("Splitting annotations...")
annotations1 = [ann for ann in annotations if ann['image_id'] in image_ids1]
annotations2 = [ann for ann in annotations if ann['image_id'] in image_ids2]

data1 = {
    'info': data.get('info', {}),
    'licenses': data.get('licenses', []),
    'categories': data['categories'],
    'images': images1,
    'annotations': annotations1
}
# LVIS-specific keys
if 'neg_category_ids' in data:
    data1['neg_category_ids'] = data['neg_category_ids']
if 'not_exhaustive_category_ids' in data:
    data1['not_exhaustive_category_ids'] = data['not_exhaustive_category_ids']


data2 = {
    'info': data.get('info', {}),
    'licenses': data.get('licenses', []),
    'categories': data['categories'],
    'images': images2,
    'annotations': annotations2
}
if 'neg_category_ids' in data:
    data2['neg_category_ids'] = data['neg_category_ids']
if 'not_exhaustive_category_ids' in data:
    data2['not_exhaustive_category_ids'] = data['not_exhaustive_category_ids']


output_dir = os.path.dirname(lvis_path)
output_path1 = os.path.join(output_dir, 'lvis_v1_val_mini_1.json')
output_path2 = os.path.join(output_dir, 'lvis_v1_val_mini_2.json')

print(f"Saving split 1 to {output_path1}")
with open(output_path1, 'w') as f:
    json.dump(data1, f)

print(f"Saving split 2 to {output_path2}")
with open(output_path2, 'w') as f:
    json.dump(data2, f)

print("\nSplitting complete.")
print(f"  - {output_path1}: {len(images1)} images, {len(annotations1)} annotations")
print(f"  - {output_path2}: {len(images2)} images, {len(annotations2)} annotations")









