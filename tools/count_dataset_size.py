import os
import json

coco_train_ann_file = "data/grounding_data/coco/annotations/lvis_v1_train_od.json"
gqa_train_ann_file = "data/grounding_data/gqa/gqa_train_vg7.jsonl"
v3det_train_ann_file = "data/grounding_data/v3det/annotations/v3det_2023_v1_train_od.json"
o365_train_ann_file = "data/grounding_data/objects365/annotations/objects365_train.fs_od.json"

def count_dataset_size(ann_file):
    with open(ann_file, 'r') as f:
        data = [json.loads(line) for line in f]
        return len(data)

print(f"coco: {count_dataset_size(coco_train_ann_file)}")
print(f"gqa: {count_dataset_size(gqa_train_ann_file)}")
print(f"v3det: {count_dataset_size(v3det_train_ann_file)}")
print(f"o365: {count_dataset_size(o365_train_ann_file)}")
