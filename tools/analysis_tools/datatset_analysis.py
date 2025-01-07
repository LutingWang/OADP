import json
import os
from prettytable import PrettyTable

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def count_coco_data(json_data):
    num_images = len(json_data.get('images', []))
    num_annotations = len(json_data.get('annotations', []))
    num_categories = len(json_data.get('categories', []))
    return num_images, num_annotations, num_categories

def main(annotation_files):
    table = PrettyTable()
    table.field_names = ["File", "Number of Images", "Number of Annotations", "Number of Categories"]

    for file_path in annotation_files:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            continue

        json_data = load_json_file(file_path)
        num_images, num_annotations, num_categories = count_coco_data(json_data)
        
        table.add_row([os.path.basename(file_path), num_images, num_annotations, num_categories])

    print(table)

if __name__ == "__main__":
    # Example usage
    annotation_files = [
        # 'data/gqa/final_mixed_train_no_coco.json',
        # 'data/flickr30k_entities/final_flickr_separateGT_train.json',
        # 'data/objects365v2/annotations/zhiyuan_objv2_train_fixname.json',
        # 'data/V3Det/annotations/v3det_2023_v1_train.json',
        'data/OpenDataLab___Objects365_v1/raw/Objects365_v1/2019-08-02/objects365_train.json'
    ]
    main(annotation_files)
