import json
from mmdet.datasets import Objects365V1Dataset, LVISDataset, CocoDataset

def clean_name(name):
    if "/" in name:
        name = name.split("/")[0]
    if "_" in name:
        name = name.replace("_", " ")
    if "-" in name:
        name = name.replace("-", " ")
    # remove str in brackets
    if "(" in name:
        name = name.split("(")[0]
    if "[" in name:
        name = name.split("[")[0]
    return name.lower().strip()

def extract_label_map(dataset_name):
    if dataset_name == "objects365":
        dataset = Objects365V1Dataset
        meta_info = dataset.METAINFO
        class_name = meta_info['classes']
        label_map = {}
        for i, name in enumerate(class_name):
            label_map[str(i)] = name

    if dataset_name == "coco":
        dataset = CocoDataset
        meta_info = dataset.METAINFO
        class_name = meta_info['classes']
        label_map = {}
        for i, name in enumerate(class_name):
            label_map[str(i)] = name

    if dataset_name == "lvis":
        mmovod_fs = json.load(open("data/mmovod-samples/lvis_image_exemplar_dict_K-005_author.json", "r"))
        label_map_new = {}
        for i, exps in enumerate(mmovod_fs):
            for exp in exps:
                if exp['dataset'] == "imagenet21k" or exp['dataset'] == "visual_genome":
                    label_map_new[i] = exp['file_name']
        print(f"len: {len(mmovod_fs)}")
        print(f"len: {len(label_map_new)}")
        return label_map_new

    if dataset_name == "V3Det":
        json_path = "data/V3Det/annotations/v3det_2023_v1_label_map.json"
        label_map = json.load(open(json_path))

    label_map_new = {}
    for k, v in label_map.items():
        label_map_new[k] = clean_name(v)
    return label_map_new


def gen_imagent_fs_datasets(imagenet_label_map:str, datasets='objects365'):
    labels_map =  extract_label_map(datasets)
    # clean the imagenet label map
    # imagenet_label_map_clean = {}
    imagenet_label2wnid = {}
    with open(imagenet_label_map, "r") as f:
        imagenet_label_map = json.load(f)
        for k, v in imagenet_label_map.items():
            # imagenet_label_map_clean[k] = clean_name(v)
            imagenet_label2wnid[clean_name(v)] = k

    datasets2imagenet = {}
    for i, label in labels_map.items():
        if label in imagenet_label2wnid.keys():
            if label == "submachine gun":
                print("debug")
            datasets2imagenet[i] = {
                "name": label,
                "samples_id": imagenet_label2wnid[label]
            }
            print(f"{label} -> {imagenet_label2wnid[label]}")
    
    print(f"overall len: {len(datasets2imagenet)}")

    with open(f"data/{datasets}/annotations/{datasets.lower()}_imagenet_label_map.json", "w") as f:
        json.dump(datasets2imagenet, f, indent=4)

def coco_in_imagenet():
    coco_annotations = json.load(open("data/coco/annotations/instances_val2017.json", "r")) # 
    coco_imagenet_map = json.load(open("data/coco/annotations/coco_imagenet_label_map.json", "r")) # coco id -> name
    
    categort_map = {}
    for cat in coco_annotations['categories']:
        categort_map[cat['id']] = cat['name']
    coco_imagenet_categories = [v['name'] for v in coco_imagenet_map.values()]
    coco_imagenet_categories = set(coco_imagenet_categories)

    # Select categories from coco_annotations that are in coco_imagenet_categories
    selected_categories = [cat for cat in coco_annotations['categories'] if cat['name'] in coco_imagenet_categories]
    
    # Get the set of selected category IDs
    selected_category_ids = set(cat['id'] for cat in selected_categories)
    
    # Filter annotations that belong to the selected categories
    selected_annotations = [ann for ann in coco_annotations['annotations'] if ann['category_id'] in selected_category_ids]
    
    # Directly update the original coco_annotations keys
    coco_annotations['categories'] = selected_categories
    coco_annotations['annotations'] = selected_annotations
    
    # Overwrite the file with the modified data using the new file name
    with open("data/coco/annotations/instances_val2017_imagenet.json", "w") as f:
        json.dump(coco_annotations, f, indent=4)
    
    coco_imagenet_map_new = {}
    for idx, (_, value) in enumerate(coco_imagenet_map.items()):
        coco_imagenet_map_new[str(idx)] = value
    with open("data/coco/annotations/coco_imagenet_label_map_new.json", "w") as f:
        json.dump(coco_imagenet_map_new, f, indent=4)

    print(f"Updated coco_annotations: {len(selected_categories)} categories and {len(selected_annotations)} annotations.")

if __name__ == '__main__':
    label_map_path = "data/imagenet21k/annotations/imagenet21k_label_map.json"
    # gen_imagent_fs_datasets(label_map_path, 'V3Det')
    # gen_imagent_fs_datasets(label_map_path, 'coco')
    # gen_imagent_fs_datasets(label_map_path, 'lvis')
    coco_in_imagenet()