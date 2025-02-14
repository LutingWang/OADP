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
        
        dataset_2 = LVISDataset
        class_1 = [clean_name(name) for name in dataset.METAINFO['classes']]
        class_2 = [clean_name(name) for name in dataset_2.METAINFO['classes']]
        print(f"len: {len(set(class_1) & set(class_2))}")

    if dataset_name == "lvis":
        mmovod_fs = json.load(open("data/samples/lvis_image_exemplar_dict_K-005_author.json", "r"))
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
                "imagenet_id": imagenet_label2wnid[label]
            }
            print(f"{label} -> {imagenet_label2wnid[label]}")
    
    print(f"overall len: {len(datasets2imagenet)}")

    with open(f"data/{datasets}/annotations/{datasets.lower()}_imagenet_label_map.json", "w") as f:
        json.dump(datasets2imagenet, f, indent=4)


if __name__ == '__main__':
    label_map_path = "data/imagenet21k/annotations/imagenet21k_label_map.json"
    # gen_imagent_fs_datasets(label_map_path, 'V3Det')
    # gen_imagent_fs_datasets(label_map_path, 'coco')
    gen_imagent_fs_datasets(label_map_path, 'lvis')