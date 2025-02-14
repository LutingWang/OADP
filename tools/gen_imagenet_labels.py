import os
import json
import nltk
import random
from tqdm import tqdm
from nltk.corpus import wordnet

# 确保已经下载 wordnet 数据
nltk.download("wordnet")

def wnid_to_classname(wnid):
    """
    输入 ImageNet 21K 的 WNID, 输出对应的类别名称
    :param wnid: str, 例如 "n02119789"
    :return: str, 类别名称
    """
    synset = wordnet.synset_from_pos_and_offset('n', int(wnid[1:]))  # 解析 WNID
    return synset.lemma_names()  # 返回类名（可能是多个）


def gen_labels(data_root: str, 
               output_path: str, 
               output_label_path:str=None, 
               output_label2images_path:str=None,
               debug: bool = False):
    label_map = {}
    label2images = {}
    annotations = {
        "categories": [],
        "images": [],
    }
    dirs = list(os.listdir(data_root))

    if debug:
        dirs = dirs[:16]
    
    # list all the images
    for i, dir_name in enumerate(tqdm(dirs)):
        # get the class name
        class_name = wnid_to_classname(dir_name)[0]
        images = []
        for img_name in os.listdir(os.path.join(data_root, dir_name)):
            images.append(os.path.join(dir_name, img_name))
        annotations["categories"].append({
            "id": i,
            "name": class_name,
            "wnid": dir_name,
            "images": images,
        })
        label_map[dir_name] = class_name
        label2images[dir_name] = images
    
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=4)

    if output_label_path is not None:
        with open(output_label_path, "w") as f:
            json.dump(label_map, f, indent=4)    

    if output_label2images_path is not None:
        with open(output_label2images_path, "w") as f:
            json.dump(label2images, f, indent=4)

if __name__ == "__main__":
    gen_labels("data/imagenet21k/images", 
               "data/imagenet21k/annotations/imagenet21k_labels.json",
                "data/imagenet21k/annotations/imagenet21k_label_map.json",
                "data/imagenet21k/annotations/imagenet21k_label2images.json",
               debug=False)