import json
import os
import torch
from tqdm import tqdm
from mmdet.datasets.lvis import LVISV1Dataset

def normalize(s: str) -> str:
    s, *_ = s.split('/', 1)
    s, *_ = s.split('(', 1)
    s, *_ = s.split('[', 1)
    s = s.replace('_', ' ')
    s = s.replace('-', ' ')
    s = s.lower().strip()
    return s

def pth_helper(image_path: str):
    # Save the aggregated features to a .pth file.
    samples_data_root = "data/grounding_data/imagenet-21k/features"
    # Assume image_path is 'data/imagenet-21k/n11908549/n11908549_2429.JPEG'
    # We extract the folder name and image id.
    base_name = os.path.basename(image_path)            # n11908549_2429.JPEG
    folder_name = os.path.basename(os.path.dirname(image_path))  # n11908549
    image_id = os.path.splitext(base_name)[0]             # n11908549_2429

    # Build the path to the .pth file, e.g., 'data/imagenet-21k/n11908549/n11908549.pth'
    pth_path = os.path.join(samples_data_root, f"{folder_name}.pth")
    # Load the .pth file features
    features = torch.load(pth_path, map_location='cpu')
    index = features['ids'].index(image_id)
    clip_features = features['clip_features'][index].unsqueeze(0)
    dino_features = features['dino_features'][index].unsqueeze(0)
    return torch.cat([clip_features, dino_features], dim=1)

def gen_lvis_val(text2pth:str):
    text2pth = json.load(open(text2pth, "r"))
    category_list = LVISV1Dataset.METAINFO['classes']
    # Dictionary to hold concatenated features per category.
    aggregated_features = {}
    for category in tqdm(category_list):
        # Assume text2pth maps each category to a list of image paths.
        image_paths = text2pth[normalize(category)]
        category_features = []

        for img_path in image_paths:
            # Use pth_helper (assumed to be a method of self) to extract features.
            feat = pth_helper(img_path)
            category_features.append(feat)

        if category_features:
            # Concatenate all features along the batch dimension.
            aggregated = torch.cat(category_features, dim=0)
            aggregated_features[normalize(category)] = aggregated

    save_path = os.path.join('data/grounding_data/imagenet-21k/annotations', "lvis_ref.pth")
    torch.save(aggregated_features, save_path)




if __name__ == "__main__":
    text2pth = 'data/grounding_data/imagenet-21k/annotations/merged.json'
    gen_lvis_val(text2pth)