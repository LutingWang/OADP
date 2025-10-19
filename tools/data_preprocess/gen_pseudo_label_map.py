import json
import pathlib
from typing import Dict, List

import torch
from tqdm import tqdm
from mmdet.datasets import LVISV1Dataset


def generate_mmovod_pseudo_list(data_root: str, save_path: str = 'data/grounding_data/mmovod/pseudo_list.pth') -> Dict[str, List[str]]:
    """
    Generate and save mapping between class names and their corresponding file paths.
    
    Args:
        data_root: Path to the directory containing the data files
        save_path: Path where to save the annotation map JSON file
        
    Returns:
        Dict mapping class names to lists of their file paths
    """
    root = pathlib.Path(data_root)
    classes_name = LVISV1Dataset.METAINFO['classes']
    pseudo_list = []
    pth_list = []

    for i in range(len(classes_name)):
        pth, _ = torch.load(root / f'{i}.pth', 'cpu')
        pseudo_list.append("[UNK]" * (len(pth) - 3))
        pth_list.append(pth)

    torch.save({"pseudo_list": pseudo_list, "pth_list": pth_list}, save_path)
        

def generate_qwen_annotation_map(data_root: str, save_path: str = 'data/grounding_data/qwen/annotations/name2pth.json') -> Dict[str, List[str]]:
    """
    Generate and save mapping between class names and their corresponding file paths.
    
    Args:
        data_root: Path to the directory containing the data files
        save_path: Path where to save the annotation map JSON file
        
    Returns:
        Dict mapping class names to lists of their file paths
    """
    root = pathlib.Path(data_root)
    class_name_to_file = dict()
    
    files = list(root.iterdir())
    for f in tqdm(files, desc="Processing files"):
        _, _, new_class_name = torch.load(f, 'cpu')
        new_class_name = new_class_name.replace('A photo of', '').strip()
        class_name_to_file[new_class_name] = str(f)

    # Save the annotation map to a JSON file
    with open(save_path, 'w') as f:
        json.dump(class_name_to_file, f, indent=2)
        
    return class_name_to_file

def generate_pseudo_label_map(
    label_map_path: str,
    name2pth_path: str = "data/grounding_data/qwen/annotations/name2pth.json"
) -> Dict[str, List[str]]:
    """
    Generate pseudo label map based on the given label map.
    
    Args:
        label_map_path: Path to the label map JSON file
        name2pth_path: Path to the name2pth mapping file
        
    Returns:
        Dict mapping class IDs to their pseudo labels
    """
    label_map = json.load(open(label_map_path))
    name2pth = json.load(open(name2pth_path))
    
    # Generate save path based on label_map_path
    label_map_dir = pathlib.Path(label_map_path).parent
    save_path = str(label_map_dir / "pseudo_label_map.json")
    
    pseudo_label_map = dict()
    for k, v in label_map.items():
        features, _, _ = torch.load(name2pth[v], 'cpu')
        feature_length = len(features)
        pseudo_label_map[k] = "[UNK]" * (feature_length - 3)

    with open(save_path, 'w') as f:
        json.dump(pseudo_label_map, f, indent=2)

    return pseudo_label_map


if __name__ == '__main__':
    # 先取消注释这两行，确保先生成name2pth.json文件
    qwen_data_path = 'data/grounding_data/qwen/features'
    generate_qwen_annotation_map(qwen_data_path)

    # 然后再执行其他操作
    label_map_path = 'data/grounding_data/objects365/annotations/objects365_label_map_refine.json'
    generate_pseudo_label_map(label_map_path)
    label_map_path = 'data/grounding_data/v3det/annotations/v3det_2023_v1_label_map_refine.json'
    generate_pseudo_label_map(label_map_path)
    mmovod_data_path = 'data/grounding_data/mmovod/features'
    generate_mmovod_pseudo_list(mmovod_data_path)