import os
import json
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from oadp.fs.llama import Chatbot, Captioner, Summarizer, MutiCaptioner
from mmdet.datasets.lvis import LVISV1Dataset
import uuid

def get_crop(img, bb, context=0.0, square=True):
    x1, y1, w, h = bb
    W, H = img.size
    y, x = y1 + h / 2.0, x1 + w / 2.0
    h, w = h * (1. + context), w * (1. + context)
    if square:
        w = max(w, h)
        h = max(w, h)
    x1, x2 = x - w / 2.0, x + w / 2.0
    y1, y2 = y - h / 2.0, y + h / 2.0
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    bb_new = [int(c) for c in [x1, y1, x2, y2]]
    crop = img.crop(bb_new)
    return crop

def run_crop(d, paths, context=0.4, square=True):
    dataset = d['dataset']
    file_name = os.path.join(paths[dataset], d['file_name'])
    if 'imagenet21k_P/train/' in file_name:
        file_name = file_name.replace('imagenet21k_P/train/', '')
    if 'imagenet21k_P/imagenet21k_small_classes/' in file_name:
        file_name = file_name.replace('imagenet21k_P/imagenet21k_small_classes/', '')
    img = Image.open(file_name)
    if dataset == "imagenet21k":
        bb = [0, 0, 0, 0]
        return img
    elif dataset == "lvis":
        bb = [
            int(c)
            for c in [
                d['bbox'][0] // 1,
                d['bbox'][1] // 1,
                d['bbox'][2] // 1 + 1,
                d['bbox'][3] // 1 + 1
            ]
        ]
    elif dataset == "visual_genome":
        bb = [int(c) for c in [d['x'], d['y'], d['w'], d['h']]]
    return get_crop(img, bb, context=context, square=square)

class CropDataset(Dataset):
    def __init__(
        self,
        exemplar_dict,
        paths,
    ):
        self.exemplar_dict = exemplar_dict
        self.paths = paths

    def __len__(self):
        return len(self.exemplar_dict)

    def __getitem__(self, idx):
        chosen_anns = self.exemplar_dict[idx]
        crops = [run_crop(ann, self.paths) for ann in chosen_anns]
        return crops

def gen_test_datasets(output_dir: str) -> None:
    exemplar_dict = json.load(open('data/mmovod-samples/lvis_image_exemplar_dict_K-005_author.json'))
    path_dict = {
        "imagenet21k": "data/imagenet21k/images",
        "visual_genome": "data/VisualGenome",
        "lvis": "data/lvis",
    }
    dataset = CropDataset(exemplar_dict, path_dict)
    annotations = {}
    lvis_categories = LVISV1Dataset.METAINFO['classes']
    label_map = {}
    for idx, category in enumerate(lvis_categories):
        label_map[idx] = {
            "samples_id": idx,
            "name": category
        }
    for idx, images in enumerate(tqdm(dataset)):
        # Create a subfolder for the current index if it doesn't exist
        subfolder = os.path.join(output_dir, "images", str(idx))
        os.makedirs(subfolder, exist_ok=True)
        
        # Get the original annotation list for the current index
        images_path= []
        for j, image in enumerate(images):
            filename = f"{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(subfolder, filename)
            image.save(save_path)
            images_path.append(save_path)
        
        annotations[idx] = images_path
    
    ann_file_path = os.path.join(output_dir, "annotations.json")
    with open(ann_file_path, "w") as f:
        json.dump(annotations, f)
    
    label_map_path = "data/lvis/annotations/mmovod_label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f)    

def single_summarize_label_map() -> None:
    exemplar_dict = json.load(open('data/mmovod-samples/lvis_image_exemplar_dict_K-005_author.json'))
    path_dict = {
        "imagenet21k": "data/imagenet21k",
        "visual_genome": "data/VisualGenome",
        "lvis": "data/lvis",
    }
    chatbot = Chatbot()
    captioner = Captioner(chatbot)
    summarizer = Summarizer(chatbot)
    dataset = CropDataset(exemplar_dict, path_dict)
    few_shot_label_map = []
    for images in tqdm(dataset):
        captions = [captioner(image) for image in images]
        summary = summarizer(captions)
        print(summary)
        cls = refine_label_map(summary)
        few_shot_label_map.append(cls)
    with open('data/mmovod-samples/lvis_single_summarize_label_map.json', 'w') as f:
        json.dump(few_shot_label_map, f)

def multi_summarize_label_map() -> None:
    exemplar_dict = json.load(open('data/mmovod-samples/lvis_image_exemplar_dict_K-005_author.json'))
    path_dict = {
        "imagenet21k": "data/imagenet21k",
        "visual_genome": "data/VisualGenome",
        "lvis": "data/lvis",
    }
    chatbot = Chatbot()
    captioner = MutiCaptioner(chatbot)
    dataset = CropDataset(exemplar_dict, path_dict)
    few_shot_label_map = []
    for images in tqdm(dataset):
        caption = captioner(images)
        cls = refine_label_map(caption)
        few_shot_label_map.append(cls)
    with open('data/mmovod-samples/lvis_multi_summarize_label_map.json', 'w') as f:
        json.dump(few_shot_label_map, f)

def refine_label_map(caption: str) -> str:
    if 'A photo of ' in caption:
        caption = caption.replace('A photo of ', '')
    if 'an ' in caption:
        caption = caption.replace('an ', '')
    if 'a ' in caption:
        caption = caption.replace('a ', '')
    if '.' in caption:
        caption = caption.replace('.', '')
    if ' ' in caption:
        caption = caption.replace(' ', '_')
    return caption.strip()

def refine_label_map_sapce(label_map_path: str) -> None:
    label_map = json.load(open(label_map_path))
    new_label_map = []
    for label in label_map:
        new_label_map.append(label.replace('\n', ''))
    with open(label_map_path, 'w') as f:
        json.dump(new_label_map, f)

# 对比label map
def compare_label_map(label_map_path: str) -> None:
    label_map = json.load(open(label_map_path))
    lvis_dataset = LVISV1Dataset.METAINFO['classes']
    compare_map = []
    for label, lvis_label in zip(label_map, lvis_dataset):
        compare_map.append(f"{label}|{lvis_label}")
    with open('data/samples/lvis_image_exemplar_dict_K-005_author_label_map_compare.txt', 'w') as f:
        f.write('\n'.join(compare_map))


if __name__ == '__main__':
    gen_test_datasets("data/mmovod-samples")
    # single_summarize_label_map()
    # multi_summarize_label_map()
    # refine_label_map_sapce('data/samples/lvis_single_summarize_label_map.json')
