import json
import torch
import os
import os.path as osp
from PIL import Image
from typing import List, Optional
from torch.utils.data import Dataset
from todd.patches.pil.image import convert_rgb

class ODVGDataset(Dataset):
    """Object Detection and Visual Grounding Dataset for PyTorch."""
    DATA_ROOT = ''
    ANNOTATIONS_FILE = ''
    IMAGE_ROOT = ''
    LABEL_MAP = None

    def __init__(self, need_text: bool = True):
        """
        Args:
            ann_file (str): Path to the annotation file.
            data_root (str): Root directory of the dataset.
            label_map_file (Optional[str]): Path to the label map file.
            need_text (bool): Whether text information is required.
            transform: Transformations to apply to the data.
        """
        self.ann_file = self.ANNOTATIONS_FILE
        self.img_prefix = self.IMAGE_ROOT
        self.data_root = self.DATA_ROOT
        self.need_text = need_text
        self.dataset_mode = 'VG'
        self.label_map = self.LABEL_MAP

        if self.label_map:
            label_map_path = os.path.join(self.data_root, self.label_map)
            with open(label_map_path, 'r') as file:
                self.label_map = json.load(file)
            self.dataset_mode = 'OD'

        self.data_list = self.load_data_list()
        

    def load_data_list(self) -> List[dict]:
        local_path = osp.join(self.data_root, self.ann_file)
        with open(local_path, 'r') as f:
            data_list = [json.loads(line) for line in f]

        out_data_list = []
        for data in data_list:
            data_info = {}
            img_path = osp.join(self.data_root, self.img_prefix, data['filename'])
            data_info['filename'] = data['filename']
            data_info['img_path'] = img_path
            data_info['height'] = data['height']
            data_info['width'] = data['width']
            if self.dataset_mode == 'OD':
                if self.need_text:
                    data_info['text'] = self.label_map
                anno = data.get('detection', {})
                instances = [obj for obj in anno.get('instances', [])]
                bboxes = [obj['bbox'] for obj in instances]
                bbox_labels = [str(obj['label']) for obj in instances]

                instances = []
                for bbox, label in zip(bboxes, bbox_labels):
                    instance = {}
                    x1, y1, x2, y2 = bbox
                    inter_w = max(0, min(x2, data['width']) - max(x1, 0))
                    inter_h = max(0, min(y2, data['height']) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue
                    instance['ignore_flag'] = 0
                    instance['bbox'] = bbox
                    instance['bbox_label'] = int(label)
                    instances.append(instance)
                data_info['instances'] = instances
                data_info['dataset_mode'] = self.dataset_mode
                out_data_list.append(data_info)
            else:
                anno = data['grounding']
                data_info['text'] = anno['caption']
                regions = anno['regions']

                instances = []
                phrases = {}
                for i, region in enumerate(regions):
                    bbox = region['bbox']
                    phrase = region['phrase']
                    tokens_positive = region['tokens_positive']
                    if not isinstance(bbox[0], list):
                        bbox = [bbox]
                    for box in bbox:
                        instance = {}
                        x1, y1, x2, y2 = box
                        inter_w = max(0, min(x2, data['width']) - max(x1, 0))
                        inter_h = max(0, min(y2, data['height']) - max(y1, 0))
                        if inter_w * inter_h == 0:
                            continue
                        if (x2 - x1) < 1 or (y2 - y1) < 1:
                            continue
                        instance['ignore_flag'] = 0
                        instance['bbox'] = box
                        instance['bbox_label'] = i
                        phrases[i] = {
                            'phrase': phrase,
                            'tokens_positive': tokens_positive
                        }
                        instances.append(instance)
                data_info['instances'] = instances
                data_info['phrases'] = phrases
                data_info['dataset_mode'] = self.dataset_mode
                out_data_list.append(data_info)

        del data_list
        return out_data_list

    def __len__(self):
        return len(self.data_list)
    
    def _getitem(self, idx):
        data = self.data_list[idx]
        img_path = data['img_path']
        image = convert_rgb(Image.open(img_path))
        return data['filename'], image, data['instances']