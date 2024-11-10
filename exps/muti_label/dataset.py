import clip
import torch
import os
import numpy as np
from PIL import Image
from lvis.lvis import LVIS

from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS, TRANSFORMS

from exps.muti_label.globals import cur_cates

@DATASETS.register_module()
class LVISDataset(BaseDataset):

    def parse_data_info(self, raw_data_info):
        raw_ann_info = raw_data_info['raw_ann_info']
        raw_img_info = raw_data_info['raw_img_info']

        # print(raw_img_info)
        # print([cur_cates[ann['category_id']-1] for ann in raw_ann_info])
        # to one-hot
        category_ids = torch.unique(torch.tensor([ann['category_id'] for ann in raw_ann_info])) - 1
        cate_one_hot = torch.eye(len(cur_cates))[category_ids].sum(dim=0)

        return {
            "img_path": os.path.join(self.data_root, raw_img_info['file_name']),
            "gt_label": cate_one_hot,
        }

    def load_data_list(self) -> list[dict]:

        self.lvis = LVIS(self.ann_file)

        img_ids = self.lvis.get_img_ids()
        data_list = []

        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            raw_img_info['file_name'] = raw_img_info['coco_url'].replace(
                'http://images.cocodataset.org/', '')
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)

            if len(raw_ann_info) == 0:
                # print(f"Image {img_id} has no annotations, skipped.")
                continue

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)

        del self.lvis
        
        return data_list

@TRANSFORMS.register_module()
class LoadImage:
    def __call__(self, data: dict) -> Image.Image:
        data['img'] = Image.open(data['img_path'])
        return data

@TRANSFORMS.register_module()
class CLIPTransforms:
    def __init__(self) -> None:
        _, self.clip_transforms = clip.load("ViT-B/32", device="cpu")

    def __call__(self, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        data['img'] = self.clip_transforms(data['img'])
        return data
    
@TRANSFORMS.register_module()
class PackData:
    def __call__(self, data: dict) -> dict:
        packed_results = {}
        packed_results['data_samples'] = data['gt_label']
        packed_results['batch_inputs'] = data['img']
        return packed_results