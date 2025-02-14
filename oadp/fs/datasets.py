import json
import os.path as osp

from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDataset
from mmdet.datasets.odvg import ODVGDataset
from mmengine.fileio import join_path, load

@DATASETS.register_module()
class ImageNet21KDataset(BaseDataset):
    def load_data_list(self):
        annotations = load(self.ann_file)
        data_list = []
        for data_info in annotations['categories']:
            imgs_path = [join_path(self.data_prefix['img'], img) for img in data_info['images']]
            data_info['img_path'] = imgs_path
            data_info['text'] = data_info['name']
            data_list.append(data_info)
        return data_list

@DATASETS.register_module()
class FsODVGDataset(ODVGDataset):
    def __init__(self, *args, imagenet_label_map = '', **kwargs):
        imagenet_label_map = osp.join(kwargs['data_root'], imagenet_label_map)
        with open(imagenet_label_map, 'r') as file:
            self.imagenet_label_map = json.load(file)
        super().__init__(*args, **kwargs)

    def filter_data(self):
        filtered_data_list = []
        filtered_num = 0
        for data in self.data_list:
            filtered_instances = []
            for instance in data['instances']:
                if str(instance['bbox_label']) in self.imagenet_label_map:
                    filtered_instances.append(instance)
            if len(filtered_instances) > 0:
                data['instances'] = filtered_instances
                filtered_data_list.append(data)
            else:
                filtered_num += 1
        return filtered_data_list