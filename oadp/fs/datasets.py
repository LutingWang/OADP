from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDataset
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