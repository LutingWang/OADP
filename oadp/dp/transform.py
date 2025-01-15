import torch
from typing import Optional, Tuple, Union
import os
import os.path as osp
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadFeature(BaseTransform):
    def __init__(self, pth_dir: str, data_root: str) -> None:
        self.data_root = data_root
        self.block_features_dir = osp.join(pth_dir, 'clip_blocks_cuda_train/output')
        self.global_features_dir = osp.join(pth_dir, 'clip_globals_cuda_train/output')
        self.object_features_dir = osp.join(pth_dir, 'clip_objects_cuda_train/output')

    def transform(self, results: dict) -> Union[dict, None]:
        key = results['img_path'].split(self.data_root)[-1].replace('.jpg', '').replace('/','-')
        block_feature_path = osp.join(self.block_features_dir, key + '.pth')
        global_feature_path = osp.join(self.global_features_dir, key + '.pth')
        object_feature_path = osp.join(self.object_features_dir, key + '.pth')
        print(block_feature_path)
        if osp.exists(block_feature_path):
            results['blocks_features'] = torch.load(block_feature_path, map_location='cpu')
        else:
            results['blocks_features'] = None

        if osp.exists(global_feature_path):
            results['globals_features'] = torch.load(global_feature_path, map_location='cpu')
        else:
            results['globals_features'] = None

        if osp.exists(object_feature_path):
            results['objects_features'] = torch.load(object_feature_path, map_location='cpu')
        else:
            results['objects_features'] = None
        return results