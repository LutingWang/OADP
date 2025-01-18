import os
import torch
import torch.distributed as dist
from mmdet.evaluation.metrics.coco_metric import CocoMetric as BaseCocoMetric
from mmdet.registry import METRICS

@METRICS.register_module(force=True)
class CocoMetric(BaseCocoMetric):
    
    def evaluate(self, *args, **kwargs):
        if self.collect_dir is None:
            random_num: torch.Tensor = torch.randint(100000, (1,1), device='cuda')
            dist.broadcast(random_num, 0)
            cache_dir = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/cache'
            self.collect_dir = f'{cache_dir}/{random_num.item()}'
            os.makedirs(self.collect_dir, exist_ok=True)
            
        return super().evaluate(*args, **kwargs)