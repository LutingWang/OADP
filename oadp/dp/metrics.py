import os
import torch
import torch.distributed as dist
from mmdet.evaluation.metrics.coco_metric import CocoMetric as BaseCocoMetric
from mmdet.registry import METRICS

@METRICS.register_module(force=True)
class CocoMetric(BaseCocoMetric):
    
    def evaluate(self, *args, **kwargs):
        if self.collect_dir is None:
            random_num: torch.Tensor = torch.randint(100000, (1,1))
            dist.broadcast(random_num, 0)
            self.collect_dir = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/{random_num}'
            os.makedirs(self.collect_dir, exist_ok=True)
            
        super().evaluate(*args, **kwargs)