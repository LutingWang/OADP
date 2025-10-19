import json

from mmdet.evaluation.metrics.lvis_metric import LVISFixedAPMetric
from mmengine.evaluator.metric import BaseMetric
from mmdet.registry import METRICS



@METRICS.register_module()
class LVISResultExportMetric(LVISFixedAPMetric):
    def __init__(self, *args, save_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path

    def compute_metrics(self, results: dict) -> dict:
        breakpoint()
        with open(self.save_path, 'w') as f:
            json.dump(results, f)
        return super().compute_metrics(results)