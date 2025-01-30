import json
from mmdet.datasets.transforms import RandomErasing
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class ReplaceLabel(BaseTransform):
    def __init__(self, label_map_path: str) -> None:
        self._label_map = tuple(json.load(open(label_map_path)))

    def transform(self, results: dict) -> dict:
        results['text'] = self._label_map
        return results
