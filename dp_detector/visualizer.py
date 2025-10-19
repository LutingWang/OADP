from mmdet.visualization import DetLocalVisualizer
from mmdet.registry import VISUALIZERS


@VISUALIZERS.register_module()
class DetVisualizer(DetLocalVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_datasample(self, *args, **kwargs):
        super().add_datasample(*args, **kwargs)
