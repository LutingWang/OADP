from mmdet.models import TwoStageDetector
from mmdet.registry import MODELS



@MODELS.register_module()
class OADP(TwoStageDetector):
    
    def forward(self, inputs, data_samples = None, mode = 'tensor'):
        return super().forward(inputs, data_samples, mode)