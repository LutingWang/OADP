from mmdet.models import TwoStageDetector
from mmdet.registry import MODELS

from ..utils.globals import Globals

@MODELS.register_module()
class OADP(TwoStageDetector):

    def text_bank(self, data_samples):
        text_batch_list = []
        for data_sample in data_samples:
            text_batch_list.append(data_sample.texts)
        return text_batch_list

    def forward(self, inputs, data_samples = None, mode = 'tensor'):
        Globals.texts = self.text_bank(data_samples=data_samples)
        return super().forward(inputs, data_samples, mode)