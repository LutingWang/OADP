import torch
import copy

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.detectors.base import BaseDetector
from mmdet.utils import OptConfigType
from mmdet.registry import MODELS

from dp_detector.image_model import FSDetectorFeatureDistill


@MODELS.register_module()
class EnsembledDetector(BaseDetector):
    def __init__(
        self,
        *args,
        text_detector: OptConfigType = None,
        image_detector: OptConfigType = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if text_detector is None or image_detector is None:
            raise ValueError("Both text_detector and image_detector must be provided")
        self.text_detector: GroundingDINO = MODELS.build(text_detector)
        self.image_detector: FSDetectorFeatureDistill = MODELS.build(image_detector)

    def _forward(self, batch_inputs, batch_data_samples):
        pass

    def extract_feat(self, batch_inputs):
        pass

    def loss(self, batch_inputs, batch_data_samples):
        pass

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_batch_data_samples = copy.deepcopy(batch_data_samples)
        text_batch_data_samples[0].text = text_batch_data_samples[0].real_text
        text_results = self.text_detector.predict(
            batch_inputs, text_batch_data_samples, rescale
        )
        image_batch_data_samples = copy.deepcopy(batch_data_samples)
        image_batch_data_samples[0].text = image_batch_data_samples[0].pseudo_text
        image_results = self.image_detector.predict(
            batch_inputs, image_batch_data_samples, rescale
        )
        # merge text_pred_instances and image_pred_instances
        text_pred_instances = text_results[0].pred_instances
        image_pred_instances = image_results[0].pred_instances

        labels = torch.cat([text_pred_instances.labels, image_pred_instances.labels])
        text_pred_instances.set_field(labels, "labels")
        scores = torch.cat([text_pred_instances.scores, image_pred_instances.scores])
        text_pred_instances.set_field(scores, "scores")
        bboxes = torch.cat([text_pred_instances.bboxes, image_pred_instances.bboxes])
        text_pred_instances.set_field(bboxes, "bboxes")
        text_results[0].set_field(text_pred_instances, "pred_instances")
        return text_results
