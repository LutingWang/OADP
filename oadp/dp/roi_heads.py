__all__ = [
    'ViLDEnsembleRoIHead',
]

from typing import Any, Dict, List, cast

import todd
import torch
from mmdet.models import HEADS, BaseRoIExtractor, BBoxHead, StandardRoIHead

from .classifiers import Classifier


@HEADS.register_module()
class ViLDEnsembleRoIHead(StandardRoIHead):
    bbox_roi_extractor: BaseRoIExtractor

    def __init__(
        self,
        *args,
        bbox_head: Dict[str, Any],
        image_head: Dict[str, Any],
        **kwargs,
    ) -> None:
        bbox_head.update(num_classes=todd.globals_.num_classes)
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        self._object_head: BBoxHead = HEADS.build(
            image_head,
            default_args=bbox_head,
        )
        classifier: Classifier = self._object_head.fc_cls
        classifier._bg_embedding.requires_grad_(False)
        cal_lambda = torch.ones(todd.globals_.num_classes + 1) / 3
        cal_lambda[:todd.globals_.num_base_classes] *= 2
        self.register_buffer('_cal_lambda', cal_lambda, persistent=False)

    @property
    def cal_lambda(self) -> torch.Tensor:
        return self._cal_lambda

    def _bbox_forward(
        self,
        x: List[torch.Tensor],
        rois: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bbox_results: Dict[str, torch.Tensor] = super()._bbox_forward(x, rois)
        if todd.globals_.training:
            return bbox_results

        assert not self.with_shared_head

        object_logits, _ = self._object_head(bbox_results['bbox_feats'])
        object_logits = cast(torch.Tensor, object_logits)
        object_logits[:, -1] = float('-inf')

        cal_score: torch.Tensor = (
            bbox_results['cls_score'].softmax(-1)**self.cal_lambda
            * object_logits.softmax(-1)**(1 - self.cal_lambda)
        )
        cal_score[:, -1] = 1 - cal_score[:, :-1].sum(-1)

        cal_logits = cal_score.log()
        bbox_results['cls_score'] = cal_logits
        return bbox_results
