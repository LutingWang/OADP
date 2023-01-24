__all__ = [
    'BlockBBoxHead',
    'ViLDEnsembleRoIHead',
    'OADPRoIHead',
]

from typing import cast

import todd
import torch
from mmdet.core import bbox2roi
from mmdet.models import HEADS, BaseRoIExtractor, BBoxHead, StandardRoIHead

from ..base import Globals
from .bbox_heads import BlockBBoxHead
from .classifiers import Classifier


@HEADS.register_module()
class ViLDEnsembleRoIHead(StandardRoIHead):
    bbox_roi_extractor: BaseRoIExtractor

    def __init__(
        self,
        *args,
        bbox_head: todd.Config,
        object_head: todd.Config,
        **kwargs,
    ) -> None:
        assert bbox_head.num_classes is None
        bbox_head.num_classes = Globals.categories.num_all
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        assert not self.with_shared_head

        self._object_head: BBoxHead = HEADS.build(
            object_head,
            default_args=bbox_head,
        )

        # object head does not perform classification
        classifier: Classifier = self._object_head.fc_cls
        assert classifier._bg_embedding is not None
        classifier._bg_embedding.requires_grad_(False)

        lambda_ = torch.ones(Globals.categories.num_all + 1) / 3
        lambda_[:Globals.categories.num_bases] *= 2
        self.register_buffer('_lambda', lambda_, persistent=False)

    @property
    def lambda_(self) -> torch.Tensor:
        return self._lambda

    def _bbox_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bbox_results: dict[str, torch.Tensor] = super()._bbox_forward(x, rois)
        if Globals.training:
            return bbox_results

        object_logits, _ = self._object_head(bbox_results['bbox_feats'])
        object_logits = cast(torch.Tensor, object_logits)
        object_logits[:, -1] = float('-inf')

        cal_score: torch.Tensor = (
            bbox_results['cls_score'].softmax(-1)**self.lambda_
            * object_logits.softmax(-1)**(1 - self.lambda_)
        )
        cal_score[:, -1] = 1 - cal_score[:, :-1].sum(-1)

        cal_logits = cal_score.log()
        bbox_results['cls_score'] = cal_logits
        return bbox_results

    def _object_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> None:
        bre = self.bbox_roi_extractor
        object_feats = bre(x[:bre.num_inputs], rois)
        self._object_head(object_feats)

    def object_forward_train(
        self,
        x: list[torch.Tensor],
        bboxes: list[torch.Tensor],
    ) -> None:
        rois = bbox2roi(bboxes)
        self._object_forward(x, rois)


@HEADS.register_module()
class OADPRoIHead(ViLDEnsembleRoIHead):

    def __init__(
        self,
        *args,
        bbox_head: todd.Config,
        block_head: todd.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        self._block_head: BlockBBoxHead = HEADS.build(
            block_head,
            default_args=bbox_head,
        )

    def _block_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> torch.Tensor:
        bre = self.bbox_roi_extractor
        block_feats = bre(x[:bre.num_inputs], rois)
        logits, _ = self._block_head(block_feats)
        return logits

    def block_forward_train(
        self,
        x: list[torch.Tensor],
        bboxes: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        rois = bbox2roi(bboxes)
        logits = self._block_forward(x, rois)
        losses = self._block_head.loss(logits[:, :-1], torch.cat(targets))
        return losses
