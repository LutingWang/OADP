__all__ = [
    'Cafe',
]

from typing import Any, Dict, List, Optional, Tuple

from mmdet.models import DETECTORS, TwoStageDetector, StandardRoIHead
from mmdet.models.utils.builder import LINEAR_LAYERS
import numpy as np
import todd
import torch
import einops
import sklearn.metrics

from .classifiers import Classifier

from .patches import one_hot


@DETECTORS.register_module()
class Cafe(TwoStageDetector):
    roi_head: StandardRoIHead

    def __init__(
        self,
        *args,
        multilabel_classifier: Dict[str, Any],
        multilabel_loss: Dict[str, Any],
        topK: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        todd.base.init_iter()
        self._multilabel_classifier: Classifier = LINEAR_LAYERS.build(
            multilabel_classifier,
        )
        self._multilabel_loss = todd.losses.LOSSES.build(
            multilabel_loss,
        )
        self._topK = topK

    @property
    def num_classes(self) -> int:
        return self._multilabel_classifier.num_classes

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        gt_masks=None,
        proposals=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        todd.base.inc_iter()

        feats = self.backbone(img)

        multilabel_logits = self.multilabel_classify(feats)

        if self.with_neck:
            feats = self.neck(feats)

        img_labels = one_hot(gt_labels, self.num_classes)
        losses = dict(
            loss_multilabel=self._multilabel_loss(multilabel_logits.sigmoid(), img_labels),
            recall_multilabel=self.topK(multilabel_logits, img_labels),
        )

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get(
                'rpn_proposal',
                self.test_cfg.rpn,
            )
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                feats,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs,
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        with todd.base.setattr_temp(self.roi_head, 'message', (multilabel_logits,)):
            roi_losses = self.roi_head.forward_train(
                feats, img_metas, proposal_list,
                gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks,
                **kwargs,
            )

        losses.update(roi_losses)

        return losses

    def multilabel_classify(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        logits = self._multilabel_classifier(feat)
        return logits

    def topK(
        self,
        multilabel_logits: torch.Tensor,
        img_labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        default_args = dict(
            labels=torch.where(img_labels.sum(0))[0].cpu().numpy(),
            average='macro',
            zero_division=0,
        )
        default_args.update(kwargs)

        logits, inds = multilabel_logits.topk(self._topK)
        preds = one_hot(inds, self.num_classes)
        recall = sklearn.metrics.recall_score(img_labels.cpu().numpy(), preds.cpu().numpy(), **default_args)
        return multilabel_logits.new_tensor(recall * 100)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        assert self.with_bbox

        feats = self.backbone(img)

        multilabel_logits = self.multilabel_classify(feats)

        if self.with_neck:
            feats = self.neck(feats)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals

        _, inds = multilabel_logits.topk(self.num_classes - self._topK, largest=False)
        with todd.base.setattr_temp(self.roi_head, 'message', (multilabel_logits, inds)):
            return self.roi_head.simple_test(
                feats, proposal_list, img_metas, rescale=rescale,
            )
