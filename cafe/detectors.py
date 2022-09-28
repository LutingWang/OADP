__all__ = [
    'Cafe',
]

from typing import Any, Dict, List, Optional, Tuple
from mmdet.models import DETECTORS, TwoStageDetector
from mmdet.models.utils.builder import LINEAR_LAYERS
import todd
import torch
import einops


@DETECTORS.register_module()
class Cafe(TwoStageDetector):

    def __init__(
        self,
        *args,
        multilabel_classifier: Dict[str, Any],
        multilabel_loss: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        todd.base.init_iter()
        self._multilabel_classifier = LINEAR_LAYERS.build(
            multilabel_classifier,
        )
        self._multilabel_loss = todd.losses.LOSSES.build(
            multilabel_loss,
        )

    def extract_feat(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        logits = self._multilabel_classifier(feat)
        if self.with_neck:
            feats = self.neck(feats)
        return feats, logits

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
        feats, logits = self.extract_feat(img)
        img_labels = img.new_zeros(
            img.shape[0],
            self._multilabel_classifier.num_classes,
            dtype=bool,
        )
        for i, gt_label in enumerate(gt_labels):
            img_labels[i, gt_label] = True

        losses = dict(
            loss_multilabel=self._multilabel_loss(logits.sigmoid(), img_labels),
        )

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                feats,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(feats, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        feats, logits = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            feats, proposal_list, img_metas, rescale=rescale)
