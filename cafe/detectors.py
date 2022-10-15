__all__ = [
    'Cafe',
]

from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple, cast

from mmdet.core import BitmapMasks, bbox2roi
from mmdet.models import DETECTORS, TwoStageDetector, StandardRoIHead, RPNHead, BBoxHead
from mmdet.models.utils.builder import LINEAR_LAYERS
import numpy as np
import todd
import torch
import einops

from .classifiers import Classifier, ViLDClassifier
from .necks import PostFPN
from .patches import one_hot


class nullcontext:  # compat odps

    def __enter__(self):
        pass

    def __exit__(self, *excinfo):
        pass


@DETECTORS.register_module()
class Cafe(
    TwoStageDetector,
    metaclass=todd.distillers.build_metaclass(
        todd.distillers.SelfDistiller, ABCMeta,
    ),
):
    rpn_head: RPNHead
    roi_head: StandardRoIHead

    def __init__(
        self,
        *args,
        multilabel_classifier: Optional[Dict[str, Any]] = None,
        multilabel_loss: Optional[Dict[str, Any]] = None,
        post_fpn: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        todd.init_iter()

        if multilabel_classifier is not None:
            self._multilabel_classifier: Classifier = LINEAR_LAYERS.build(
                multilabel_classifier,
            )
        else:
            self._multilabel_classifier = None

        if multilabel_classifier is not None:
            self._multilabel_loss = todd.losses.LOSSES.build(
                multilabel_loss,
            )
        else:
            self._multilabel_loss = None

        if post_fpn is not None:
            self._post_fpn = PostFPN(
                **post_fpn,
            )
        else:
            self._post_fpn = None

    @property
    def num_classes(self) -> int:
        return self._multilabel_classifier.num_classes

    def _multilabel_classify(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        logits: torch.Tensor = self._multilabel_classifier(feat)
        return logits

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        gt_masks: Optional[List[BitmapMasks]] = None,
        proposals=None,
        clip_image: Optional[torch.Tensor] = None,
        clip_patches: Optional[List[torch.Tensor]] = None,
        clip_bboxes: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        todd.inc_iter()
        todd.globals_.training = True

        feats = self.extract_feat(img)

        losses = dict()

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

        roi_losses = self.roi_head.forward_train(
            feats, img_metas, proposal_list,
            gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks,
            **kwargs,
        )
        losses.update(roi_losses)

        if self._post_fpn is not None:
            feats = self._post_fpn(feats)

        if self._multilabel_classifier is not None:
            multilabel_logits = self._multilabel_classify(feats)
            img_labels = one_hot(gt_labels, self.num_classes)
            multilabel_loss: torch.Tensor = self._multilabel_loss(
                multilabel_logits.sigmoid(),
                img_labels,
            )
            losses.update(loss_multilabel=multilabel_loss)

        distiller = cast(todd.distillers.DistillableProto, self).distiller
        distiller_spec = distiller.spec()

        custom_tensors = dict()
        if 'clip_image' in distiller_spec.inputs:
            assert clip_image is not None
            custom_tensors.update(clip_image=clip_image.float())
        if 'clip_patches' in distiller_spec.inputs:
            assert clip_patches is not None
            custom_tensors.update(clip_patches=torch.cat(clip_patches).float())
        if 'patches' in distiller_spec.inputs:
            assert clip_bboxes is not None
            clip_rois = bbox2roi(clip_bboxes)
            custom_tensors.update(
                patches=self.roi_head._bbox_forward_distill(
                    feats, clip_rois,
                ),
            )

        if len(distiller_spec.outputs) > 0:
            distill_losses = distiller.distill(custom_tensors)
            distiller.reset()
            losses.update(distill_losses)

        return losses

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        proposals=None,
        rescale: bool = False,
    ):
        todd.globals_.training = False

        assert self.with_bbox, 'Bbox head must be implemented.'
        feats = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals

        if self._post_fpn is not None:
            image_feats = self._post_fpn(feats)
            if self._multilabel_classifier is not None:
                pass
            feats = [torch.stack(feat) for feat in zip(feats, image_feats)]
        else:
            if self._multilabel_classifier is not None:
                pass

        return self.roi_head.simple_test(
            feats,
            proposal_list,
            img_metas,
            rescale=rescale,
        )
