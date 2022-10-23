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
import sklearn.metrics

from mldec import debug

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
        num_classes: int,
        multilabel_classifier: Optional[Dict[str, Any]] = None,
        multilabel_loss: Optional[Dict[str, Any]] = None,
        post_fpn: Optional[Dict[str, Any]] = None,
        # caption_loss: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        todd.globals_.num_classes = num_classes

        super().__init__(*args, **kwargs)

        todd.init_iter()

        if multilabel_classifier is not None:
            self._multilabel_topK = multilabel_classifier.pop('topK')
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

        # if caption_loss is not None:
        #     self._caption_loss = todd.losses.LOSSES.build(
        #         caption_loss,
        #     )
        # else:
        #     self._caption_loss = None

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
        proposals: Optional[List[torch.Tensor]] = None,
        clip_image: Optional[torch.Tensor] = None,
        clip_patch_feats: Optional[List[torch.Tensor]] = None,
        clip_patches: Optional[List[torch.Tensor]] = None,
        clip_patch_labels: Optional[List[torch.Tensor]] = None,
        clip_bbox_feats: Optional[List[torch.Tensor]] = None,
        clip_bboxes: Optional[List[torch.Tensor]] = None,
        # clip_captions: Optional[torch.Tensor] = None,
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

        if self._multilabel_classifier is not None:
            multilabel_logits = self._multilabel_classify(feats)
            img_labels = one_hot(gt_labels, self.num_classes)
            multilabel_loss: torch.Tensor = self._multilabel_loss(
                multilabel_logits.sigmoid(),
                img_labels,
            )
            losses.update(loss_multilabel=multilabel_loss)

            topK_logits, topK_inds = multilabel_logits.topk(self._multilabel_topK)
            topK_preds = one_hot(topK_inds, self.num_classes)
            topK_recall = sklearn.metrics.recall_score(
                img_labels.cpu().numpy(),
                topK_preds.cpu().numpy(),
                labels=torch.where(img_labels.sum(0))[0].cpu().numpy(),
                average='macro',
                zero_division=0,
            )
            multilabel_topK_recall = torch.tensor(
                topK_recall * 100,
                device=topK_inds.device,
            )
            losses.update(recall_multilabel=multilabel_topK_recall)

            ce = self._multilabel_classifier.embeddings[topK_inds]
        else:
            ce = None

        if self._post_fpn is not None:
            feats = self._post_fpn(feats, ce)

        if self.roi_head.with_patch:
            assert clip_patches is not None
            assert clip_patch_labels is not None
            clip_rois = bbox2roi(clip_patches)
            patch_feats, patch_loss, patch_topK_recall = self.roi_head._bbox_forward_patch(
                feats, clip_rois, torch.cat(clip_patch_labels),
            )
            losses.update(
                loss_patch=patch_loss,
                recall_patch_topK=patch_topK_recall,
            )
        else:
            patch_feats = None

        distiller = cast(todd.distillers.DistillableProto, self).distiller
        distiller_spec = distiller.spec()

        custom_tensors = dict()
        if 'clip_image' in distiller_spec.inputs:
            assert clip_image is not None
            custom_tensors.update(clip_image=clip_image.float())
        if 'clip_patch_feats' in distiller_spec.inputs:
            assert clip_patch_feats is not None
            custom_tensors.update(clip_patch_feats=torch.cat(clip_patch_feats).float())
        if 'patch_feats' in distiller_spec.inputs:
            assert patch_feats is not None
            custom_tensors.update(patch_feats=patch_feats)
        if 'clip_bbox_feats' in distiller_spec.inputs:
            assert clip_bbox_feats is not None
            custom_tensors.update(clip_bbox_feats=torch.cat(clip_bbox_feats).float())
        if 'bbox_feats' in distiller_spec.inputs:
            assert clip_bboxes is not None
            clip_rois = bbox2roi(clip_bboxes)
            custom_tensors.update(
                bbox_feats=self.roi_head._bbox_forward_distill(
                    feats, clip_rois,
                ),
            )
        # if 'clip_captions' in distiller_spec.inputs:
        #     assert clip_captions is not None
        #     custom_tensors.update(clip_captions=clip_captions.float())

        if len(distiller_spec.outputs) > 0:
            distill_losses = distiller.distill(custom_tensors)
            distiller.reset()
            losses.update(distill_losses)

        return losses

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        proposals: Optional[List[torch.Tensor]] = None,
        clip_patches: Optional[List[torch.Tensor]] = None,
        rescale: bool = False,
    ):
        todd.globals_.training = False

        assert self.with_bbox, 'Bbox head must be implemented.'
        feats = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals

        if self._multilabel_classifier is not None:
            multilabel_logits = self._multilabel_classify(feats)
            if debug.DUMP:
                todd.globals_.multilabel_logits = multilabel_logits

            topK_logits, topK_inds = multilabel_logits.topk(self._multilabel_topK)
            ce = self._multilabel_classifier.embeddings[topK_inds]
        else:
            ce = None

        if self._post_fpn is not None:
            todd.globals_.extra_feats = self._post_fpn(feats, ce)

        if debug.DUMP:
            clip_rois = bbox2roi(clip_patches[0])
            todd.globals_.clip_rois = clip_rois

        return self.roi_head.simple_test(
            feats,
            proposal_list,
            img_metas,
            rescale=rescale,
        )
