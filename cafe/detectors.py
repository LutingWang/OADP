__all__ = [
    'Cafe',
]

from abc import ABCMeta
from typing import Any, Dict, List, Optional, Tuple, cast

from mmdet.core import BitmapMasks, bbox2roi
from mmdet.models import DETECTORS, TwoStageDetector, StandardRoIHead, RPNHead
from mmdet.models.utils.builder import LINEAR_LAYERS
import numpy as np
import todd
import torch
import einops
import sklearn.metrics
import torch.nn as nn
import torch.nn.functional as F

from .classifiers import Classifier
from .necks import PreFPN, PostFPN
from .patches import one_hot


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
        multilabel_classifier: Dict[str, Any],
        multilabel_loss: Dict[str, Any],
        topK: int,
        hidden_dims: int,
        pre_fpn: Dict[str, Any],
        post_fpn: Dict[str, Any],
        attn_weights_loss: Dict[str, Any],
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
        self._hidden_dims = hidden_dims

        self._ce_proj = nn.Sequential(
            nn.Linear(self._multilabel_classifier.embedding_dim, hidden_dims),
            nn.LayerNorm(hidden_dims),
        )

        self._pre_fpn = PreFPN(
            out_channels=hidden_dims,
            **pre_fpn,
        )
        self._post_fpn = PostFPN(
            channels=hidden_dims,
            **post_fpn,
        )

        self._gt_downsample = attn_weights_loss.pop('gt_downsample')

        self._attn_weights_loss = todd.losses.LOSSES.build(
            attn_weights_loss,
        )

    @property
    def num_classes(self) -> int:
        return self._multilabel_classifier.num_classes

    def extract_feat(
        self,
        img: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        feats = self.backbone(img)

        multilabel_logits = self.multilabel_classify(feats)

        topK_logits, topK_inds = multilabel_logits.topk(self._topK)

        ce = self._multilabel_classifier.embeddings[topK_inds]
        ce = self._ce_proj(ce)

        feats = self._pre_fpn(feats, ce, topK_logits)

        assert self.with_neck
        feats = self.neck(feats)

        feats, masks = self._post_fpn(feats, ce, topK_logits)
        return feats, multilabel_logits, topK_inds, masks

    def get_losses(
        self,
        gt_labels: List[torch.Tensor],
        multilabel_logits: torch.Tensor,
        topK_inds: torch.Tensor,
        masks: List[torch.Tensor],
        gt_masks_tensor: Optional[torch.Tensor] = None,
    ):
        img_labels = one_hot(gt_labels, self.num_classes)
        multilabel_loss = self._multilabel_loss(multilabel_logits.sigmoid(), img_labels)
        multilabel_topK_recall = self.multilabel_topK_recall(topK_inds, img_labels)

        losses = dict(
            loss_multilabel=multilabel_loss,
            recall_multilabel=multilabel_topK_recall,
        )

        if gt_masks_tensor is not None:
            i = einops.repeat(
                torch.arange(topK_inds.shape[0], device=topK_inds.device),
                'b -> b c',
                c=topK_inds.shape[1],
            )
            gt_masks_tensor = gt_masks_tensor[i, topK_inds].float()
            if self._gt_downsample == 'avg':
                gt_masks_tensor = F.adaptive_avg_pool2d(gt_masks_tensor, masks[0].shape[-2:])
            elif self._gt_downsample == 'max':
                gt_masks_tensor = F.adaptive_max_pool2d(gt_masks_tensor, masks[0].shape[-2:])
            elif self._gt_downsample == 'nearest':
                gt_masks_tensor = F.interpolate(gt_masks_tensor, masks[0].shape[-2:])
            else:
                raise ValueError(self._gt_downsample)
            losses.update(
                loss_attn_weights=sum(
                    self._attn_weights_loss(mask, gt_masks_tensor)
                    for mask in masks
                ),
            )

        return losses

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        clip_image: torch.Tensor,
        clip_patches: List[torch.Tensor],
        clip_bboxes: List[torch.Tensor],
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        gt_masks: Optional[List[BitmapMasks]] = None,
        gt_masks_tensor: Optional[torch.Tensor] = None,
        proposals=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        todd.base.inc_iter()

        feats, multilabel_logits, topK_inds, masks = self.extract_feat(img)

        losses = self.get_losses(gt_labels, multilabel_logits, topK_inds, masks, gt_masks_tensor)

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

        clip_rois = bbox2roi(clip_bboxes)
        with todd.hooks.hook(
            dict(
                type='StandardHook',
                path='.bbox_head.fc_cls._linear',
            ),
            self.roi_head,
        ) as hook_status:
            self.roi_head._bbox_forward(feats, clip_rois)

        distiller = cast(todd.distillers.DistillableProto, self).distiller
        distiller.track_tensors()
        distill_losses = distiller.distill(
            dict(
                clip_image=clip_image,
                patches=hook_status.value,
                clip_patches=torch.cat(clip_patches),
            ),
        )
        distiller.reset()
        losses.update(distill_losses)

        return losses

    def multilabel_classify(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        logits = self._multilabel_classifier(feat)
        return logits

    def multilabel_topK_recall(
        self,
        topK_inds: torch.Tensor,
        img_labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        default_args = dict(
            labels=torch.where(img_labels.sum(0))[0].cpu().numpy(),
            average='macro',
            zero_division=0,
        )
        default_args.update(kwargs)

        preds = one_hot(topK_inds, self.num_classes)
        recall = sklearn.metrics.recall_score(img_labels.cpu().numpy(), preds.cpu().numpy(), **default_args)
        return torch.tensor(recall * 100, device=topK_inds.device)

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, Any]],
        proposals=None,
        rescale: bool = False,
    ):
        assert self.with_bbox

        feats, multilabel_logits, topK_inds, masks = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals

        _, inds = multilabel_logits.topk(self.num_classes - self._topK, largest=False)
        with todd.base.setattr_temp(self.roi_head, 'message', (multilabel_logits, inds)):
            return self.roi_head.simple_test(
                feats, proposal_list, img_metas, rescale=rescale,
            )
