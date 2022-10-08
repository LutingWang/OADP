__all__ = [
    'Cafe',
]

from abc import ABCMeta
import contextlib
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
        cls_predictor_cfg: Dict[str, Any],
        multilabel_classifier: Optional[Dict[str, Any]] = None,
        multilabel_loss: Optional[Dict[str, Any]] = None,
        topK: Optional[int] = None,
        hidden_dims: Optional[int] = None,
        pre_fpn: Optional[Dict[str, Any]] = None,
        post_fpn: Optional[Dict[str, Any]] = None,
        attn_weights_loss: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        todd.init_iter()

        self.roi_head.bbox_head.fc_cls = LINEAR_LAYERS.build(
            cls_predictor_cfg,
            default_args=dict(
                in_features=self.roi_head.bbox_head.fc_cls.in_features,
                out_features=self.roi_head.bbox_head.fc_cls.out_features,
            ),
        )

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

        self._topK = topK
        self._hidden_dims = hidden_dims

        if hidden_dims is not None:
            self._ce_proj = nn.Sequential(
                nn.Linear(self._multilabel_classifier.embedding_dim, hidden_dims),
                nn.LayerNorm(hidden_dims),
            )
        else:
            self._ce_proj = None

        if hidden_dims is not None and pre_fpn is not None:
            self._pre_fpn = PreFPN(
                out_channels=hidden_dims,
                **pre_fpn,
            )
        else:
            self._pre_fpn = None

        if hidden_dims is not None and post_fpn is not None:
            self._post_fpn = PostFPN(
                channels=hidden_dims,
                **post_fpn,
            )
        else:
            self._post_fpn = None

        if attn_weights_loss is not None:
            self._attn_weights_gt_downsample: str = attn_weights_loss.pop('gt_downsample')
            self._attn_weights_loss = todd.losses.LOSSES.build(
                attn_weights_loss,
            )
        else:
            self._attn_weights_gt_downsample = None
            self._attn_weights_loss = None

    @property
    def num_classes(self) -> int:
        return self._multilabel_classifier.num_classes

    def _multilabel_classify(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        logits: torch.Tensor = self._multilabel_classifier(feat)
        return logits

    def _multilabel_topK_recall(
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

    def _extract_feat(
        self,
        img: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
    ]:
        feats: Tuple[torch.Tensor, ...] = self.backbone(img)

        if self._multilabel_classifier is not None:
            multilabel_logits = self._multilabel_classify(feats)
        else:
            multilabel_logits = None

        if self._topK is not None:
            assert multilabel_logits is not None
            topK_logits, topK_inds = multilabel_logits.topk(self._topK)
        else:
            topK_logits, topK_inds = None, None

        if self._ce_proj is not None:
            assert multilabel_logits is not None
            assert topK_inds is not None
            ce: torch.Tensor = self._multilabel_classifier.embeddings[topK_inds]
            ce = self._ce_proj(ce)
        else:
            ce = None

        if self._pre_fpn is not None:
            assert topK_logits is not None
            assert ce is not None
            feats = self._pre_fpn(feats, ce, topK_logits)

        assert self.with_neck
        feats = self.neck(feats)

        if self._post_fpn is not None:
            assert topK_logits is not None
            assert ce is not None
            feats, masks = self._post_fpn(feats, ce, topK_logits)
            masks = cast(List[torch.Tensor], masks)
        else:
            masks = None

        return feats, multilabel_logits, topK_inds, masks

    def _get_losses(
        self,
        gt_labels: List[torch.Tensor],
        multilabel_logits: Optional[torch.Tensor],
        topK_inds: Optional[torch.Tensor],
        masks: Optional[List[torch.Tensor]],
        gt_masks_tensor: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = dict()

        if multilabel_logits is None:
            return losses

        img_labels = one_hot(gt_labels, self.num_classes)
        multilabel_loss: torch.Tensor = self._multilabel_loss(
            multilabel_logits.sigmoid(),
            img_labels,
        )
        losses.update(loss_multilabel=multilabel_loss)

        if topK_inds is None:
            return losses

        multilabel_topK_recall = self._multilabel_topK_recall(topK_inds, img_labels)
        losses.update(recall_multilabel=multilabel_topK_recall)

        if self._attn_weights_loss is None:
            return losses

        assert gt_masks_tensor is not None
        assert self._attn_weights_gt_downsample is not None
        assert masks is not None
        i = einops.repeat(
            torch.arange(topK_inds.shape[0], device=topK_inds.device),
            'b -> b c',
            c=topK_inds.shape[1],
        )
        gt_masks_tensor = gt_masks_tensor[i, topK_inds].float()
        if self._attn_weights_gt_downsample == 'avg':
            gt_masks_tensor = F.adaptive_avg_pool2d(gt_masks_tensor, masks[0].shape[-2:])
        elif self._attn_weights_gt_downsample == 'max':
            gt_masks_tensor = F.adaptive_max_pool2d(gt_masks_tensor, masks[0].shape[-2:])
        elif self._attn_weights_gt_downsample == 'nearest':
            gt_masks_tensor = F.interpolate(gt_masks_tensor, masks[0].shape[-2:])
        else:
            raise ValueError(self._attn_weights_gt_downsample)
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
        gt_bboxes_ignore: Optional[List[torch.Tensor]] = None,
        gt_masks: Optional[List[BitmapMasks]] = None,
        gt_masks_tensor: Optional[torch.Tensor] = None,
        proposals=None,
        clip_image: Optional[torch.Tensor] = None,
        clip_patches: Optional[List[torch.Tensor]] = None,
        clip_bboxes: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        todd.inc_iter()
        todd.globals_.training = True

        feats, multilabel_logits, topK_inds, masks = self._extract_feat(img)

        losses = self._get_losses(gt_labels, multilabel_logits, topK_inds, masks, gt_masks_tensor)

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

        # TODO: design a switch for this
        if multilabel_logits is not None:
            context = todd.setattr_temp(self.roi_head, 'message', (multilabel_logits,))
        else:
            context = nullcontext()

        with context:
            roi_losses = self.roi_head.forward_train(
                feats, img_metas, proposal_list,
                gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks,
                **kwargs,
            )
        losses.update(roi_losses)

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
            with todd.hooks.hook(
                dict(
                    type='StandardHook',
                    path='.bbox_head.fc_cls._linear',
                ),
                self.roi_head,
            ) as hook_status:
                self.roi_head._bbox_forward(feats, clip_rois)
            custom_tensors.update(patches=hook_status.value)

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
        assert self.with_bbox

        feats, multilabel_logits, topK_inds, masks = self._extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(feats, img_metas)
        else:
            proposal_list = proposals

        if multilabel_logits is not None:
            message = (multilabel_logits,)
            if self._topK is not None:
                _, topK_inds = multilabel_logits.topk(self.num_classes - self._topK, largest=False)
                message = message + (topK_inds,)
            context = todd.base.setattr_temp(self.roi_head, 'message', message)
        else:
            context = nullcontext()

        with context:
            return self.roi_head.simple_test(
                feats, proposal_list, img_metas, rescale=rescale,
            )
