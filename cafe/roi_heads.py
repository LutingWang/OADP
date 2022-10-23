__all__ = [
    'MessageMixin',
    'StandardRoIHead',
    'DoubleHeadRoIHead',
]

import os
from typing import Dict, Sequence, List, cast, Any, Optional

import einops
import mmdet.models
import torch
import todd

from mldec import debug

from .classifiers import Classifier


class MessageMixin(mmdet.models.StandardRoIHead):

    def _bbox_forward(self, x: Sequence[torch.Tensor], rois: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = super()._bbox_forward(x, rois)
        if not hasattr(self, 'message'):
            return results

        message = getattr(self, 'message')
        if len(message) == 1:  # train
            logits = message[0].detach()
            absent_classes = None
        elif len(message) == 2:  # test
            logits, absent_classes = message
        else:
            raise ValueError(len(message))

        cls_score = results['cls_score']
        for i in range(logits.shape[0]):
            sample_inds = rois[:, 0].eq(i)
            sample_cls_score = cls_score[sample_inds]
            sample_cls_score[:, :-1] += logits[[i]] * 0.1
            if absent_classes is not None:
                sample_cls_score[:, absent_classes[i]] = float('-inf')
            cls_score[sample_inds] = sample_cls_score  # useful in test time

        return results


@mmdet.models.HEADS.register_module(force=True)
class StandardRoIHead(MessageMixin, mmdet.models.StandardRoIHead):

    def _bbox_forward_distill(self, x: Sequence[torch.Tensor], rois: torch.Tensor) -> torch.Tensor:
        with todd.hooks.hook(
            dict(
                type='StandardHook',
                path='.bbox_head.fc_cls._linear',
            ),
            self,
        ) as hook_status:
            self._bbox_forward(x, rois)
        return hook_status.value


@mmdet.models.HEADS.register_module(force=True)
class DoubleHeadRoIHead(MessageMixin, mmdet.models.DoubleHeadRoIHead):
    bbox_roi_extractor: mmdet.models.BaseRoIExtractor
    bbox_head: mmdet.models.DoubleConvFCBBoxHead

    def _bbox_forward_distill(self, x: Sequence[torch.Tensor], rois: torch.Tensor) -> torch.Tensor:
        x_cls: torch.Tensor = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois,
        )
        if self.with_shared_head:
            x_cls = self.shared_head(x_cls)
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.bbox_head.fc_branch:
            x_fc = self.bbox_head.relu(fc(x_fc))

        with todd.hooks.hook(
            dict(
                type='StandardHook',
                path='.bbox_head.fc_cls._linear',
            ),
            self,
        ) as hook_status:
            self.bbox_head.fc_cls(x_fc)
        return hook_status.value


@mmdet.models.HEADS.register_module()
class ViLDEnsembleRoIHead(mmdet.models.StandardRoIHead):
    bbox_roi_extractor: mmdet.models.BaseRoIExtractor

    def __init__(
        self,
        *args,
        bbox_head: Dict[str, Any],
        image_head: Dict[str, Any],
        patch_head: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        self._image_head: mmdet.models.BBoxHead = mmdet.models.HEADS.build(
            image_head,
            default_args=bbox_head,
        )
        classifier: Classifier = self._image_head.fc_cls
        classifier._bg_embedding.requires_grad_(False)
        ensemble_mask = torch.ones(classifier.num_classes + 1) / 3
        ensemble_mask[:classifier.num_base_classes] *= 2
        self.register_buffer('_ensemble_mask', ensemble_mask, persistent=False)

        if patch_head is not None:
            self._patch_loss = todd.losses.LOSSES.build(
                patch_head.pop('loss'),
            )
            self._patch_head: mmdet.models.BBoxHead = mmdet.models.HEADS.build(
                patch_head,
                default_args=bbox_head,
            )

    @property
    def ensemble_mask(self) -> torch.Tensor:
        return self._ensemble_mask

    @property
    def with_mask(self) -> bool:
        return todd.globals_.training and super().with_mask

    @property
    def with_patch(self) -> bool:
        return hasattr(self, '_patch_head')

    def _bbox_forward(
        self,
        x: List[torch.Tensor],
        rois: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bbox_results = super()._bbox_forward(x, rois)

        if todd.globals_.training:
            return bbox_results

        if 'extra_feats' in todd.globals_:
            bbox_feats = self.bbox_roi_extractor(
                todd.globals_.pop('extra_feats')[:self.bbox_roi_extractor.num_inputs],
                rois,
            )
        else:
            bbox_feats = bbox_results['bbox_feats']

        assert not self.with_shared_head
        cls_score, _ = self._image_head(bbox_feats)
        cls_score = cast(torch.Tensor, cls_score)
        cls_score[:, -1] = float('-inf')

        ensemble_score: torch.Tensor = (
            bbox_results['cls_score'].softmax(-1) ** self.ensemble_mask
            * cls_score.softmax(-1) ** (1 - self.ensemble_mask)
        )
        ensemble_score[:, -1] = 1 - ensemble_score[:, :-1].sum(-1)
        bbox_results['cls_score'] = ensemble_score.log()
        return bbox_results

    def _bbox_forward_distill(
        self,
        x: Sequence[torch.Tensor],
        rois: torch.Tensor,
    ) -> torch.Tensor:
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois,
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        with todd.hooks.hook(
            dict(
                type='StandardHook',
                path='.fc_cls._linear',
            ),
            self._image_head,
        ) as hook_status:
            self._image_head(bbox_feats)
        return hook_status.value

    def _bbox_forward_patch(
        self,
        x: Sequence[torch.Tensor],
        rois: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois,
        )
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        with todd.hooks.hook(
            dict(
                type='StandardHook',
                path='.fc_cls._linear',
            ),
            self._patch_head,
        ) as hook_status:
            logits, _ = self._patch_head(bbox_feats)
        logits = logits[:, :-1]
        patch_loss = self._patch_loss(logits.sigmoid(), labels)
        _, topK_inds = logits.detach().topk(10)
        inds = einops.repeat(torch.arange(labels.shape[0]), 'n -> n k', k=10)
        patch_topK_recall = labels[inds, topK_inds].sum() / labels.sum()
        return hook_status.value, patch_loss, patch_topK_recall

    if debug.DUMP:
        if not os.path.exists(os.getenv('DUMP')):
            os.makedirs(os.getenv('DUMP'))

        def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
            assert len(img_metas) == 1
            with todd.hooks.hook(
                dict(type='StandardHook', path='fc_cls'),
                self.bbox_head,
            ) as bbox_head_status, todd.hooks.hook(
                dict(type='StandardHook', path='fc_cls'),
                self._image_head,
            ) as image_head_status:
                det_bboxes, _ = super().simple_test_bboxes(x, img_metas, proposals, None, rescale)

            patch_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs],
                todd.globals_.pop('clip_rois'),
            )
            if self.with_shared_head:
                patch_feats = self.shared_head(patch_feats)
            patch_logits, _ = self._patch_head(patch_feats)

            image_id = img_metas[0]['ori_filename'][:-4]
            save_path = os.path.join(os.getenv('DUMP'), f'{image_id}.pth')
            torch.save(
                dict(
                    bboxes=det_bboxes[0].half(),
                    objectness=proposals[0][:, -1].half(),
                    bbox_scores=bbox_head_status.value.half(),
                    image_scores=image_head_status.value.half(),
                    multilabel_logits=todd.globals_.pop('multilabel_logits', None),
                    patch_logits=patch_logits,
                ),
                save_path,
            )
            return [torch.tensor([[0, 0, 1, 1]])], [torch.empty([0])]
