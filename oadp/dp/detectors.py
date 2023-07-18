__all__ = [
    'GlobalHead',
    'OADP',
]

from typing import Any

import einops
import torch
from mmdet.core import BitmapMasks
from mmdet.models import DETECTORS, RPNHead, TwoStageDetector
from mmdet.models.utils.builder import LINEAR_LAYERS

import todd
from todd.distillers import SelfDistiller, Student
from todd.losses import LossRegistry as LR

from ..base import Globals
from .roi_heads import OADPRoIHead
from .utils import MultilabelTopKRecall


class GlobalHead(todd.Module):

    def __init__(
        self,
        *args,
        topk: int,
        classifier: todd.Config,
        loss: todd.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._multilabel_topk_recall = MultilabelTopKRecall(k=topk)
        self._classifier = LINEAR_LAYERS.build(classifier)
        self._loss = LR.build(loss)

    def forward(self, feats: tuple[torch.Tensor, ...]) -> torch.Tensor:
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        return self._classifier(feat)

    def forward_train(
        self,
        *args,
        labels: list[torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        logits = self.forward(*args, **kwargs)
        targets = logits.new_zeros(
            logits.shape[0],
            Globals.categories.num_all,
            dtype=torch.bool,
        )
        for i, label in enumerate(labels):
            targets[i, label] = True
        return dict(
            loss_global=self._loss(logits.sigmoid(), targets),
            recall_global=self._multilabel_topk_recall(logits, targets),
        )


@DETECTORS.register_module()
class OADP(TwoStageDetector, Student[SelfDistiller]):
    rpn_head: RPNHead
    roi_head: OADPRoIHead

    def __init__(
        self,
        *args,
        global_head: todd.Config | None = None,
        distiller: todd.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if global_head is not None:
            self._global_head = GlobalHead(**global_head)
        distiller.setdefault('type', 'SelfDistiller')
        Student.__init__(self, distiller)

    @property
    def num_classes(self) -> int:
        return Globals.categories.num_all

    @property
    def with_global(self) -> bool:
        return hasattr(self, '_global_head')

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]],
        gt_bboxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        clip_global: torch.Tensor,
        clip_blocks: list[torch.Tensor],
        block_bboxes: list[torch.Tensor],
        block_labels: list[torch.Tensor],
        clip_objects: list[torch.Tensor],
        object_bboxes: list[torch.Tensor],
        gt_masks: list[BitmapMasks] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        Globals.training = True
        feats = self.extract_feat(img)
        losses = dict()
        custom_tensors = dict()

        if self.with_global:
            global_losses = self._global_head.forward_train(
                feats,
                labels=gt_labels,
            )
            losses.update(global_losses)
            custom_tensors['clip_global'] = clip_global.float()

        rpn_losses, proposals = self.rpn_head.forward_train(
            feats,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=None,
            proposal_cfg=self.train_cfg.rpn_proposal,
            **kwargs,
        )
        losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(
            feats,
            img_metas,
            proposals,
            gt_bboxes,
            gt_labels,
            None,
            gt_masks,
            **kwargs,
        )
        losses.update(roi_losses)

        if self.roi_head.with_block:
            block_losses = self.roi_head.block_forward_train(
                feats,
                block_bboxes,
                block_labels,
            )
            losses.update(block_losses)
            custom_tensors['clip_blocks'] = torch.cat(clip_blocks).float()

        self.roi_head.object_forward_train(feats, object_bboxes)
        custom_tensors['clip_objects'] = torch.cat(clip_objects).float()

        distill_losses = self.distiller(custom_tensors)
        self.distiller.reset()
        self.distiller.step()
        losses.update(distill_losses)

        return losses

    def simple_test(self, *args, **kwargs):
        Globals.training = False
        return super().simple_test(*args, **kwargs)
