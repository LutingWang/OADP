__all__ = [
    'GlobalHead',
    'OADP',
]

import copy
from typing import Any, Sequence

import einops
import todd
import todd.tasks.knowledge_distillation as kd
import torch
from mmdet.models import RPNHead, TwoStageDetector
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from todd.models import LossRegistry
from torch import nn

from ..utils import Globals
from .roi_heads import OADPRoIHead
# from .utils import MultilabelTopKRecall

SelfDistiller = kd.distillers.SelfDistiller
StudentMixin = kd.distillers.StudentMixin


class GlobalHead(nn.Module):

    def __init__(
        self,
        *args,
        topk: int,
        classifier: todd.Config,
        loss: todd.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # self._multilabel_topk_recall = MultilabelTopKRecall(k=topk)
        self._classifier = MODELS.build(classifier)
        self._loss = LossRegistry.build(loss)

    def _forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        return self._classifier(feat)

    def forward(
        self,
        *args,
        labels: list[torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        logits = self._forward(*args, **kwargs)
        targets = logits.new_zeros(
            logits.shape[0],
            Globals.categories.num_all,
            dtype=torch.bool,
        )
        for i, label in enumerate(labels):
            targets[i, label] = True
        return dict(
            loss_global=self._loss(logits.sigmoid(), targets),
            # recall_global=self._multilabel_topk_recall(logits, targets),
        )


@MODELS.register_module()
class OADP(StudentMixin[SelfDistiller], TwoStageDetector):
    rpn_head: RPNHead
    roi_head: OADPRoIHead

    @kd.distillers.distiller_decorator
    def __init__(
        self,
        *args,
        global_head: todd.Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if global_head is not None:
            self._global_head = GlobalHead(**global_head)

    @property
    def num_classes(self) -> int:
        return Globals.categories.num_all

    @property
    def with_global_head(self) -> bool:
        return hasattr(self, '_global_head')

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: OptSampleList = None,
        mode: str = 'tensor',
        **kwargs,
    ):
        if mode == 'predict':
            Globals.training = False
            return self.predict(inputs, data_samples)

        Globals.training = True
        feats = self.extract_feat(inputs)
        losses: dict[str, Any] = dict()
        custom_tensors: dict[str, Any] = dict()

        self._forward(
            feats,
            data_samples,
            losses,
            custom_tensors,
            **kwargs,
        )

        distill_losses = self.distiller(custom_tensors)
        self.distiller.reset()
        self.distiller.step()
        losses.update(distill_losses)

        return losses

    def _forward(
        self,
        feats: list[torch.Tensor],
        data_samples: OptSampleList,
        losses: dict[str, Any],
        custom_tensors: dict[str, Any],
        *,
        clip_global: list[torch.Tensor],
        clip_blocks: list[torch.Tensor],
        block_bboxes: list[torch.Tensor],
        block_labels: list[torch.Tensor],
        clip_objects: list[torch.Tensor],
        object_bboxes: list[torch.Tensor],
        **kwargs,
    ) -> None:
        # RPN forward and loss
        proposal_cfg = self.train_cfg.rpn_proposal
        rpn_data_samples = copy.deepcopy(data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            feats, rpn_data_samples, proposal_cfg=proposal_cfg
        )
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        # ROI loss
        roi_losses = self.roi_head.loss(feats, rpn_results_list, data_samples)
        losses.update(roi_losses)

        self.roi_head.object_forward(feats, object_bboxes)
        custom_tensors['clip_objects'] = torch.cat(clip_objects).float().cuda()

        if self.with_global_head:
            global_losses = self._global_head.forward(
                feats,
                labels=[sample.gt_instances.labels for sample in data_samples],
            )
            losses.update(global_losses)
            custom_tensors['clip_global'] = torch.stack(clip_global,
                                                        dim=0).float().cuda()

        if self.roi_head.with_block:
            block_labels = [label.cuda() for label in block_labels]
            block_losses = self.roi_head.block_forward(
                feats,
                block_bboxes,
                block_labels,
            )
            losses.update(block_losses)
            custom_tensors['clip_blocks'] = torch.cat(clip_blocks
                                                      ).float().cuda()
