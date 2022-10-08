__all__ = [
    'MessageMixin',
    'StandardRoIHead',
    'DoubleHeadRoIHead',
]

from typing import Dict, Sequence
import mmdet.models
import torch
import todd


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
