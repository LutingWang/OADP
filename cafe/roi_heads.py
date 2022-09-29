__all__ = [
    'StandardRoIHead',
]

from typing import Dict, Sequence
import mmdet.models
import torch


@mmdet.models.HEADS.register_module(force=True)
class StandardRoIHead(mmdet.models.StandardRoIHead):

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
