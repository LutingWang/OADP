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

        # if hasattr(self, 'message'):
        #     cls_score = results['cls_score']
        #     assert rois[:, 0].eq(0).all()
        #     logits, topK_inds = getattr(self, 'message')
        #     assert logits.shape[0] == topK_inds.shape[0] == 1
        #     foreground = logits.argmax(-1) < logits.shape[1] - 1
        #     cls_score[foreground, :-1] += logits * 0.1
        #     # cls_score[:, topK_inds[0]] = float('-inf')

        return results
