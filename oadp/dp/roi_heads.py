__all__ = [
    'ViLDEnsembleRoIHead',
    'OADPRoIHead',
]

import pathlib
from typing import Any, cast

import mmcv
import todd
import torch
from mmdet.structures.bbox import bbox2roi
from mmdet.models import BaseRoIExtractor, StandardRoIHead
from mmdet.registry import MODELS

from ..base import Globals
from ..base.globals_ import Store
from .bbox_heads import BlockMixin, ObjectMixin


@MODELS.register_module()
class ViLDEnsembleRoIHead(StandardRoIHead):
    bbox_roi_extractor: BaseRoIExtractor

    def __init__(
        self,
        *args,
        bbox_head: todd.Config,
        object_head: todd.Config,
        mask_head: todd.Config | None = None,
        **kwargs,
    ) -> None:
        # automatically detect `num_classes`
        assert bbox_head.num_classes is None
        bbox_head.num_classes = Globals.categories.num_all
        if mask_head is not None:
            assert mask_head.num_classes is None
            mask_head.num_classes = Globals.categories.num_all

        super().__init__(
            *args,
            bbox_head=bbox_head,
            mask_head=mask_head,
            **kwargs,
        )

        # `shared_head` is not supported for simplification
        assert not self.with_shared_head

        self._object_head: ObjectMixin = MODELS.build(
            object_head,
            default_args=bbox_head,
        )

        # :math:`lambda` for base and novel categories are :math:`2 / 3` and
        # :math:`1 / 3`, respectively
        lambda_ = torch.ones(Globals.categories.num_all + 1) / 3
        lambda_[:Globals.categories.num_bases] *= 2
        self.register_buffer('_lambda', lambda_, persistent=False)

    @property
    def lambda_(self) -> torch.Tensor:
        return cast(torch.Tensor, self._lambda)

    def _bbox_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Monkey patching `simple_test_bboxes`.

        Args:
            x: multilevel feature maps.
            rois: regions of interest.

        Returns:
            During training, act the same as `StandardRoIHead`.
            During Inference, replace the classification score with the
            calibrated version.

        The method breaks the `single responsibility principle`, in order for
        monkey patching `mmdet`.
        During training, the method forwards the `bbox_head` and returns the
        `bbox_results` as `StandardRoIHead` does.
        However, during inference, the method forwards both the `bbox_head`
        and the `object_head`.
        The `object_head` classifies each RoI and the predicted logits are
        used to calibrate the outputs of `bbox_head`.

        For more details, refer to ViLD_.

        .. _ViLD: https://readpaper.com/paper/3206072662
        """
        bbox_results: dict[str, torch.Tensor] = super()._bbox_forward(x, rois)
        if Globals.training:
            return bbox_results

        bbox_logits = bbox_results['cls_score']
        bbox_scores = bbox_logits.softmax(-1)**self.lambda_

        object_logits, _ = self._object_head(bbox_results['bbox_feats'])
        object_logits = cast(torch.Tensor, object_logits)
        object_scores = object_logits.softmax(-1)**(1 - self.lambda_)

        if Store.DUMP:
            self._bbox_logits = bbox_logits
            self._object_logits = object_logits

        cls_score = bbox_scores * object_scores
        cls_score[:, -1] = 1 - cls_score[:, :-1].sum(-1)

        bbox_results['cls_score'] = cls_score.log()
        return bbox_results

    def _object_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> None:
        bre = self.bbox_roi_extractor
        object_feats = bre(x[:bre.num_inputs], rois)
        self._object_head(object_feats)

    def object_forward(
        self,
        x: list[torch.Tensor],
        bboxes: list[torch.Tensor],
    ) -> None:
        rois = bbox2roi(bboxes)
        self._object_forward(x, rois)
    #TODO: Algin to MMdet3
    if Store.DUMP:
        access_layer = todd.datasets.PthAccessLayer(
            data_root=Store.DUMP,
            readonly=False,
        )

        def simple_test_bboxes(
            self,
            x: torch.Tensor,
            img_metas: list[dict[str, Any]],
            proposals: list[torch.Tensor],
            rcnn_test_cfg: mmcv.ConfigDict,
            rescale: bool = False,
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            assert x.shape[0] == len(img_metas) == len(proposals) == 1
            filename = pathlib.Path(img_metas[0]['filename']).stem
            objectness = proposals[0][:, -1]

            bboxes, _ = super().simple_test_bboxes(
                x,
                img_metas,
                proposals,
                None,
                rescale,
            )

            record = dict(
                bboxes=bboxes,
                bbox_logits=self._bbox_logits,
                object_logits=self._object_logits,
                objectness=objectness,
            )
            record = {k: v.half() for k, v in record.items()}
            self.access_layer[filename] = record

            return [torch.tensor([[0, 0, 1, 1]])], [torch.empty([0])]


@MODELS.register_module()
class OADPRoIHead(ViLDEnsembleRoIHead):

    def __init__(
        self,
        *args,
        bbox_head: todd.Config,
        block_head: todd.Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        if block_head is not None:
            self._block_head: BlockMixin = MODELS.build(
                block_head,
                default_args=bbox_head,
            )

    @property
    def with_block(self) -> bool:
        return hasattr(self, '_block_head')

    def _block_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> torch.Tensor:
        bre = self.bbox_roi_extractor
        block_feats = bre(x[:bre.num_inputs], rois)
        logits, _ = self._block_head(block_feats)
        return logits

    def block_forward(
        self,
        x: list[torch.Tensor],
        bboxes: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        rois = bbox2roi(bboxes)
        logits = self._block_forward(x, rois)
        losses = self._block_head.loss(logits[:, :-1], torch.cat(targets))
        return losses
