__all__ = [
    'ViLDEnsembleRoIHead',
    'OADPRoIHead',
]

from typing import cast

import todd
import torch
import numpy as np
import torch.nn as nn
from einops import repeat
from mmdet.models import BaseRoIExtractor, StandardRoIHead
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from mmdet.models.utils import empty_instances

from ..utils import Globals
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


@MODELS.register_module()
class RAMModel(nn.Module):
    def __init__(
        self, 
        ram_pred: str,
    ) -> None:
        super().__init__()
        self.ram_pred_result = torch.load(ram_pred)

    @torch.no_grad()
    def forward(
        self, 
        cls_score: torch.Tensor,
        batch_img_metas: list[dict]
    ) -> torch.Tensor:
        ram_cls_score = []
        for img_meta in batch_img_metas: # bs = 1
            img_path = img_meta['img_path']
            if img_path not in self.ram_pred_result:
                print(f"Image path {img_path} not found in RAM prediction results")
                ram_cls_score.append(np.ones((cls_score.shape[1] - 1, )))
                # raise ValueError(f"Image path {img_path} not found in RAM prediction results")
            else:
                ram_cls_score.append(self.ram_pred_result[img_path])
        ram_cls_score = np.concatenate(ram_cls_score, axis=0)
        ram_cls_score = torch.from_numpy(ram_cls_score) # (1, num_classes)
        
        num_box, _ = cls_score.shape
        background_score = torch.ones((num_box, 1))
        ram_cls_score = repeat(ram_cls_score, 'c -> b c', b=num_box).sigmoid()
        ram_cls_score_with_bg = torch.cat([ram_cls_score, background_score], dim=1).to(cls_score.device)

        return (cls_score.softmax(-1) * ram_cls_score_with_bg).log()

    
@MODELS.register_module()
class RAMEnsembleOADPRoIHead(OADPRoIHead):
    def __init__(self, *args,  classifier_model: todd.Config, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_model = MODELS.build(classifier_model)

    def predict_bbox(self,
                     x: tuple[torch.Tensor],
                     batch_img_metas: list[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois, batch_img_metas)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list

    def _bbox_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
        batch_img_metas: list[dict],
    ) -> dict[str, torch.Tensor]:
        bbox_results = super()._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_results['cls_score'] = self.classifier_model(cls_score, batch_img_metas)
        return bbox_results