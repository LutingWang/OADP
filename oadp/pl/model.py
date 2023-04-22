import torch
from mmcv.utils import Registry
from mmdet.core import multiclass_nms
from torch import nn

from .datasets import Batch

PLMODELS = Registry('PLModels')


@PLMODELS.register_module()
class VLIDModel(nn.Module):

    def __init__(
        self,
        softmax_t=None,
        topK_clip_scores=None,
        nms_score_thres=None,
        nms_iou_thres=None,
        bbox_objectness=None,
        dist=True
    ) -> None:
        super().__init__()
        self.dist = dist
        self.softmax_t = softmax_t
        self.topK_clip_scores = topK_clip_scores
        self.nms_score_thres = nms_score_thres
        self.nms_iou_thres = nms_iou_thres
        self.score_fusion_cfg = bbox_objectness

    def forward(self, batch: Batch) -> torch.Tensor:
        proposal_embeddings = batch.proposal_embeddings
        proposal_objectness = batch.proposal_objectness
        assert proposal_embeddings.shape[1] == proposal_objectness.shape[1]
        proposal_embeddings = proposal_embeddings / \
            proposal_embeddings.norm(dim=2, keepdim=True)

        clip_logit = (proposal_embeddings @ batch.class_embeddings.T)
        clip_logit = (1 / self.softmax_t) * clip_logit
        clip_logit = torch.softmax(clip_logit, dim=2)
        clip_logit_v, _ = torch.topk(
            clip_logit, self.topK_clip_scores, dim=2
        )

        clip_logit_k = clip_logit * (clip_logit >= clip_logit_v[..., -1:])

        # fusion
        if self.score_fusion_cfg['_name'] == 'add':
            final_logit_k = (
                clip_logit_k * self.score_fusion_cfg['clip_score_ratio']
            ) + ((clip_logit_k > 0) * batch.proposal_objectness[..., None]
                 * self.score_fusion_cfg['obj_score_ratio'])
        elif self.score_fusion_cfg['_name'] == 'mul':
            final_logit_k = (
                clip_logit_k**self.score_fusion_cfg['clip_score_ratio']
            ) * (
                batch.proposal_objectness **
                self.score_fusion_cfg['obj_score_ratio']
            )
        else:
            raise ValueError(self.score_fusion_cfg['_name'])

        # split batch to each image to nms/thresh
        final_bboxes = []
        final_labels = []
        final_image = []
        for i, (result, logit) in enumerate(
            zip(batch.proposal_bboxes, final_logit_k)
        ):
            final_bbox_c, final_label = multiclass_nms(
                result[..., :4].float(),
                logit.float(),
                score_thr=self.nms_score_thres,
                nms_cfg=dict(type='nms', iou_threshold=self.nms_iou_thres)
            )
            image_ids = batch.image_ids[i].repeat(final_bbox_c.shape[0])
            final_bboxes.append(final_bbox_c)
            final_labels.append(final_label)
            final_image.append(image_ids)
        final_bboxes = torch.cat(final_bboxes)
        final_labels = torch.cat(final_labels)
        final_image = torch.cat(final_image)
        return final_bboxes, final_labels, final_image


@PLMODELS.register_module()
class MLCOCOModel(nn.Module):

    def __init__(
        self,
        softmax_t=None,
        topK_clip_scores=None,
        nms_score_thres=None,
        nms_iou_thres=None,
        bbox_objectness=None,
        dist=True
    ) -> None:
        super().__init__()
        self.dist = dist
        self.softmax_t = softmax_t
        self.topK_clip_scores = topK_clip_scores
        self.nms_score_thres = nms_score_thres
        self.nms_iou_thres = nms_iou_thres
        self.score_fusion_cfg = bbox_objectness

    def forward(self, batch: Batch) -> torch.Tensor:
        proposal_embeddings = batch.proposal_embeddings
        proposal_objectness = batch.proposal_objectness
        assert proposal_embeddings.shape[1] == proposal_objectness.shape[1]
        proposal_embeddings = proposal_embeddings / \
            proposal_embeddings.norm(dim=2, keepdim=True)

        clip_logit = (
            proposal_embeddings @ batch.class_embeddings.T
        ) * batch.scaler - batch.bias
        clip_logit = torch.softmax(clip_logit, dim=2)
        clip_logit_v, _ = torch.topk(
            clip_logit, self.topK_clip_scores, dim=2
        )
        clip_logit_k = clip_logit * (clip_logit >= clip_logit_v[..., -1:])

        # fusion
        if self.score_fusion_cfg['_name'] == 'add':
            final_logit_k = (
                clip_logit_k * self.score_fusion_cfg['clip_score_ratio']
            ) + ((clip_logit_k > 0) * batch.proposal_objectness[..., None]
                 * self.score_fusion_cfg['obj_score_ratio'])
        elif self.score_fusion_cfg['_name'] == 'mul':
            final_logit_k = (
                clip_logit_k**self.score_fusion_cfg['clip_score_ratio']
            ) * (
                batch.proposal_objectness **
                self.score_fusion_cfg['obj_score_ratio']
            )
        else:
            raise ValueError(self.score_fusion_cfg['_name'])

        final_bboxes = []
        final_labels = []
        final_image = []
        for i, (result, logit) in enumerate(
            zip(batch.proposal_bboxes, final_logit_k)
        ):
            final_bbox_c, final_label = multiclass_nms(
                result[..., :4].float(),
                logit.float(),
                score_thr=self.nms_score_thres,
                nms_cfg=dict(type='nms', iou_threshold=self.nms_iou_thres)
            )
            image_ids = batch.image_ids[i].repeat(final_bbox_c.shape[0])
            final_bboxes.append(final_bbox_c)
            final_labels.append(final_label)
            final_image.append(image_ids)
        final_bboxes = torch.cat(final_bboxes)
        final_labels = torch.cat(final_labels)
        final_image = torch.cat(final_image)
        return final_bboxes, final_labels, final_image


def build_model(config):
    return PLMODELS.build(config)
