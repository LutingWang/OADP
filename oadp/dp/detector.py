import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple, Union

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.detectors.grounding_dino import GroundingDINO


@MODELS.register_module()
class DPGroundingDino(GroundingDINO):
    def __init__(self, *args, bbox_roi_extractor, **kwargs):
        super(DPGroundingDino, self).__init__(*args, **kwargs)
        # self.dp_w = 0.1
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.feature_conv = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=7, 
            stride=7, 
            padding=0
        )

    def rpn_distillation_loss(self, visual_features: Tensor, batch_data_samples: Tensor) -> dict:
        rois, embedings_gt = [], []
        for i, data_samples in enumerate(batch_data_samples):
            # get rois
            tensor = []
            if data_samples.blocks_features is not None:
                block_tensor = data_samples.blocks_features['bboxes'].to_tensor()
                tensor.append(block_tensor)
                embedings_gt.append(data_samples.blocks_features['embeddings'])
            elif data_samples.objects_features is not None:
                object_tensor = data_samples.objects_features['bboxes'].to_tensor()
                tensor.append(object_tensor)
                embedings_gt.append(data_samples.objects_features['tensors'])
            
            if len(tensor) == 0:
                continue
            tensor = torch.cat(tensor, dim=0)
            num, _ = tensor.shape
            index = torch.full((num,), i, dtype=torch.long)
            rois.append(torch.cat([index.unsqueeze(1), tensor], dim=1).to(torch.int32))
        assert len(rois) > 0
        if len(rois) == 0:
            roi = torch.zeros(1, 256, 7, 7, device='cuda')
            roi_features_cov = self.feature_conv(roi).squeeze()
            loss = roi_features_cov.mean() * 0.0
        else:
            rois = torch.cat(rois, dim=0)
            embedings_gt = torch.cat(embedings_gt, dim=0)
            roi_features = self.bbox_roi_extractor(visual_features, rois) # (N, 256, 7, 7)
            roi_features_cov = self.feature_conv(roi_features.cuda()).squeeze() # (N, 512)
            # L1 loss
            L1_loss = nn.L1Loss()
            loss = L1_loss(roi_features_cov, embedings_gt.cuda())
        return {'block_distillation_loss': loss}
    
    def global_distillation_loss(self, visual_features: Tensor, batch_data_samples: Tensor) -> dict:
        pass

    def visual_distillation_loss(self, visual_features: Tensor, batch_data_samples: SampleList) -> dict:
        # rpn distillation loss
        losses = self.rpn_distillation_loss(visual_features, batch_data_samples)
        # global distillation loss
        return losses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)

        visual_features = self.extract_feat(batch_inputs)
        distillation_loss = self.visual_distillation_loss(
            visual_features, batch_data_samples)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        losses.update(distillation_loss)
        return losses
