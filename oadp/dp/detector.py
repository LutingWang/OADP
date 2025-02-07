import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Union

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.models.detectors.grounding_dino import GroundingDINO

from loralib import mark_only_lora_as_trainable
from .lora import replace_linear_with_lora
from .moe import replace_linear_with_moe, freeze_module, mark_only_moe_as_trainable, extract_moe_aux_loss

@MODELS.register_module()
class DPGroundingDino(GroundingDINO):
    def __init__(self, *args, 
                bbox_roi_extractor,
                moe_cfg=None,
                use_lora=False,
                distill_visual_encoder=False,
                distill_dino_encoder=False,
                **kwargs):
        super(DPGroundingDino, self).__init__(*args, **kwargs)
        self.dp_w = 50
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        
        self.encoder_outputs_dict = None
        self.distill_visual_encoder = distill_visual_encoder
        self.distill_dino_encoder = distill_dino_encoder
        self.use_moe = moe_cfg is not None
        freeze_module(self)

        if use_lora:
            self.encoder = replace_linear_with_lora(self.encoder, alpha=128, rank=64, blacklist=['out_proj'])
            mark_only_lora_as_trainable(self)

        if moe_cfg:
            replace_linear_with_moe(self.encoder, 
                                    in_features=moe_cfg['inputs_dim'], 
                                    linear_name_pattern='ffn', 
                                    num_experts=moe_cfg['expert_num'], 
                                    topk=moe_cfg['topk'])
            mark_only_moe_as_trainable(self.encoder)

        # add distillation head
        if self.distill_visual_encoder or self.distill_dino_encoder:
            self.visual_feature_dim = None
            self.feature_conv = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
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
            if self.distill_dino_encoder:
                all_dim = [x*y for x, y in self.visual_feature_dim]
                permute_feat = visual_features.permute(0, 2, 1)
                bs, c, _ = permute_feat.shape
                split_tensors = torch.split(permute_feat, all_dim, dim=2)
                visual_features = [tensor.view(bs, c, x, y) for tensor, 
                                    (x, y) in zip(split_tensors, self.visual_feature_dim)]
            rois = torch.cat(rois, dim=0)
            embedings_gt = torch.cat(embedings_gt, dim=0)
            roi_features = self.bbox_roi_extractor(visual_features, rois) # (N, 256, 7, 7)
            roi_features_cov = self.feature_conv(roi_features.cuda()).squeeze() # (N, 512)
            # L1 loss
            L1_loss = nn.L1Loss()
            loss = L1_loss(roi_features_cov, embedings_gt.cuda())
        return {'block_distillation_loss': self.dp_w * loss}
    
    def global_distillation_loss(self, visual_features: Tensor, batch_data_samples: Tensor) -> dict:
        pass

    def visual_distillation_loss(self, visual_features: Tensor, batch_data_samples: SampleList) -> dict:
        # rpn distillation loss
        losses = self.rpn_distillation_loss(visual_features, batch_data_samples)
        # global distillation loss
        return losses

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        self.encoder_outputs_dict = encoder_outputs_dict
        return encoder_outputs_dict

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
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.distill_visual_encoder:
            distillation_loss = self.visual_distillation_loss(
                visual_features, batch_data_samples)
            losses.update(distillation_loss)
        elif self.distill_dino_encoder:
            self.visual_feature_dim = [list(feat.shape)[-2:] for feat in visual_features]
            distillation_loss = self.visual_distillation_loss(
                self.encoder_outputs_dict['memory'], batch_data_samples)
            losses.update(distillation_loss)
        if self.use_moe:
            aux_loss = extract_moe_aux_loss(self.encoder)
            losses.update(aux_loss)
        return losses