import copy
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from mmdet.structures import SampleList
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS

from .fs_model import FewShotModel


@MODELS.register_module()
class FsGroundingDINO(GroundingDINO):
    def __init__(self, fs_model_cfg, use_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_model = None
        self.fs_model: FewShotModel = MODELS.build(fs_model_cfg)
        self.fs_model_mode = "features" if use_features else "images"
        # freeze fs_model language model and clip model
        for param in self.fs_model.language_model.parameters():
            param.requires_grad = False
        for param in self.fs_model.clip_model.parameters():
            param.requires_grad = False

    def extract_fs_features(self, batch_data_samples: SampleList, device: torch.device):
        # Extract features from the few-shot model
        ref_images_cat = []
        shots = []
        texts = []
        for data_sample in batch_data_samples:
            ref_images_cat.extend(data_sample.ref_images)
            shots.extend([data_sample.n_images] * data_sample.n_samples)
            texts.extend(data_sample.text)
        ref_images_cat = torch.stack(ref_images_cat, dim=0).to(device)
        align_loss, image_feats = self.fs_model(ref_images_cat, texts, shots, self.fs_model_mode)
        image_feats = self.text_feat_map(image_feats) # [bs * n_samples, visual_dim]

        # align output with text model and padding to num_classes
        n_samples = batch_data_samples[0].n_samples
        bs = len(batch_data_samples)
        num_classes = self.bbox_head.num_classes
        embedded = rearrange(image_feats, '(b n) s -> b n s', n=n_samples)
        # when trainning the model, the number of classes is less than the number of samples
        if embedded.shape[1] <= num_classes:
            pad = torch.zeros(embedded.shape[0], num_classes - embedded.shape[1], embedded.shape[2],
                              device=embedded.device, dtype=embedded.dtype)
            embedded = torch.cat([embedded, pad], dim=1)
            # create masks and position_ids
            masks = torch.eye(num_classes, dtype=torch.bool).repeat(bs, 1, 1).to(device)
            text_token_mask = torch.ones(bs, num_classes, dtype=torch.bool).to(device)
            text_token_mask[:, n_samples:] = False # padding mask
            position_ids = torch.zeros(num_classes).repeat(bs, 1).to(device)
            if self.training:
                return {
                    'embedded': embedded, # [bs, num_classes, visual_dim]
                    'masks': masks, # [bs, num_classes, num_classes]
                    'position_ids': position_ids, # [bs, num_classes]
                    'text_token_mask': text_token_mask, # [bs, num_classes]
                }, align_loss
            else:
                return [{
                    'embedded': embedded, # [bs, num_classes, visual_dim]
                    'masks': masks, # [bs, num_classes, num_classes]
                    'position_ids': position_ids, # [bs, num_classes]
                    'text_token_mask': text_token_mask, # [bs, num_classes]
                    'token_positive_map': {j+1: [j] for j in range(n_samples)},
                }]
        else: # when testing the number of classes is greater than the number of samples
            # Calculate number of chunks needed by splitting the second dimension into pieces of size num_classes
            num_chunks = (embedded.shape[1] + num_classes - 1) // num_classes
            result = []
            for i in range(num_chunks):
                start = i * num_classes
                end = (i + 1) * num_classes
                chunk = embedded[:, start:end, :]
                bs, cur_len, feat_dim = chunk.shape

                # If the current chunk is smaller than num_classes, pad it
                if cur_len < num_classes:
                    pad = torch.zeros(bs, num_classes - cur_len, feat_dim,
                                      device=chunk.device, dtype=chunk.dtype)
                    chunk = torch.cat([chunk, pad], dim=1)
                    token_mask = torch.ones(bs, num_classes, dtype=torch.bool, device=chunk.device)
                    token_mask[:, cur_len:] = False
                    token_positive_map = {j+1: [j] for j in range(cur_len)}
                else:
                    token_mask = torch.ones(bs, num_classes, dtype=torch.bool, device=chunk.device)
                    token_positive_map = {j+1: [j] for j in range(num_classes)}

                masks = torch.eye(num_classes, dtype=torch.bool, device=chunk.device).unsqueeze(0).repeat(bs, 1, 1)
                position_ids = torch.zeros(bs, num_classes, device=chunk.device)

                result.append({
                    'embedded': chunk,
                    'masks': masks,
                    'position_ids': position_ids,
                    'text_token_mask': token_mask,
                    'token_positive_map': token_positive_map,
                })

                return result

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        ref_dict, align_loss = self.extract_fs_features(batch_data_samples, batch_inputs.device)
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = data_samples.tokens_positive
            # padding to num_classes
            num_classes = self.bbox_head.num_classes
            if positive_map.shape[0] < num_classes:
                pad = torch.zeros(positive_map.shape[0], num_classes - positive_map.shape[1],
                              device=positive_map.device, dtype=positive_map.dtype)
                positive_map = torch.cat([positive_map, pad], dim=1)
            # create masks and position_ids
            data_samples.gt_instances.positive_maps = positive_map.to(batch_inputs.device) # [num_instance, num_classes]
            text_token_mask = ref_dict['text_token_mask'][i]
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        
        visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, ref_dict,
                                                  batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        chuncked_ref_dict = self.extract_fs_features(batch_data_samples, batch_inputs.device)
        visual_features = self.extract_feat(batch_inputs)
        # predict the instances for each chunk
        count = 0
        results_list = []
        for ref_dict in chuncked_ref_dict:
            token_positive_maps_once = ref_dict.pop('token_positive_map')
            batch_data_samples[0].token_positive_map = token_positive_maps_once
            head_inputs_dict = self.forward_transformer(copy.deepcopy(visual_features), 
                                                        ref_dict, batch_data_samples)
            pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]
            if len(pred_instances) > 0: # fix the labels
                pred_instances.labels += count
            count += len(token_positive_maps_once)
            results_list.append(pred_instances)
        results_list = [results_list[0].cat(results_list)]
        
        # assign the predicted instances to the data samples
        for data_sample, pred_instances in zip(batch_data_samples, results_list):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                        label_names.append(data_sample.text[labels])
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples