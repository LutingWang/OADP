from typing import Union
import os
import torch
from torch import Tensor
import torch.nn.functional as F

from mmdet.models.detectors.base import SampleList
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS
from mmdet.models.language_models.bert import BertModel
from transformers import BertModel as HFBertModel

from dp_detector.projector import VisionProjector


@MODELS.register_module()
class FSDetectorFeature(GroundingDINO):
    def __init__(self, *args, connector_cfg: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_model: BertModel
        bert_model: HFBertModel = self.language_model.language_backbone.body.model
        self.language_model.register_forward_pre_hook(self.language_model_forward_hook)
        bert_model.embeddings.register_forward_hook(
            self.embeddings_forward_hook, with_kwargs=True
        )

        # freeze all the layers
        for param in self.parameters():
            param.requires_grad = False

        self.connector: VisionProjector = MODELS.build(connector_cfg)

    def language_model_forward_hook(self, module, input):
        self.new_text_prompts = input[0]  # input is a tuple, get first element
        if not self.training:
            self.qwen_feature_lists = []
            for qwen_feature_list in self.chunked_qwen_feature_lists:
                self.qwen_feature_lists.append(
                    qwen_feature_list[self.current_chunk_idx]
                )
            self.current_chunk_idx += 1

    def embeddings_forward_hook(self, module, args, kwargs, output):
        tokenized = self.language_model.tokenizer.batch_encode_plus(
            self.new_text_prompts,
            max_length=self.language_model.max_tokens,
            padding="max_length" if self.language_model.pad_to_max else "longest",
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids: Tensor = tokenized.input_ids

        batch_size = input_ids.shape[0]
        for batch_idx in range(batch_size):
            # 获取所有special tokens的位置并排序
            special_positions = torch.cat(
                [
                    torch.where(input_ids[batch_idx] == token)[0]
                    for token in self.language_model.special_tokens
                ]
            ).sort()[0]

            qwen_features = self.qwen_feature_lists[batch_idx]

            # 确保special tokens数量比qwen特征数量多1或2
            assert (
                len(special_positions) == len(qwen_features) + 2
                or len(special_positions) == len(qwen_features) + 1
            ), f"Special tokens count ({len(special_positions)}) should be equal to qwen features count + 2 ({len(qwen_features) + 1}) or qwen features count + 1 ({len(qwen_features)})"

            # 对每个区间进行替换
            for i, qwen_feat in enumerate(qwen_features):
                start_pos = special_positions[i] + 1
                end_pos = special_positions[i + 1]
                qwen_feature = self.connector(qwen_feat.to(output.device))
                assert (
                    qwen_feature.shape[0] == end_pos - start_pos
                ), f"Qwen feature length ({qwen_feature.shape[0]}) should be equal to segment length ({end_pos - start_pos})\
                    special_positions {special_positions}, input_ids[batch_idx] {input_ids[batch_idx]}"
                output[batch_idx, start_pos:end_pos] = qwen_feature

        return output

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        self.qwen_feature_lists = []
        for data_sample in batch_data_samples:
            self.qwen_feature_lists.append(data_sample.qwen_feature_list)
        return super().loss(batch_inputs, batch_data_samples)

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        chunked_size = self.test_cfg.get("chunked_size", -1)
        if chunked_size > 0:
            self.current_chunk_idx = 0
            # get chunked qwen feature lists [bs, chunk_size, *]
            self.chunked_qwen_feature_lists = []
            for data_sample in batch_data_samples:
                qwen_feature_list = data_sample.qwen_feature_list
                chunked_qwen_features = []
                for i in range(0, len(qwen_feature_list), chunked_size):
                    chunked_qwen_features.append(
                        qwen_feature_list[i : i + chunked_size]
                    )
                self.chunked_qwen_feature_lists.append(chunked_qwen_features)
        else:
            self.chunked_qwen_feature_lists = self.qwen_feature_lists
        return super().predict(batch_inputs, batch_data_samples, rescale)


@MODELS.register_module()
class FSDetectorFeatureDistill(FSDetectorFeature):
    def __init__(self, *args, w_distill=0.05, w_global=0.8, w_structure=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_model.register_forward_hook(self.language_hook, with_kwargs=True)
        self.w_distill = w_distill
        self.w_global = w_global  # weight for global MSE loss
        self.w_structure = w_structure  # weight for structure MSE loss

    def language_model_forward_hook(self, module, input):
        if not self.use_real_texts:
            return super().language_model_forward_hook(module, input)
        else:
            return input

    def language_hook(self, module, args, kwargs, output):
        if not self.use_real_texts:
            self.pusedo_text_embeds = output["embedded"]
        else:
            return output

    def embeddings_forward_hook(self, module, args, kwargs, output):
        if not self.use_real_texts:
            return super().embeddings_forward_hook(module, args, kwargs, output)
        else:
            return output

    def compute_distill_loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        # real text forward bert model
        self.real_name_list = []
        for data_sample in batch_data_samples:
            name_cat = ". ".join(data_sample.text_list)
            self.real_name_list.append(name_cat)
        self.use_real_texts = True
        self.language_model.pad_to_max = False
        text_dict = self.language_model(self.real_name_list)
        self.language_model.pad_to_max = True

        # get tokenized information
        tokenized = self.language_model.tokenizer.batch_encode_plus(
            self.real_name_list,
            max_length=self.language_model.max_tokens,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = tokenized.input_ids.to(text_dict["embedded"].device)

        # Create special tokens mask
        special_tokens = self.language_model.special_tokens
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_tokens:
            special_tokens_mask |= input_ids == token_id

        # Get real text embeddings and pseudo text embeddings
        real_text_embeds = text_dict["embedded"]  # [bs, seq_len, dim]
        pseudo_text_embeds = self.pusedo_text_embeds  # [bs, seq_len, dim]

        # Extract mean embeddings for each segment between special tokens
        all_real_segments = []
        all_pseudo_segments = []
        segment_batch_idx = []  # to track which batch example each segment comes from

        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            # Find indices where special tokens occur
            special_token_indices = torch.where(special_tokens_mask[i])[0]

            # Extract embeddings between each pair of special tokens
            for j in range(len(special_token_indices) - 1):
                start_idx = special_token_indices[j] + 1
                end_idx = special_token_indices[j + 1]

                # Only process if there are tokens between special tokens
                if end_idx > start_idx:
                    # Get mean embedding for tokens between special tokens
                    real_segment = real_text_embeds[i, start_idx:end_idx].mean(dim=0)
                    pseudo_segment = pseudo_text_embeds[i, start_idx:end_idx].mean(
                        dim=0
                    )

                    all_real_segments.append(real_segment)
                    all_pseudo_segments.append(pseudo_segment)
                    segment_batch_idx.append(i)

        # Skip if no valid segments
        if not all_real_segments:
            return {"loss_distill": torch.tensor(0.0, device=real_text_embeds.device)}

        # Stack all segments
        all_real_segments = torch.stack(all_real_segments)  # [total_segments, dim]
        all_pseudo_segments = torch.stack(all_pseudo_segments)  # [total_segments, dim]

        # 1. Global MSE Loss - compute mean of all segments and calculate MSE
        global_real = all_real_segments.mean(dim=0)
        global_pseudo = all_pseudo_segments.mean(dim=0)
        loss_global = F.mse_loss(global_pseudo, global_real)

        # 2. Structure MSE Loss - compute similarity matrices and calculate MSE
        # Normalize embeddings before computing similarity matrices
        real_norm = F.normalize(all_real_segments, p=2, dim=-1)
        pseudo_norm = F.normalize(all_pseudo_segments, p=2, dim=-1)

        # Compute normalized similarity matrices (cosine similarity)
        real_sim_matrix = torch.matmul(real_norm, real_norm.T)
        pseudo_sim_matrix = torch.matmul(pseudo_norm, pseudo_norm.T)
        loss_structure = F.mse_loss(pseudo_sim_matrix, real_sim_matrix)

        # 3. Contrastive Loss (original)
        temperature = 0.07  # temperature parameter for scaling

        # Reuse the normalized embeddings from above
        # Compute similarity matrix
        logits = torch.matmul(pseudo_norm, real_norm.T) / temperature

        # Labels are the diagonal indices (matching pairs)
        labels = torch.arange(len(all_real_segments), device=real_norm.device)

        # Cross-entropy loss in both directions
        loss_p2r = F.cross_entropy(logits, labels)
        loss_r2p = F.cross_entropy(logits.T, labels)

        loss_contrastive = (loss_p2r + loss_r2p) / 2.0

        return {
            "loss_contrastive": loss_contrastive * self.w_distill,
            "loss_global": loss_global * self.w_global,
            "loss_structure": loss_structure * self.w_structure,
        }

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        self.qwen_feature_lists = []
        for data_sample in batch_data_samples:
            self.qwen_feature_lists.append(data_sample.qwen_feature_list)
        self.use_real_texts = False
        loss = super().loss(batch_inputs, batch_data_samples)
        distill_loss = self.compute_distill_loss(batch_inputs, batch_data_samples)
        loss.update(distill_loss)
        return loss

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        self.use_real_texts = False
        preds = super().predict(batch_inputs, batch_data_samples, rescale)
        pred_instances = preds[0].pred_instances
        img_id = batch_data_samples[0].img_id
        path = "data/image_val_results"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save({
            "bboxes": pred_instances.bboxes.cpu().numpy(),
            "labels": pred_instances.labels.cpu().numpy(),
            "scores": pred_instances.scores.cpu().numpy(),
        }, f"{path}/pred_instances_{img_id}.pth")
        return preds
