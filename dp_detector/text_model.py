from typing import Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.detectors.base import SampleList
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.test_time_augs import DetTTAModel
from mmdet.registry import MODELS


@MODELS.register_module()
class TextDetectorDistill(GroundingDINO):
    def __init__(
        self,
        *args,
        bbox_roi_extractor,
        obj_loss_weight=0.025,
        block_loss_weight=0.1,
        global_loss_weight=0.025,
        clip_hidden_dim=1024,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.backbone.requires_grad_(False)
        self.neck.requires_grad_(False)
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.obj_feature_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(
                in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(
                in_channels=512, out_channels=clip_hidden_dim, kernel_size=3, stride=2, padding=1
            ),
        )
        self.global_head = nn.Linear(256, clip_hidden_dim)

        self.obj_loss_weight = obj_loss_weight  # Weight for L1 loss for objects
        self.block_loss_weight = block_loss_weight  # Weight for L1 loss for blocks
        self.global_loss_weight = (
            global_loss_weight  # Weight for L1 loss for global features
        )

    def global_loss(self, batch_data_samples: Tensor) -> dict:
        level_num = len(self.splited_visual_features)
        embedings_gt = []
        for i in range(level_num):
            for j in range(len(batch_data_samples)):
                if batch_data_samples[j].globals_features is not None:
                    embedings_gt.append(batch_data_samples[j].globals_features)
                else:
                    print(f"globals_features is None for {batch_data_samples[j].img_path}")
        embedings_gt = torch.stack(embedings_gt, dim=0)  # [bs * level_num, 1024]

        mean_feature = []
        for i, feat in enumerate(self.splited_visual_features):
            mean_feature.append(
                feat.mean(dim=(2, 3))
            )  # mean pooling for [bs, 256, ...]
        mean_feature = torch.cat(mean_feature, dim=0)  # [bs * level_num, 256]
        global_feature = self.global_head(mean_feature)  # [bs * level_num, 1024]
        l1_loss = nn.L1Loss()(global_feature, embedings_gt.cuda())
        return {"global_loss": self.global_loss_weight * l1_loss}

    def extract_obj_block_features(self, data_samples):
        """Extract object and block features separately"""
        block_rois, block_embeddings = [], []
        obj_rois, obj_embeddings = [], []

        for i, data_sample in enumerate(data_samples):
            # Process blocks
            if data_sample.blocks_features is not None:
                block_tensor = data_sample.blocks_features["bboxes"]
                num_blocks, _ = block_tensor.shape
                block_idx = torch.full((num_blocks,), i, dtype=torch.long)
                block_rois.append(
                    torch.cat([block_idx.unsqueeze(1), block_tensor], dim=1).to(
                        torch.int32
                    )
                )
                block_embeddings.append(data_sample.blocks_features["embeddings"])
            else:
                print(f"blocks_features is None for {data_sample.img_path}")

            # Process objects
            if data_sample.objects_features is not None:
                object_tensor = data_sample.objects_features["bboxes"]
                num_objs, _ = object_tensor.shape
                obj_idx = torch.full((num_objs,), i, dtype=torch.long)
                obj_rois.append(
                    torch.cat([obj_idx.unsqueeze(1), object_tensor], dim=1).to(
                        torch.int32
                    )
                )
                obj_embeddings.append(data_sample.objects_features["embeddings"])
            else:
                print(f"objects_features is None for {data_sample.img_path}")
        # Combine all blocks and objects
        block_rois = torch.cat(block_rois, dim=0) if block_rois else None
        block_embeddings = (
            torch.cat(block_embeddings, dim=0) if block_embeddings else None
        )
        obj_rois = torch.cat(obj_rois, dim=0) if obj_rois else None
        obj_embeddings = torch.cat(obj_embeddings, dim=0) if obj_embeddings else None

        return block_rois, block_embeddings, obj_rois, obj_embeddings

    def feature_loss(self, rois, embeddings_gt, loss_weight):
        """Compute feature loss for either blocks or objects"""
        if rois is None or embeddings_gt is None:
            return 0.0

        roi_features = self.bbox_roi_extractor(
            self.splited_visual_features, rois
        )  # (N, 256, 7, 7)
        roi_features_conv = self.obj_feature_conv(
            roi_features.cuda()
        ).squeeze()  # (N, 1024)

        # Ensure both tensors have the same dtype
        embeddings_gt = embeddings_gt.cuda()
        # Convert both to float32 for numerical stability
        roi_features_conv = roi_features_conv.float()
        embeddings_gt = embeddings_gt.float()

        # Normalize after ensuring same dtype
        roi_features_conv = F.normalize(roi_features_conv, p=2, dim=1)
        embeddings_gt = F.normalize(embeddings_gt, p=2, dim=1)

        # L1 loss
        l1_loss = nn.L1Loss()(roi_features_conv, embeddings_gt)

        return loss_weight * l1_loss

    def obj_block_loss(
        self, visual_features: Tensor, batch_data_samples: Tensor
    ) -> dict:
        # Prepare visual features
        all_dim = [x * y for x, y in self.visual_feature_dim]
        permute_feat = visual_features.permute(0, 2, 1)
        bs, c, _ = permute_feat.shape
        split_tensors = torch.split(permute_feat, all_dim, dim=2)
        self.splited_visual_features = [
            tensor.view(bs, c, x, y)
            for tensor, (x, y) in zip(split_tensors, self.visual_feature_dim)
        ]

        # Extract features for blocks and objects separately
        block_rois, block_embeddings, obj_rois, obj_embeddings = (
            self.extract_obj_block_features(batch_data_samples)
        )

        losses = {}

        # Compute block loss if blocks are present
        if block_rois is not None:
            block_loss = self.feature_loss(
                block_rois, block_embeddings, self.block_loss_weight
            )
            losses["block_loss"] = block_loss

        # Compute object loss if objects are present
        if obj_rois is not None:
            obj_loss = self.feature_loss(obj_rois, obj_embeddings, self.obj_loss_weight)
            losses["obj_loss"] = obj_loss

        return losses

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor]:
        visual_features = super().extract_feat(batch_inputs)
        self.visual_feature_dim = [list(feat.shape)[-2:] for feat in visual_features]
        return visual_features

    def forward_encoder(self, *args, **kwargs) -> dict:
        encoder_outputs_dict = super().forward_encoder(*args, **kwargs)
        self.encoder_outputs_dict = encoder_outputs_dict
        return encoder_outputs_dict

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        loss = super().loss(batch_inputs, batch_data_samples)
        obj_block_losses = self.obj_block_loss(
            self.encoder_outputs_dict["memory"], batch_data_samples
        )
        global_loss = self.global_loss(batch_data_samples)
        loss.update(obj_block_losses)
        loss.update(global_loss)
        return loss

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        preds = super().predict(batch_inputs, batch_data_samples, rescale)
        pred_instances = preds[0].pred_instances
        img_id = batch_data_samples[0].img_id
        path = f"data/textual_val_shortest_results/"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save({
            "bboxes": pred_instances.bboxes.cpu().numpy(),
            "labels": pred_instances.labels.cpu().numpy(),
            "scores": pred_instances.scores.cpu().numpy(),
        }, f"{path}/pred_instances_{img_id}.pth")
        return preds