import torch
import torch.nn as nn
import clip
import clip.model
from clip.clip import _tokenizer
from mmdet.registry import MODELS
from mmdet.models.language_models import BertModel
from transformers import BertModel as HFBertModel
from mmengine.model import BaseModel

from .projector import VisionProjector
from .visual_agg import VisualAggregatorT

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, image_embeds, text_embeds):
        logits = torch.matmul(image_embeds, text_embeds.T) / self.temperature
        labels = torch.arange(len(image_embeds), device=image_embeds.device)
        loss_i = self.cross_entropy(logits, labels)
        loss_t = self.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2

@MODELS.register_module()
class FewShotModel(BaseModel):
    def __init__(self, 
            language_model_cfg:dict,
            vision_agg_cfg:dict,
            projector_type:str, 
            loss_type:str
        ) -> None:
        super().__init__()
        # bert model
        self.language_model: BertModel = MODELS.build(language_model_cfg)
        self.language_dim = self.language_model.language_backbone.body.language_dim
        # visual agg model
        self.visual_agg: VisualAggregatorT = MODELS.build(vision_agg_cfg)
        self.visual_dim = self.visual_agg.d_model
        # projector
        self.vision_projector = VisionProjector(self.visual_dim, self.language_dim, projector_type)
        # loss
        if loss_type == "contrastive":
            self.loss = ContrastiveLoss()
        elif loss_type == "l1":
            self.loss = nn.L1Loss()
        elif loss_type == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        self.language_model.language_backbone.body.model.embeddings.word_embeddings.requires_grad_(False)
    
    def image_bert_forward(self, image_feats):
        self.bert_model: HFBertModel = self.language_model.language_backbone.body.model
        outputs = self.bert_model(
            inputs_embeds=image_feats.unsqueeze(1), # [batch_size, seq_length, embed_dim]
            output_hidden_states=True,
        )
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-1:], 1).mean(1)
        return features.squeeze(1)

    def text_bert_forward(self, texts):
        return self.language_model(texts)['embedded'].sum(dim=1)

    def forward(self, 
            inputs: torch.Tensor, 
            texts: list[str], 
            shots: list[int], 
            to_bert: bool = False
        ):
        image_feats = self.visual_agg(inputs, shots)
        image_feats_align = self.vision_projector(image_feats)
        if to_bert:
            image_feats_align = self.image_bert_forward(image_feats_align)
        text_feats = self.text_bert_forward(texts)
        loss = {"loss": self.loss(image_feats_align, text_feats)}
        return loss, image_feats_align