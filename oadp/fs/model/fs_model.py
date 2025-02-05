import torch
import torch.nn as nn
import clip
import clip.model
from clip.clip import _tokenizer
from mmdet.registry import MODELS
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
            clip_model_path:str,
            vision_agg_cfg:dict,
            projector_type:str, 
            loss_type:str
        ) -> None:
        super().__init__()
        self.language_model = MODELS.build(language_model_cfg)
        self.language_dim = self.language_model.language_backbone.body.language_dim

        self.clip_model, _ = clip.load(clip_model_path)
        self.visual_agg: VisualAggregatorT = MODELS.build(vision_agg_cfg)
        self.visual_dim = self.visual_agg.d_model

        self.vision_projector = VisionProjector(self.visual_dim, self.language_dim, projector_type)
        
        self.loss = ContrastiveLoss()

        # freeze all the parameters
        for param in self.parameters():
            param.requires_grad = False

        # unfreeze the parameters of the clip text encoder
        for param in self.visual_agg.parameters():
            param.requires_grad = True
        for param in self.vision_projector.parameters():
            param.requires_grad = True


        if loss_type == "contrastive":
            self.loss = ContrastiveLoss()
        elif loss_type == "l1":
            self.loss = nn.L1Loss()
        elif loss_type == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
    def forward(self, inputs, texts, shots, mode):
        if mode == "loss":
            image_feats = self.clip_model.encode_image(inputs)
            image_feats = self.visual_agg(image_feats, shots)
            text_feats = self.language_model(texts)['embedded'].sum(dim=1)
            image_feats_align = self.vision_projector(image_feats)
            loss = {"loss": self.loss(image_feats_align, text_feats)}
            return loss



@MODELS.register_module()
class FewShotModelCLIP(BaseModel):
    def __init__(self, language_model, clip_model_path) -> None:
        super().__init__()
        self.bert_model_cfg = language_model

        self.bert_model = MODELS.build(self.bert_model_cfg)
        self.embed_dims = self.bert_model.language_backbone.body.language_dim
        
        self.clip_model, _ = clip.load(clip_model_path)
        self.clip_tokenizer = clip.tokenize

        self.visual2text_projector = nn.Linear(self.clip_model.visual.output_dim, self.embed_dims)

        self.loss = nn.L1Loss()

        # freeze all the parameters
        for param in self.parameters():
            param.requires_grad = False

        # unfreeze the parameters of the clip text encoder
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = True
        for param in self.visual2text_projector.parameters():
            param.requires_grad = True
        self.clip_model.text_projection.requires_grad = True 
    
    def aggregate_visual(self, image_feats: torch.Tensor, eot_index: torch.Tensor):
        x = image_feats + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_index] @ self.clip_model.text_projection

        return x

    def embedding_visual_tokens(self, image_feats: torch.Tensor, shots: list[int], context_length=77):
        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        start_em, end_em = self.clip_model.token_embedding(torch.tensor([sot_token, eot_token]).to(image_feats.device))
        assert context_length >= max(shots) + 2
        embedding = torch.zeros((len(shots), context_length, self.clip_model.visual.output_dim), 
                                dtype=image_feats.dtype, 
                                device=image_feats.device) # [batch_size, n_ctx, d_model]
        start = 0
        for i, shot in enumerate(shots):
            embedding[i, 0] = start_em
            embedding[i, 1: shot + 1] = image_feats[start: start + shot, :]
            embedding[i, shot + 1] = end_em
            start += shot
        assert start == image_feats.shape[0]
        eot_index = torch.tensor([shot + 1 for shot in shots])
        return embedding, eot_index

    def forward(self, inputs, texts, shots, mode):
        if mode == "loss":
            image_feats = self.clip_model.encode_image(inputs)
            embeded_feats, eot_index= self.embedding_visual_tokens(image_feats, shots)
            aggregate_feature = self.aggregate_visual(embeded_feats, eot_index)

            text_feats = self.bert_model(texts)['embedded'].sum(dim=1)
            aggregate_feature_align = self.visual2text_projector(aggregate_feature.type(text_feats.dtype))
            loss = {"loss": self.loss(aggregate_feature_align, text_feats)}
            return loss