import torch
import clip
from clip.clip import _tokenizer
import torch.nn as nn
from mmdet.registry import MODELS, DATASETS
from mmdet.datasets.base_det_dataset import BaseDataset
from mmengine.model import BaseModel
from mmengine.fileio import join_path, load

@DATASETS.register_module()
class ImageNet21KDataset(BaseDataset):
    def load_data_list(self):
        annotations = load(self.ann_file)
        data_list = []
        for data_info in annotations['categories']:
            imgs_path = [join_path(self.data_prefix['img'], img) for img in data_info['images']]
            data_info['img_path'] = imgs_path
            data_info['text'] = data_info['name']
            data_list.append(data_info)
        return data_list


@MODELS.register_module()
class FewShotModel(BaseModel):
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