import clip
import torch
import json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from ram.models import ram_plus

from mmengine.model import BaseModel
from mmengine.registry import MODELS

from exps.muti_label.globals import cur_cates

@MODELS.register_module()
class CLIPModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        self.cate_text = clip.tokenize(cur_cates)

    @torch.no_grad()
    def forward(self, batch_inputs, data_samples, mode='tensor', **kwargs) -> torch.Tensor:
        logits_per_image, logits_per_text = self.model(batch_inputs, self.cate_text.cuda())
        probs = logits_per_image.softmax(dim=-1)
        pred = {
            'pred_logits': probs,
            'data_samples': data_samples
        }
        return pred, None

@MODELS.register_module()
class RAMModel(BaseModel):
    def __init__(self, model_path: str, llm_tag_des:str, image_size=384) -> None:
        super().__init__()
        self.model = ram_plus(pretrained=model_path, image_size=image_size, vit='swin_l')

        label_info = torch.load(llm_tag_des)
        openset_label_embedding = label_info['openset_label_embedding']
        openset_categories = label_info['openset_categories']

        self.model.tag_list = np.array(openset_categories)
        self.model.label_embed = nn.Parameter(openset_label_embedding.float())
        self.model.num_class = len(openset_categories)
        self.model.class_threshold = torch.ones(self.model.num_class) * 0.5

    def model_forward(self, image):

        image_embeds = self.model.image_proj(self.model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]

        des_per_class = int(self.model.label_embed.shape[0] / self.model.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
        reweight_scale = self.model.reweight_scale.exp()
        logits_per_image = (reweight_scale * image_cls_embeds @ self.model.label_embed.t())
        logits_per_image = logits_per_image.view(bs, -1,des_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        label_embed_reweight = torch.empty(bs, self.model.num_class, 512).to(image.device).to(image.dtype)
                     
        for i in range(bs):
            # 这里对 value_ori 进行 reshape，然后使用 broadcasting
            reshaped_value = self.model.label_embed.view(-1, des_per_class, 512)
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(dim=1)

        label_embed = torch.nn.functional.relu(self.model.wordvec_proj(label_embed_reweight))

        # recognized image tags using alignment decoder
        tagging_embed = self.model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.model.fc(tagging_embed[0]).squeeze(-1)
        return logits

    @torch.no_grad()
    def forward(self, batch_inputs, data_samples, mode='tensor', **kwargs) -> torch.Tensor:
        probs = self.model_forward(batch_inputs)
        pred = {
            'pred_logits': probs,
            'data_samples': data_samples
        }
        return pred, None