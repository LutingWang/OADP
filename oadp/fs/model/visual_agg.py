import torch
import torch.nn as nn
from clip.model import Transformer
from mmengine.model import BaseModel
from mmdet.registry import MODELS

@MODELS.register_module()
class VisualAggregatorT(BaseModel):
    def __init__(self, max_shots:int, d_model:int, layers:int, heads:int) -> None:
        super().__init__()
        self.context_length = max_shots + 1
        self.d_model = d_model
        self.transformer = Transformer(
            width=d_model,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(),
        )
        self.class_embedding = nn.Parameter(torch.randn(d_model))
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, d_model))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def embedding_visual_tokens(self, image_feats: torch.Tensor, shots: list[int]):
        assert self.context_length >= max(shots) + 1
        embedding = torch.zeros((len(shots), self.context_length, self.d_model), 
                                dtype=image_feats.dtype, 
                                device=image_feats.device) # [batch_size, n_ctx, d_model]
        start = 0
        for i, shot in enumerate(shots):
            embedding[i, 0: shot] = image_feats[start: start + shot, :]
            embedding[i, shot] = self.class_embedding
            start += shot
        assert start == image_feats.shape[0]
        cls_index = torch.tensor([shot for shot in shots])
        return embedding, cls_index
    
    def forward(self, image_feats: torch.Tensor, shots: list[int]):
        x, cls_index = self.embedding_visual_tokens(image_feats, shots)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x[torch.arange(x.shape[0]), cls_index]







