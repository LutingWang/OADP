import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
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


@MODELS.register_module()
class QFormer(nn.Module):
    def __init__(
        self,
        d_model: int,           # 输入数据的维度（如文本/图像特征的维度）
        d_query: int,           # 查询向量的维度
        num_queries: int,       # 查询向量的数量
        max_shots: int,         # 最大样本数
        num_layers: int = 1,    # 编码器层数
        num_heads: int = 8,     # 注意力头数
        dropout: float = 0.0,   # dropout 概率
    ):
        super().__init__()
        self.context_length = max_shots + 1
        self.num_queries = num_queries
        self.d_query = d_query
        self.d_model = d_model
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, d_query))  # 可学习的查询向量

        # 自注意力层：查询向量内部交互
        self_encoder_layer = TransformerEncoderLayer(
            d_model=d_query,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = TransformerEncoder(self_encoder_layer, num_layers=num_layers)

        # 交叉注意力层：查询与输入数据交互
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_query,
            kdim=d_model,
            vdim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
    def embedding_visual_tokens(self, image_feats: torch.Tensor, shots: list[int]):
        assert self.context_length >= max(shots) + 1
        embedding = torch.zeros((len(shots), self.context_length, self.d_model), 
                                dtype=image_feats.dtype, 
                                device=image_feats.device) # [batch_size, n_ctx, d_model]
        padding_mask = torch.zeros((len(shots), self.context_length), dtype=torch.bool, device=image_feats.device)
        start = 0
        for i, shot in enumerate(shots):
            embedding[i, 0: shot] = image_feats[start: start + shot, :]
            padding_mask[i, shot] = True
            start += shot
        assert start == image_feats.shape[0]
        return embedding, padding_mask
    
    def forward(self, x: torch.Tensor, shots: list[int]) -> torch.Tensor:
        x, key_padding_mask = self.embedding_visual_tokens(x, shots)
        batch_size = x.shape[0] # [batch_size, n_ctx, d_model]

        # 扩展查询向量到 batch 维度
        queries = self.query_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, num_queries, d_query]

        # 自注意力：查询向量内部交互
        queries = self.self_attn(queries)  # [batch, num_queries, d_query]

        # 交叉注意力：查询与输入数据交互
        attn_output, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        ) 
        return attn_output  # [batch, num_queries, d_query]