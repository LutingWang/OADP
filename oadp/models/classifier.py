import itertools
import torch
import torch.nn as nn
from torch.functional import F
from einops import rearrange, repeat
from torch import Tensor
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig
from transformers import (AutoTokenizer, CLIPTextConfig)
from transformers import CLIPTextModelWithProjection as CLIPTP

from..utils.globals import Globals

@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):

    def __init__(self,
                 model_name: str,
                 dropout: float = 0.0,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = CLIPTextConfig.from_pretrained(model_name,
                                                     attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=self.config)
        self.out_dim = self.config.projection_dim
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pt', padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: list[list[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True)
        text = text.to(device=self.model.device)
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                      txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):
        self.model.eval()
        for _, module in self.model.named_modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        return

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


class NormalizedLinear(nn.Linear):

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = super().forward(*args, **kwargs)
        return F.normalize(x)

@MODELS.register_module()
class OADPClassifier(BaseModule):
    def __init__(
        self,
        *args,
        text_model: OptMultiConfig,
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.text_model = MODELS.build(text_model)
        self._linear = NormalizedLinear(in_features, self.text_model.out_dim)
        self.bg_embedding = nn.Parameter(torch.zeros(1, self.text_model.out_dim))

    @property
    def texts_feat(self) -> torch.Tensor:
        texts_feat = self.text_model(Globals.texts)
        b, _, _ = texts_feat.shape
        bg_embedding = F.normalize(repeat(self.bg_embedding, '1 c -> b 1 c', b=b))
        return torch.cat([texts_feat, bg_embedding], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        texts_feat_t = rearrange(self.texts_feat, 'b n c -> b c n')
        b, _, _ = texts_feat_t.shape
        roi_feat = rearrange(x, '(b n) c -> b n c', b=b)
        x = self._linear(roi_feat)
        y = torch.matmul(x, texts_feat_t)
        return rearrange(y, 'b n c -> (b n) c')