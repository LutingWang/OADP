import clip
import torch

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
        # shape = [global_batch_size, global_batch_size]
        pred = {
            'pred_logits': probs,
            'gt_label': data_samples
        }
        return pred, None