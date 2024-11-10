import clip
import torch

from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class CLIPModel(BaseModel):
    def __init__(self, categories: list[str]) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        self.cate_text = clip.tokenize(categories)

    @torch.no_grad()
    def forward(self, batch_inputs, data_samples, mode='tensor', **kwargs) -> torch.Tensor:
        text_features = self.model.encode_text(self.cate_text.cuda())
        image_features = self.model.encode_image(batch_inputs)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # shape = [global_batch_size, global_batch_size]
        pred = {
            'pred_logits': logits_per_image,
            'gt_label': data_samples
        }
        return pred, None