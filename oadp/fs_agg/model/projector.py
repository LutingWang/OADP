import re
import torch.nn as nn
from mmengine.model import BaseModel

class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class VisionProjector(BaseModel):
    def __init__(self, visual_dim, text_dim, projector_type) -> None:
        super().__init__()
        if projector_type == "linear":
            self.projector = nn.Linear(visual_dim, text_dim)
        
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(visual_dim, text_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(text_dim, text_dim))
            self.projector = nn.Sequential(*modules)

        mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
        if mlp_gelu_resnet_match:
            mlp_depth = int(mlp_gelu_resnet_match.group(1))
            res_depth = int(mlp_gelu_resnet_match.group(2))
            modules = [nn.Linear(visual_dim, text_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(text_dim, text_dim))
            for _ in range(res_depth):
                modules.append(SimpleResBlock(text_dim))
            self.projector = nn.Sequential(*modules)

    def forward(self, visual_feats):
        return self.projector(visual_feats)