import torch.nn as nn
from mmdet.registry import MODELS


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


@MODELS.register_module()
class VisionProjector(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim):
        super().__init__()
        self._module_list = nn.ModuleList()
        self._module_list.append(nn.Linear(visual_dim, hidden_dim))
        self._module_list.append(nn.GELU())
        self._module_list.append(nn.Linear(hidden_dim, text_dim))
        self._module_list.append(SimpleResBlock(text_dim))
        self._module_list.append(SimpleResBlock(text_dim))
        self._module_list.append(SimpleResBlock(text_dim))

    def forward(self, visual_feats):
        for module in self._module_list:
            visual_feats = module(visual_feats)
        return visual_feats
