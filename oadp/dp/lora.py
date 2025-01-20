import torch
import torch.nn as nn
from loralib import Linear


def replace_linear_with_lora(model: nn.Module, 
                             alpha: int = 8, 
                             rank: int = 4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, Linear(module.in_features, module.out_features, alpha, rank))
        else:
            replace_linear_with_lora(module, alpha, rank)
    return model
