import torch
import torch.nn as nn
from loralib import Linear
from mmengine.logging import print_log

def is_in_blacklist(name: str, blacklist: list[str]):
    for black in blacklist:
        if black in name:
            return True
    return False


def replace_linear_with_lora(model: nn.Module, 
                             alpha: int = 8, 
                             rank: int = 4,
                             blacklist: list = [],
                             parent_name: str = None):
    for name, module in model.named_children():
        current_name = parent_name + '.' + name if parent_name is not None else name
        if isinstance(module, nn.Linear) and not is_in_blacklist(name, blacklist):
            print_log(f'Replacing {current_name} with LoRA')
            setattr(model, name, Linear(module.in_features, module.out_features, alpha, rank))
        else:
            replace_linear_with_lora(module, alpha, rank, blacklist, parent_name=current_name)
    return model
