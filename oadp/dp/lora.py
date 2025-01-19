import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=1):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank

        # Assume original_layer is a Linear layer
        self.lora_A = nn.Parameter(torch.zeros((original_layer.out_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, original_layer.in_features)))
        
        # Initialize the low-rank matrices
        nn.init.kaiming_uniform_(self.lora_A, a=nn.init.calculate_gain('relu'))
        nn.init.kaiming_uniform_(self.lora_B, a=nn.init.calculate_gain('relu'))

    def forward(self, x):
        # Original output from the layer
        original_output = self.original_layer(x)
        
        # Low-rank adaptation output
        lora_output = F.linear(x, self.lora_B.T @ self.lora_A)
        
        # Combine original output with LoRA output
        return original_output + lora_output