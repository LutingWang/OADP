import re
import torch
import torch.nn as nn
import torch.nn.functional as F

def unfreeze_module(module: nn.Module,
                  white_list: list[str]):
    # unfreeze all parameters in the white list
    for name, child in module.named_children():
        if name in white_list:
            for param in child.parameters():
                param.requires_grad = True
        else:
            unfreeze_module(child, white_list)

def freeze_module(module: nn.Module):
    # freeze all parameters
    for param in module.parameters():
        param.requires_grad = False


def print_all_trainbale_param(module: nn.Module):
    print("============== Trainable parameters ==================")
    for name, param in module.named_parameters():
        if param.requires_grad:
            print(name)


def replace_linear_with_moe(model: nn.Module, 
                            in_features: int,
                            linear_name_pattern: str,
                            num_experts: int,
                            topk: int):
    for name, module in model.named_children():
        if linear_name_pattern == name:
            moe = MoE(module, in_features, num_experts, topk)
            setattr(model, name, moe)
        else:
            replace_linear_with_moe(module, in_features, linear_name_pattern, num_experts, topk)
    return model

class Router(nn.Module):
    """路由器，用于选择专家"""
    def __init__(self, input_dim, num_experts, k=1):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        x = x.mean(dim=1)  # 池化
        logits = self.gate(x)  # 每个专家的logit分数
        top_k = torch.topk(logits, self.k, dim=-1)  # 选择k个专家
        indices = top_k.indices
        scores = F.softmax(top_k.values, dim=-1)  # 转为概率分布
        return indices, scores


class MoE(nn.Module):
    """混合专家模型"""
    def __init__(self, 
                expert_model: nn.Module,  
                input_dim: int, 
                num_experts: int, 
                k: int):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([expert_model for _ in range(num_experts)])
        self.router = Router(input_dim, num_experts, k)
        self.device = next(expert_model.parameters()).device
        self.router.to(self.device)

    def forward(self, x):
        indices, scores = self.router(x)
        # 初始化输出
        batch_size = x.size(0)
        output = torch.zeros_like(x)
        for i in range(self.router.k):
            expert_idx = indices[:, i]
            score = scores[:, i].unsqueeze(-1)
            for b in range(batch_size):
                output[b] += score[b] * self.experts[expert_idx[b]](x[b])
        return output