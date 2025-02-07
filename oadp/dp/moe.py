import re
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def mark_only_moe_as_trainable(module: nn.Module):
    if isinstance(module, MoE):
        module.set_requires_grad(True)
    else:
        for name, child in module.named_children():
            mark_only_moe_as_trainable(child)

def extract_moe_aux_loss(module: nn.Module, w: float=0.1, aux_loss={}, prefix="moe_aux_"):
    from .moe import MoE
    if isinstance(module, MoE):
        aux_loss[prefix] = module.aux_loss * w
    for name, child in module.named_children():
        extract_moe_aux_loss(child, w, aux_loss, prefix + name + "_")
    return aux_loss

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
        self.experts = []
        for _ in range(num_experts):
            new_expert = copy.deepcopy(expert_model)
            self.experts.append(new_expert)
        self.experts = nn.ModuleList(self.experts)
        self.router = Router(input_dim, num_experts, k)
        self.device = next(expert_model.parameters()).device
        self.router.to(self.device)

    def set_requires_grad(self, requires_grad: bool):
        for expert in self.experts[1:]:
            for param in expert.parameters():
                param.requires_grad = requires_grad
        self.router.requires_grad_(requires_grad)

    def forward(self, x):
        indices, scores = self.router(x)
        num_experts = len(self.experts)
        # 计算辅助损失（负载均衡）
        expert_counts = torch.zeros(num_experts, device=x.device)
        for idx in indices.view(-1):
            expert_counts[idx] += 1
        expert_frac = expert_counts / expert_counts.sum()
        uniform_frac = torch.ones_like(expert_frac) / num_experts
        self.aux_loss = F.mse_loss(expert_frac, uniform_frac)

        # 初始化输出
        batch_size = x.size(0)
        output = torch.zeros_like(x)
        for i in range(self.router.k):
            expert_idx = indices[:, i]
            score = scores[:, i].unsqueeze(-1)
            for b in range(batch_size):
                output[b] += score[b] * self.experts[expert_idx[b]](x[b])
        return output