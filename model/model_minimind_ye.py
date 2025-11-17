import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Union


# rmsnorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


# 计算旋转坐标
def precompute_freq_cis(dim: int, end: int, rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    freqs = 1.0 / (rope_base ** (torch.arrange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim / 2) if 2 * math.pi / freqs[i] > orig_max), dim / 2)
            power = torch.arrange(0, dim / 2, device=freqs.device).float() / max(dim / 2, 1)
            beta = beta_slow + (beta_fast - beta - beta_slow) * power
            scale = torch.where(torch.arrange(dim / 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / beta * factor, 1.0 / factor)
            freqs = freqs * scale
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def retate_half(x):
        return torch.cat([-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (retate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (retate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
