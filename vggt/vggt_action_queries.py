"""
VGGTActionQueryModule
=====================
Learnable action queries cross-attend over VGGT's intermediate layer
representations. A stride parameter controls which VGGT layers are
attended to, reducing the number of cross-attention layers while
expanding the output back to the full layer count.

E.g. with num_layers=24 and stride=2:
  - 12 cross-attention layers are created
  - Attend to VGGT layers [0, 2, 4, ..., 22]
  - Each output is repeated to fill the stride gap
  - Final output shape: [B, 24, N_q, llm_dim]
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]   
        x2 = x[..., 1::2] 

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot

def headify(x, heads):
    B, T, _ = x.shape
    return x.view(B, T, heads, -1).transpose(1, 2)

def deheadify(x):
    B, _, T, _ = x.shape
    return x.transpose(1, 2).reshape(B, T, -1)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class BridgeAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.head_dim = dim // num_heads
        self.heads = num_heads
        self.rope = RotaryPositionEmbedding(self.head_dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        self.x_self_qkv = nn.Linear(dim, dim*3)
        self.x_cross_q = nn.Linear(dim, dim)
        self.vggt_kv = nn.Linear(dim, dim*2)
        
        self.out = nn.Linear(dim, dim)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )

        

    def forward(self, x: torch.Tensor, vggt: torch.Tensor) -> torch.Tensor:
        # norm
        x_norm = self.norm_q(x)
        vggt_norm = self.norm_kv(vggt)

        # projection
        qkv_self, q_cross = self.x_self_qkv(x_norm), self.x_cross_q(x_norm)
        vggt_kv = self.vggt_kv(vggt_norm)

        # head reshape
        vggt_kv = headify(vggt_kv, self.heads)
        qkv_self, q_cross = headify(qkv_self, self.heads), headify(q_cross, self.heads)

        q_self, k_self, v_self = qkv_self.chunk(3, dim=-1)
        vggt_k, vggt_v = vggt_kv.chunk(2, dim=-1)
        
        # RoPE
        cos_main, sin_main = self.rope(seq_len=q_self.shape[2], device=x.device, dtype=x.dtype)
        q_self, k_self = apply_rope(q_self, k_self, cos_main, sin_main)

        # scaled dot
        attn_scores = [torch.matmul(q_self, k_self.transpose(-2, -1)), torch.matmul(q_cross, vggt_k.transpose(-2, -1))]
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, torch.cat([v_self, vggt_v], dim=2)) 
    
        attn_out = deheadify(attn_out)

        x = self.out(self.norm_out(attn_out)) + x
        x = self.ffn(x) + x

        return x


class VGGTActionQueryModule(nn.Module):
    def __init__(
        self,
        num_queries: int = 64,
        vggt_dim: int = 2048,
        llm_dim: int = 896,
        num_layers: int = 24,
        num_heads: int = 8,
        dropout: float = 0.0,
        stride: int = 2,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.vggt_dim = vggt_dim
        self.llm_dim = llm_dim
        self.num_layers = num_layers
        self.stride = stride
        self.num_ca_layers = num_layers // stride

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)

        self.ca_layers = nn.ModuleList([
            BridgeAttentionLayer(dim=llm_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(1)
        ])

        self.vggt_proj = nn.Sequential(
            nn.LayerNorm(vggt_dim),
            nn.Linear(vggt_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, vggt_layers, patch_start_idx):
        B = vggt_layers[0].shape[0]
        queries = self.query_tokens.expand(B, -1, -1)

        per_layer_outputs = []
        for vggt_idx in range(0, len(vggt_layers), self.stride):
            vggt_kv = vggt_layers[vggt_idx].squeeze(1)[:, patch_start_idx:]
            queries = self.ca_layers[0](queries, self.vggt_proj(vggt_kv))
            per_layer_outputs.append(queries)

        stacked = torch.stack(per_layer_outputs, dim=1)
        stacked = stacked.repeat_interleave(self.stride, dim=1)[:, :self.num_layers]

        return stacked