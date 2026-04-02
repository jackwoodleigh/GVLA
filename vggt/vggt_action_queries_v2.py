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


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: queries attend to VGGT key/values."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(queries)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(query=q, key=kv_normed, value=kv_normed)
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


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
        self.num_layers = num_layers
        self.stride = stride
        self.num_ca_layers = num_layers // stride

        # Learnable query parameters — small random init (not zero!)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, vggt_dim) * 0.02)

        # Only num_layers // stride cross-attention layers
        self.ca_layers = nn.ModuleList([
            CrossAttentionLayer(dim=vggt_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(1)
        ])

        # Final projection: vggt_dim -> llm_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(vggt_dim),
            nn.Linear(vggt_dim, llm_dim)
        )

    def forward(self, vggt_layers, patch_start_idx):
        B = vggt_layers[0].shape[0]
        queries = self.query_tokens.expand(B, -1, -1)

        per_layer_outputs = []
        #for i, ca_layer in enumerate(self.ca_layers):
        for vggt_idx in range(0, len(vggt_layers), self.stride):
            kv = vggt_layers[vggt_idx].squeeze(1)[:, patch_start_idx:]
            queries = self.ca_layers[0](queries, kv)
            per_layer_outputs.append(self.output_proj(queries))

        # [B, num_ca_layers, N_q, llm_dim]
        stacked = torch.stack(per_layer_outputs, dim=1)

        # Expand to full num_layers by repeating each along layer dim
        # [B, num_ca_layers, N_q, llm_dim] -> [B, num_layers, N_q, llm_dim]
        stacked = stacked.repeat_interleave(self.stride, dim=1)[:, :self.num_layers]

        return stacked