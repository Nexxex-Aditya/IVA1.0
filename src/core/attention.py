"""Attention implementations"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger


logger = get_logger(__name__)


class AttentionTorch(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.causal = config.causal
        
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        rope_state: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        sink_cfg: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        if rope_state is not None:
            q = self._apply_rope(q, rope_state)
            k = self._apply_rope(k, rope_state)
        
        if kv_cache is not None:
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)
            
            new_kv_cache = type(kv_cache)(
                batch_size=kv_cache.batch_size,
                n_heads=kv_cache.n_heads,
                head_dim=kv_cache.head_dim,
                seq_len=kv_cache.seq_len + seq_len,
                device=device
            )
            new_kv_cache.k = k
            new_kv_cache.v = v
        else:
            new_kv_cache = None
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, k.size(2), device=device))
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        if sliding_window is not None:
            window_mask = self._create_sliding_window_mask(
                seq_len, k.size(2), sliding_window, device
            )
            scores = scores.masked_fill(window_mask == 0, float('-inf'))
        
        if sink_cfg is not None:
            sink_mask = self._create_sink_mask(
                seq_len, k.size(2), sink_cfg, device
            )
            scores = scores.masked_fill(sink_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.o_proj(attn_output)
        
        return output, new_kv_cache
    
    def _apply_rope(self, x: torch.Tensor, rope_state: torch.Tensor) -> torch.Tensor:
        return x
    
    def _create_sliding_window_mask(
        self,
        seq_len: int,
        k_len: int,
        window_size: int,
        device: torch.device
    ) -> torch.Tensor:
        mask = torch.zeros(seq_len, k_len, device=device)
        
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            end = min(k_len, i + 1)
            mask[i, start:end] = 1
        
        return mask
    
    def _create_sink_mask(
        self,
        seq_len: int,
        k_len: int,
        sink_cfg: Dict[str, Any],
        device: torch.device
    ) -> torch.Tensor:
        sink_tokens = sink_cfg.get("sink_tokens", 4)
        mask = torch.zeros(seq_len, k_len, device=device)
        
        for i in range(min(sink_tokens, seq_len)):
            mask[i, :i+1] = 1
        
        for i in range(sink_tokens, seq_len):
            mask[i, :sink_tokens] = 1
            mask[i, sink_tokens:i+1] = 1
        
        return mask


class AttentionTriton(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.causal = config.causal
        
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        rope_state: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        sink_cfg: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        logger.warning("Triton attention not implemented, falling back to PyTorch")
        
        torch_attention = AttentionTorch(self.config)
        torch_attention.load_state_dict(self.state_dict())
        
        return torch_attention(
            x, attention_mask, kv_cache, rope_state, sliding_window, sink_cfg
        )


def get_attention(config) -> nn.Module:
    if config.attn_impl == "triton":
        return AttentionTriton(config)
    else:
        return AttentionTorch(config)

Attention = AttentionTorch
