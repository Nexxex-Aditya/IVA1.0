from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class RoPE(nn.Module):
    
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        
        self.register_buffer(
            'freqs',
            self._compute_freqs(head_dim, theta),
            persistent=False
        )
    
    def _compute_freqs(self, head_dim: int, theta: float) -> torch.Tensor:
        freqs = torch.zeros(head_dim, dtype=torch.float32)
        
        for i in range(0, head_dim, 2):
            freqs[i] = 1.0 / (theta ** (i / head_dim))
            if i + 1 < head_dim:
                freqs[i + 1] = freqs[i]
        
        return freqs
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = x.shape
        device = x.device
        
        pos_freqs = position_ids.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        pos_freqs = pos_freqs.to(device)
        
        cos = torch.cos(pos_freqs)
        sin = torch.sin(pos_freqs)
        
        cos = cos.unsqueeze(1).expand(-1, n_heads, -1, -1)
        sin = sin.unsqueeze(1).expand(-1, n_heads, -1, -1)
        
        x_rotated = torch.zeros_like(x)
        
        for i in range(0, head_dim, 2):
            if i + 1 < head_dim:
                x_rotated[..., i] = x[..., i] * cos[..., i] - x[..., i + 1] * sin[..., i]
                x_rotated[..., i + 1] = x[..., i] * sin[..., i] + x[..., i + 1] * cos[..., i]
            else:
                x_rotated[..., i] = x[..., i]
        
        return x_rotated
    
    def get_rope_state(self, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = position_ids.shape
        device = position_ids.device
        
        pos_freqs = position_ids.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        pos_freqs = pos_freqs.to(device)
        
        cos = torch.cos(pos_freqs)
        sin = torch.sin(pos_freqs)
        
        rope_state = torch.stack([cos, sin], dim=-1)
        
        return rope_state


class KVCache:
    
    def __init__(
        self,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16
    ):
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype
        
        self.k = torch.zeros(
            batch_size, n_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v = torch.zeros(
            batch_size, n_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )
    
    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        start_pos: int
    ) -> None:
        end_pos = start_pos + new_k.size(2)
        
        if end_pos > self.seq_len:
            self.resize(end_pos)
        
        self.k[:, :, start_pos:end_pos] = new_k
        self.v[:, :, start_pos:end_pos] = new_v
    
    def resize(self, new_seq_len: int) -> None:
        if new_seq_len <= self.seq_len:
            return
        
        new_k = torch.zeros(
            self.batch_size, self.n_heads, new_seq_len, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        new_v = torch.zeros(
            self.batch_size, self.n_heads, new_seq_len, self.head_dim,
            device=self.device, dtype=self.dtype
        )
        
        new_k[:, :, :self.seq_len] = self.k
        new_v[:, :, :self.seq_len] = self.v
        
        self.k = new_k
        self.v = new_v
        self.seq_len = new_seq_len
    
    def clear(self) -> None:
        self.k.zero_()
        self.v.zero_()
    
    def get_slice(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k[:, :, start:end], self.v[:, :, start:end]


def apply_rope(
    x: torch.Tensor,
    rope_state: torch.Tensor
) -> torch.Tensor:
    batch_size, n_heads, seq_len, head_dim = x.shape
    device = x.device
    
    cos = rope_state[..., 0]
    sin = rope_state[..., 1]
    
    cos = cos.unsqueeze(1).expand(-1, n_heads, -1, -1)
    sin = sin.unsqueeze(1).expand(-1, n_heads, -1, -1)
    
    x_rotated = torch.zeros_like(x)
    
    for i in range(0, head_dim, 2):
        if i + 1 < head_dim:
            x_rotated[..., i] = x[..., i] * cos[..., i] - x[..., i + 1] * sin[..., i]
            x_rotated[..., i + 1] = x[..., i] * sin[..., i] + x[..., i + 1] * cos[..., i]
        else:
            x_rotated[..., i] = x[..., i]
    
    return x_rotated


def create_kv_cache(
    batch_size: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16
) -> KVCache:
    return KVCache(batch_size, n_heads, head_dim, seq_len, device, dtype)


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        end = i + 1
        mask[i, start:end] = 1
    
    return mask


def create_causal_mask(
    seq_len: int,
    device: torch.device
) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_attention_sink_mask(
    seq_len: int,
    sink_tokens: int = 4,
    device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(sink_tokens):
        mask[i, :i+1] = 1
    
    for i in range(sink_tokens, seq_len):
        mask[i, :sink_tokens] = 1
        mask[i, sink_tokens:i+1] = 1
    
    return mask
