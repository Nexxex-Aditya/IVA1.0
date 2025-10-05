"""Feed forward network implementations"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger


logger = get_logger(__name__)


class SwiGLU(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FFN(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.hidden_dim = config.d_model * config.hidden_mult
        self.activation = config.activation
        self.dropout = config.dropout
        
        self.gate_proj = nn.Linear(self.d_model, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, self.d_model, bias=False)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        if self.activation == "swiglu":
            self.activation_fn = SwiGLU(self.hidden_dim)
        elif self.activation == "gelu":
            self.activation_fn = nn.GELU()
        elif self.activation == "relu":
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = self.activation_fn(torch.cat([gate, up], dim=-1))
        else:
            hidden = self.activation_fn(self.gate_proj(x))
            hidden = self.dropout_layer(hidden)
            hidden = hidden * self.up_proj(x)  # gated activation
        
        output = self.down_proj(hidden)
        
        return output


class FFNTriton(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.hidden_dim = config.d_model * config.hidden_mult
        self.activation = config.activation
        self.dropout = config.dropout
        
        self.gate_proj = nn.Linear(self.d_model, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, self.d_model, bias=False)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        if self.activation == "swiglu":
            self.activation_fn = SwiGLU(self.hidden_dim)
        elif self.activation == "gelu":
            self.activation_fn = nn.GELU()
        elif self.activation == "relu":
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.warning("Triton FFN not implemented, falling back to PyTorch")
        
        torch_ffn = FFN(self.config)
        torch_ffn.load_state_dict(self.state_dict())
        
        return torch_ffn(x)


def get_ffn(config) -> nn.Module:
    if config.attn_impl == "triton":
        return FFNTriton(config)
    else:
        return FFN(config)
