from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger


logger = get_logger(__name__)


class RMSNormTorch(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.eps = config.norm_eps
        self.elementwise_affine = config.elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.d_model))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        x = x / rms
        
        if self.weight is not None:
            x = x * self.weight
        
        return x


class RMSNormTriton(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.eps = config.norm_eps
        self.elementwise_affine = config.elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.d_model))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.warning("Triton RMSNorm not implemented, falling back to PyTorch")
        
        torch_rmsnorm = RMSNormTorch(self.config)
        torch_rmsnorm.load_state_dict(self.state_dict())
        
        return torch_rmsnorm(x)


def get_norm(config) -> nn.Module:
    if config.norm_impl == "triton":
        return RMSNormTriton(config)
    else:
        return RMSNormTorch(config)


RMSNorm = RMSNormTorch
