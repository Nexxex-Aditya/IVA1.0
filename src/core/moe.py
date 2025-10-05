"""Mixture of Experts"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger
from ..utils.distributed import get_expert_group


logger = get_logger(__name__)


class Top2Router(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.aux_loss_coef = config.aux_loss_coef
        self.dropless = config.dropless
        
        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)
        
        self.temp = nn.Parameter(torch.ones(1))
        
        self.jitter = 0.1
        self.noise_std = 0.1
    
    def forward(
        self,
        x: torch.Tensor,
        router_temp: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        logits = self.router(x)

        temp = router_temp if router_temp is not None else self.temp.clamp(min=0.1, max=5.0)
        logits = logits / temp
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)

        expert_probs, expert_indices = torch.topk(probs, self.top_k, dim=-1)
        
        expert_mask = torch.zeros(batch_size, seq_len, self.n_experts, device=device)
        expert_mask.scatter_(-1, expert_indices, 1.0)
        
        aux_losses = self._compute_aux_losses(probs, expert_mask)
        
        return expert_probs, expert_indices, expert_mask, aux_losses
    
    def _compute_aux_losses(
        self,
        full_probs: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        - compute auxiliary losses for load balancing.
        - Importance = mean routing probability per expert.
        - Load = fraction of tokens assigned per expert.
        - L_aux = sum_e importance_e * load_e.
        """
        aux_losses: Dict[str, torch.Tensor] = {}
        if self.aux_loss_coef <= 0:
            return aux_losses

        batch_size, seq_len, n_experts = full_probs.shape
        device = full_probs.device

        importance = full_probs.mean(dim=(0, 1))

        assigned = expert_mask.sum(dim=(0, 1))
        load = assigned / (batch_size * seq_len + 1e-6)

        aux = (importance * load).sum() * n_experts
        aux_losses["aux_load_loss"] = aux * self.aux_loss_coef

        token_entropy = -(full_probs * (full_probs.clamp_min(1e-8)).log()).sum(dim=-1).mean()
        aux_losses["router_entropy"] = token_entropy * 0.01
        return aux_losses


class ExpertMLP(nn.Module):
    
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
            self.activation_fn = self._swiglu
        elif self.activation == "gelu":
            self.activation_fn = nn.GELU()
        elif self.activation == "relu":
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _swiglu(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = self.activation_fn(torch.cat([gate, up], dim=-1))
        else:
            hidden = self.activation_fn(self.gate_proj(x))
            hidden = self.dropout_layer(hidden)
            hidden = hidden * self.up_proj(x)
        
        output = self.down_proj(hidden)
        
        return output


class MoEFFN(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.dropless = config.dropless
        self.expert_parallel = config.expert_parallel
        
        self.router = Top2Router(config)
        
        self.experts = nn.ModuleList([
            ExpertMLP(config) for _ in range(self.n_experts)
        ])
        
        if self.expert_parallel:
            self.expert_groups = [
                get_expert_group(i, self.n_experts) for i in range(self.n_experts)
            ]
        else:
            self.expert_groups = None
    
    def forward(
        self,
        x: torch.Tensor,
        router_temp: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        expert_weights, expert_indices, expert_mask, aux_losses = self.router(
            x, router_temp
        )
        
        capacity = max(1, int((batch_size * seq_len * self.top_k / self.n_experts) * self.capacity_factor))
        
        expert_outputs = []
        overflow_count = torch.zeros(self.n_experts, device=x.device)
        for expert_id in range(self.n_experts):
            expert_mask_i = expert_mask[:, :, expert_id]
            token_indices = expert_mask_i.nonzero(as_tuple=False)
            if token_indices.numel() == 0:
                expert_outputs.append((expert_id, None, expert_mask_i))
                continue

            if not self.dropless and token_indices.size(0) > capacity:
                keep_idx = token_indices[:capacity]
                drop_idx = token_indices[capacity:]
                drop_mask = torch.zeros_like(expert_mask_i)
                drop_mask[drop_idx[:, 0], drop_idx[:, 1]] = 1.0
                expert_mask_i = expert_mask_i * (1.0 - drop_mask)
                overflow_count[expert_id] = drop_idx.size(0)
                token_indices = keep_idx

            expert_tokens = x[token_indices[:, 0], token_indices[:, 1]]
            
            if expert_tokens.size(0) > 0:
                if self.expert_parallel and self.expert_groups[expert_id] is not None:
                    expert_output = self._expert_parallel_forward(
                        expert_tokens, expert_id
                    )
                else:
                    expert_output = self.experts[expert_id](expert_tokens)
                
                expert_outputs.append((expert_id, expert_output, expert_mask_i, token_indices))
            else:
                expert_outputs.append((expert_id, None, expert_mask_i, token_indices))
        
        output = self._combine_expert_outputs(
            expert_outputs, expert_weights, expert_indices, x.shape
        )

        total_assignments = expert_mask.sum().clamp_min(1.0)
        overflow_pct = (overflow_count.sum() / total_assignments).detach()
        aux_losses["overflow_pct"] = overflow_pct
        
        return output, aux_losses
    
    def _expert_parallel_forward(
        self,
        expert_tokens: torch.Tensor,
        expert_id: int
    ) -> torch.Tensor:
        logger.warning("Expert parallel processing not implemented, using standard processing")
        return self.experts[expert_id](expert_tokens)
    
    def _combine_expert_outputs(
        self,
        expert_outputs: List[Tuple[int, Optional[torch.Tensor], torch.Tensor, torch.Tensor]],
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        output_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = output_shape
        device = expert_weights.device
        
        output = torch.zeros(batch_size, seq_len, d_model, device=device)
        
        for expert_id, expert_output, expert_mask, token_indices in expert_outputs:
            if expert_output is None or token_indices.numel() == 0:
                continue
            matches = (expert_indices == expert_id).float()
            weight_per_token = (expert_weights * matches).sum(dim=-1)
            weights = weight_per_token[token_indices[:, 0], token_indices[:, 1]].unsqueeze(-1)
            contrib = expert_output * weights
            output[token_indices[:, 0], token_indices[:, 1]] += contrib
        
        return output


def get_moe_ffn(config) -> nn.Module:
    return MoEFFN(config)
