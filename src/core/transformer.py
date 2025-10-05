
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from .attention import get_attention
from .ffn import get_ffn
from .norms import get_norm
from .rope_kv import RoPE, KVCache
from .moe import get_moe_ffn
from .hrm_controller import get_hrm_controller
from .heads import get_lm_head, get_value_head
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TransformerConfig:
    
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    vocab_size: int = 32000
    seq_len: int = 4096
    
    rope_theta: float = 100000.0
    
    attn_impl: str = "torch"
    norm_impl: str = "torch"
    
    moe_enable: bool = False
    hrm_enable: bool = False
    
    tie_embeddings: bool = True
    sliding_window: Optional[int] = None
    
    n_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25
    dropless: bool = False
    aux_loss_coef: float = 0.01
    expert_parallel: bool = False
    
    max_steps: int = 8
    halt_penalty: float = 0.1
    sink_strength_init: float = 0.5
    temp_bounds: Tuple[float, float] = (0.1, 2.0)
    
    dropout: float = 0.1
    use_flash_attn: bool = False
    causal: bool = True
    
    norm_eps: float = 1e-6
    elementwise_affine: bool = True
    
    hidden_mult: int = 4
    activation: str = "swiglu"
    
    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        if self.attn_impl not in ["torch", "triton"]:
            raise ValueError(f"attn_impl must be 'torch' or 'triton', got {self.attn_impl}")
        
        if self.norm_impl not in ["torch", "triton"]:
            raise ValueError(f"norm_impl must be 'torch' or 'triton', got {self.norm_impl}")
        
        if self.activation not in ["swiglu", "gelu", "relu"]:
            raise ValueError(f"activation must be 'swiglu', 'gelu', or 'relu', got {self.activation}")


class TransformerBlock(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.attention = get_attention(config)
        
        if config.moe_enable:
            self.ffn = get_moe_ffn(config)
        else:
            self.ffn = get_ffn(config)
        
        self.norm1 = get_norm(config)
        self.norm2 = get_norm(config)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.hrm_enable:
            self.hrm_controller = get_hrm_controller(config)
        else:
            self.hrm_controller = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        rope_state: Optional[torch.Tensor] = None,
        sliding_window: Optional[int] = None,
        sink_cfg: Optional[Dict[str, Any]] = None,
        router_temp: Optional[float] = None,
        hrm_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Optional[KVCache], Dict[str, Any]]:
        aux_losses = {}
        
        if self.hrm_controller is not None and hrm_state is not None:
            x, ctrl = self.hrm_controller.pre_block(x, hrm_state)
            sink_cfg = ctrl.get("sink_cfg", sink_cfg)
            router_temp = ctrl.get("router_temp", router_temp)
        
        residual = x
        x = self.norm1(x)
        
        attn_out, new_kv_cache = self.attention(
            x,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            rope_state=rope_state,
            sliding_window=sliding_window,
            sink_cfg=sink_cfg
        )
        
        x = residual + self.dropout(attn_out)
        
        residual = x
        x = self.norm2(x)
        
        if self.config.moe_enable:
            ffn_out, moe_aux_losses = self.ffn(x, router_temp=router_temp)
            aux_losses.update(moe_aux_losses)
        else:
            ffn_out = self.ffn(x)
        
        x = residual + self.dropout(ffn_out)
        
        if self.hrm_controller is not None and hrm_state is not None:
            x, hrm_state, halt_loss = self.hrm_controller.post_block(x, hrm_state)
            aux_losses["halt_loss"] = halt_loss
        
        return x, new_kv_cache, aux_losses


class Transformer(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = get_norm(config)
        
        self.lm_head = get_lm_head(config)
        
        self.value_head = get_value_head(config)
        
        self.rope = RoPE(config.d_model // config.n_heads, config.rope_theta)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        return_logits: bool = True,
        return_value: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if kv_cache is not None:
            position_ids = torch.arange(
                kv_cache.seq_len, kv_cache.seq_len + seq_len,
                device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(position_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        rope_state = self.rope.get_rope_state(position_ids)
        
        sliding_window = self.config.sliding_window
        
        hrm_state = None
        if self.config.hrm_enable:
            hrm_state = {
                "z_h": torch.ones(batch_size, seq_len, device=device) * self.config.sink_strength_init,
                "z_l": torch.ones(batch_size, seq_len, device=device) * 0.1,
                "steps": torch.zeros(batch_size, seq_len, device=device, dtype=torch.long),
                "halted": torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
            }
        
        aux_losses = {}
        for block in self.blocks:
            x, kv_cache, block_aux_losses = block(
                x,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                rope_state=rope_state,
                sliding_window=sliding_window,
                hrm_state=hrm_state
            )
            aux_losses.update(block_aux_losses)
        
        x = self.norm(x)
        
        outputs = {}
        
        if return_logits:
            logits = self.lm_head(x)
            outputs["logits"] = logits
        
        if return_value:
            value = self.value_head(x)
            outputs["value"] = value
        
        if aux_losses:
            outputs["aux_losses"] = aux_losses
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        **kwargs
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        kv_cache = KVCache(
            batch_size=batch_size,
            n_heads=self.config.n_heads,
            head_dim=self.config.d_model // self.config.n_heads,
            seq_len=0,
            device=device
        )
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(
                    generated,
                    kv_cache=kv_cache,
                    return_logits=True,
                    return_value=False
                )
                
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if (next_token == eos_token_id).all():
                    break
        
        return generated
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        trust_remote_code: bool = False,
        **kwargs
    ) -> Transformer:
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        config = TransformerConfig(
            d_model=model.config.hidden_size,
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            vocab_size=model.config.vocab_size,
            seq_len=model.config.max_position_embeddings,
            rope_theta=getattr(model.config, 'rope_theta', 10000.0),
            attn_impl="torch",
            norm_impl="torch"
        )
        
        transformer = cls(config)
        
        transformer.load_state_dict(model.state_dict(), strict=False)
        
        return transformer
    
    def save_pretrained(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        
        import json
        config_dict = {
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "vocab_size": self.config.vocab_size,
            "seq_len": self.config.seq_len,
            "rope_theta": self.config.rope_theta,
            "attn_impl": self.config.attn_impl,
            "norm_impl": self.config.norm_impl,
            "moe_enable": self.config.moe_enable,
            "hrm_enable": self.config.hrm_enable,
            "tie_embeddings": self.config.tie_embeddings,
            "sliding_window": self.config.sliding_window,
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved model to {path}")
