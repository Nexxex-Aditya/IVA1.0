"""HRM - halting and reasoning module controller"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class HRMState:
    
    z_h: torch.Tensor  # high level reasoning strength
    z_l: torch.Tensor  # low level reasoning strength
    steps: torch.Tensor  # number of steps taken
    halted: torch.Tensor  # whether token is halted
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "z_h": self.z_h,
            "z_l": self.z_l,
            "steps": self.steps,
            "halted": self.halted
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, torch.Tensor]) -> HRMState:
        return cls(
            z_h=state_dict["z_h"],
            z_l=state_dict["z_l"],
            steps=state_dict["steps"],
            halted=state_dict["halted"]
        )


class HRMController(nn.Module):
    """HRM controller for adaptive computation time.

    Implementing ACT style halting with Bernoulli halting probabilities p_h = sigmoid(w^T h + b).
    Maintains per token cumulative halt mass and halts tokens when mass reaches 1.0 or max_steps.
    Also modulates attention sink and MoE router temperature based on controller state.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_steps = config.max_steps
        self.halt_penalty = config.halt_penalty
        self.sink_strength_init = config.sink_strength_init
        self.temp_bounds = config.temp_bounds
        self.z_h_init = getattr(config, 'z_h_init', 1.0)
        self.z_l_init = getattr(config, 'z_l_init', 0.1)
        
        self.halt_threshold = 1.0  # cumulative mass threshold for ACT
        self.min_steps = 1
        self.entropy_weight = 0.01

        # ACT state buffers - tracked per sequence during a block recurrence
        self.register_buffer("eps", torch.tensor(1e-6), persistent=False)
        
        # sink strength controller
        self.sink_strength = nn.Parameter(torch.tensor(self.sink_strength_init))
        
        # router temperature controller
        self.router_temp = nn.Parameter(torch.tensor(1.0))
        
        # halt probability predictor - per token state -> scalar prob
        self.halt_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # reasoning strength predictor
        self.reasoning_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 2),  # z_h and z_l
            nn.Softmax(dim=-1)
        )
    
    def pre_block(
        self,
        x: torch.Tensor,
        hrm_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # predict halting probability per token
        halt_probs = self.halt_predictor(x).squeeze(-1)
        
        # predict reasoning strengths
        reasoning_strengths = self.reasoning_predictor(x)
        z_h_pred = reasoning_strengths[:, :, 0]
        z_l_pred = reasoning_strengths[:, :, 1]
        
        # update HRM state
        hrm_state["z_h"] = z_h_pred
        hrm_state["z_l"] = z_l_pred
        
        if "cum_mass" not in hrm_state:
            hrm_state["cum_mass"] = torch.zeros(batch_size, seq_len, device=device)
        if "rema" not in hrm_state:
            hrm_state["rema"] = torch.ones(batch_size, seq_len, device=device)

        still_running = (~hrm_state["halted"]).float()  # ACT accumulation only for non halted tokens
        new_mass = halt_probs * hrm_state["rema"] * still_running
        hrm_state["cum_mass"] = hrm_state["cum_mass"] + new_mass
        hrm_state["rema"] = hrm_state["rema"] * (1.0 - halt_probs)

        sink_cfg = self._compute_sink_config(hrm_state)
        
        router_temp = self._compute_router_temp(hrm_state)
        
        x_modified = x * hrm_state["z_h"].unsqueeze(-1)
        
        control_signals = {
            "sink_cfg": sink_cfg,
            "router_temp": router_temp,
            "halt_probs": halt_probs,
            "reasoning_strengths": reasoning_strengths
        }
        
        return x_modified, control_signals
    
    def post_block(
        self,
        x: torch.Tensor,
        hrm_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        hrm_state["steps"] = hrm_state["steps"] + 1

        reached_mass = (hrm_state.get("cum_mass", torch.zeros_like(hrm_state["z_h"])) >= self.halt_threshold)
        reached_max_steps = (hrm_state["steps"] >= self.max_steps)
        halt_condition = (reached_mass | reached_max_steps) & (~hrm_state["halted"]) & (hrm_state["steps"] >= self.min_steps)

        hrm_state["halted"] = hrm_state["halted"] | halt_condition
        
        halt_loss = self._compute_halt_loss(hrm_state)
        
        x_modified = x * hrm_state["z_l"].unsqueeze(-1)
        
        return x_modified, hrm_state, halt_loss
    
    def _compute_sink_config(
        self,
        hrm_state: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        sink_strength = self.sink_strength * hrm_state["z_h"].mean()
        
        sink_strength = torch.clamp(
            sink_strength,
            min=self.temp_bounds[0],
            max=self.temp_bounds[1]
        )
        
        return {
            "sink_tokens": 4,
            "sink_strength": sink_strength.item(),
            "decay_rate": 0.95,
            "min_strength": self.temp_bounds[0],
            "max_strength": self.temp_bounds[1]
        }
    
    def _compute_router_temp(
        self,
        hrm_state: Dict[str, torch.Tensor]
    ) -> float:
        router_temp = self.router_temp * hrm_state["z_l"].mean()
        
        router_temp = torch.clamp(
            router_temp,
            min=self.temp_bounds[0],
            max=self.temp_bounds[1]
        )
        
        return router_temp.item()
    
    def _compute_halt_loss(
        self,
        hrm_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        cum_mass = hrm_state.get("cum_mass", torch.zeros_like(hrm_state["z_h"]))
        rema = hrm_state.get("rema", torch.ones_like(hrm_state["z_h"]))
        exp_steps_penalty = (1.0 - rema).mean()

        z_h = hrm_state["z_h"]
        z_l = hrm_state["z_l"]
        entropy = -(z_h * torch.log(z_h + 1e-8) + z_l * torch.log(z_l + 1e-8))
        entropy_penalty = entropy.mean()
        
        halt_loss = (exp_steps_penalty * self.halt_penalty + entropy_penalty * self.entropy_weight)
        
        return halt_loss
    
    def get_hrm_metrics(self, hrm_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        return {
            "halt_rate": hrm_state["halted"].float().mean().item(),
            "avg_steps": hrm_state["steps"].float().mean().item(),
            "avg_z_h": hrm_state["z_h"].mean().item(),
            "avg_z_l": hrm_state["z_l"].mean().item(),
            "sink_strength": self.sink_strength.item(),
            "router_temp": self.router_temp.item()
        }
    
    def reset_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "z_h": torch.full((batch_size, seq_len), self.z_h_init, device=device),
            "z_l": torch.full((batch_size, seq_len), self.z_l_init, device=device),
            "steps": torch.zeros((batch_size, seq_len), device=device, dtype=torch.long),
            "halted": torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool),
            "cum_mass": torch.zeros((batch_size, seq_len), device=device),
            "rema": torch.ones((batch_size, seq_len), device=device),
        }


def get_hrm_controller(config) -> nn.Module:
    return HRMController(config)
