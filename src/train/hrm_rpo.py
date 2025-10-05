
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ppo import PPOTrainer
from ..core.transformer import Transformer
from ..utils.logging import get_logger

logger = get_logger(__name__)


def nt_xent(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)
    logits = z_i @ z_j.t() / temperature
    labels = torch.arange(z_i.size(0), device=z_i.device)
    return F.cross_entropy(logits, labels)


class HRMRPOTrainer(PPOTrainer):

    def __init__(
        self,
        model: Transformer,
        reward_model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, reward_model, config, device)
        self.beta_halt = config.get("beta_halt", 0.05)
        self.beta_entropy = config.get("beta_entropy", 0.05)
        self.beta_mem = config.get("beta_mem", 0.05)
        self.mem_temperature = config.get("mem_temperature", 0.1)

    def _shape_rewards(
        self, base_rewards: torch.Tensor, controller_stats: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        halt_rate = controller_stats.get("halt_rate", torch.tensor(0.5, device=base_rewards.device))
        r_halt = -((halt_rate - 0.5) ** 2)

        route_entropy = controller_stats.get("route_entropy", torch.tensor(1.0, device=base_rewards.device))
        r_entropy = -((route_entropy - 1.0) ** 2)

        r_mem = torch.zeros_like(base_rewards)

        return base_rewards + self.beta_halt * r_halt + self.beta_entropy * r_entropy + self.beta_mem * r_mem

    def _memory_regularizer(self, z_h_a: torch.Tensor, z_h_b: torch.Tensor) -> torch.Tensor:
        return nt_xent(z_h_a, z_h_b, temperature=self.mem_temperature)

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base = super().compute_loss(outputs, rewards, advantages, returns, old_log_probs, attention_mask)

        ctrl_stats = {
            "halt_rate": torch.tensor(0.5, device=rewards.device),
            "route_entropy": torch.tensor(1.0, device=rewards.device),
        }
        shaped_rewards = self._shape_rewards(rewards, ctrl_stats)

        values = outputs["value"].detach()
        adv_shaped = shaped_rewards - values.squeeze(-1).mean(dim=-1, keepdim=False)
        adv_shaped = (adv_shaped - adv_shaped.mean()) / (adv_shaped.std() + 1e-8)
        base["total_loss"] = base["total_loss"] + 0.0 * adv_shaped.mean()

        hidden = outputs.get("logits").detach()
        z_h_a = hidden.mean(dim=1)[:, :64]
        z_h_b = torch.roll(z_h_a, shifts=1, dims=0)
        mem_loss = self._memory_regularizer(z_h_a, z_h_b)
        base["total_loss"] = base["total_loss"] + self.beta_mem * mem_loss
        base["mem_loss"] = mem_loss

        return base


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run HRM RPO training")
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()
    logger.info("HRM RPO trainer stub; integrate with PPO training loops as needed.")


if __name__ == "__main__":
    main()
