
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .engine import TrainingEngine, get_training_engine
from .rm import RewardModel
from ..core.transformer import Transformer, TransformerConfig
from ..utils.logging import get_logger, log_metrics, log_training_metrics
from ..utils.distributed import get_rank, get_world_size, is_main_process


logger = get_logger(__name__)


class GRPOTrainer:
    
    def __init__(
        self,
        model: Transformer,
        reward_model: RewardModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.reward_model = reward_model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.reward_model.to(self.device)
        self.engine = get_training_engine(self.model, config)
        self.grp_epochs = config.get("grp_epochs", 4)
        self.grp_clip_ratio = config.get("grp_clip_ratio", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.kl_coef = config.get("kl_coef", 0.1)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.gamma = config.get("gamma", 0.99)
        self.group_size = config.get("group_size", 8)
        self.normalize_rewards = config.get("normalize_rewards", True)
        self.reference_model = None
        self.metrics_history: List[Dict[str, Any]] = []
    
    def set_reference_model(self, reference_model: Transformer) -> None:
        self.reference_model = reference_model
        self.reference_model.to(self.device)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
    
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            reward_outputs = self.reward_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            rewards = reward_outputs["rewards"]
        
        return rewards
    
    def normalize_rewards_group(
        self,
        rewards: torch.Tensor,
        group_size: Optional[int] = None
    ) -> torch.Tensor:
        if not self.normalize_rewards:
            return rewards
        
        if group_size is None:
            group_size = self.group_size
        
        batch_size = rewards.size(0)
        num_groups = (batch_size + group_size - 1) // group_size
        
        normalized_rewards = rewards.clone()
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, batch_size)
            
            group_rewards = rewards[start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std()
            
            if group_std > 0:
                normalized_group = (group_rewards - group_mean) / group_std
                normalized_rewards[start_idx:end_idx] = normalized_group
        
        return normalized_rewards
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        for t in range(seq_len - 1, -1, -1):
            if t == seq_len - 1:
                next_value = 0.0
            else:
                next_value = values[:, t + 1]
            td_error = rewards[:, t] + self.gamma * next_value - values[:, t]
            if t == seq_len - 1:
                advantages[:, t] = td_error
            else:
                advantages[:, t] = td_error + self.gae_lambda * self.gamma * advantages[:, t + 1]
            returns[:, t] = rewards[:, t] + self.gamma * next_value
        
        return advantages, returns
    
    def compute_kl_divergence(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        kl_div = F.kl_div(
            F.log_softmax(logits, dim=-1),
            ref_probs,
            reduction="batchmean"
        )
        
        return kl_div
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        logits = outputs.get("logits")
        values = outputs.get("value") 
        if logits is None or values is None:
            raise ValueError("Model must output both logits and values")
        policy_loss = self._compute_policy_loss(
            logits, advantages, old_log_probs, attention_mask
        )
        losses["policy_loss"] = policy_loss
        value_loss = self._compute_value_loss(values, returns, attention_mask)
        losses["value_loss"] = value_loss * self.value_loss_coef
        entropy_loss = self._compute_entropy_loss(logits, attention_mask)
        losses["entropy_loss"] = entropy_loss * self.entropy_coef
        if self.reference_model is not None:
            kl_loss = self._compute_kl_loss(logits, attention_mask)
            losses["kl_loss"] = kl_loss * self.kl_coef
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss
        
        return losses
    
    def _compute_policy_loss(
        self,
        logits: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        current_log_probs = F.log_softmax(logits, dim=-1)
        ratio = torch.exp(current_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.grp_clip_ratio, 1 + self.grp_clip_ratio)
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return policy_loss
    
    def _compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        value_loss = F.mse_loss(values, returns)
        
        return value_loss
    
    def _compute_entropy_loss(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_loss = -entropy.mean()
        
        return entropy_loss
    
    def _compute_kl_loss(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.reference_model is None:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=logits,
                attention_mask=attention_mask
            )
            ref_logits = ref_outputs["logits"]
        kl_loss = self.compute_kl_divergence(logits, ref_logits)
        
        return kl_loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        old_log_probs = batch.get("old_log_probs")
        rewards = self.compute_rewards(input_ids, attention_mask)
        rewards = self.normalize_rewards_group(rewards)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_logits=True,
            return_value=True
        )
        logits = outputs["logits"]
        values = outputs["value"]
        advantages, returns = self.compute_advantages(rewards, values)
        losses = self.compute_loss(
            outputs, rewards, advantages, returns, old_log_probs, attention_mask
        )
        total_loss = losses["total_loss"]
        total_loss.backward()
        step_results = {
            "loss": total_loss.item(),
            "step": self.engine.step,
            "global_step": self.engine.global_step,
            "epoch": self.engine.epoch
        }
        for key, value in losses.items():
            if key != "total_loss":
                step_results[key] = value.item()
        step_results["reward"] = rewards.mean().item()
        step_results["advantage"] = advantages.mean().item()
        step_results["return"] = returns.mean().item()
        
        return step_results
    
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            rewards = self.compute_rewards(input_ids, attention_mask)
            rewards = self.normalize_rewards_group(rewards)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_logits=True,
                return_value=True
            )
            logits = outputs["logits"]
            values = outputs["value"]
            advantages, returns = self.compute_advantages(rewards, values)
            kl_div = 0.0
            if self.reference_model is not None:
                kl_div = self.compute_kl_divergence(logits, logits).item()
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            step_results = {
                "reward": rewards.mean().item(),
                "advantage": advantages.mean().item(),
                "return": returns.mean().item(),
                "kl_divergence": kl_div,
                "entropy": entropy,
                "step": self.engine.step,
                "global_step": self.engine.global_step,
                "epoch": self.engine.epoch
            }
            
            return step_results
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 1
    ) -> Dict[str, Any]:
        logger.info(f"Starting GRPO training for {num_epochs} epochs")
        original_train_step = self.engine.train_step
        self.engine.train_step = self.train_step
        original_eval_step = self.engine.eval_step
        self.engine.eval_step = self.eval_step
        
        try:
            training_results = self.engine.train(train_loader, eval_loader, num_epochs)
            training_results["grpo_metrics"] = {
                "grp_epochs": self.grp_epochs,
                "grp_clip_ratio": self.grp_clip_ratio,
                "value_loss_coef": self.value_loss_coef,
                "entropy_coef": self.entropy_coef,
                "kl_coef": self.kl_coef,
                "gae_lambda": self.gae_lambda,
                "gamma": self.gamma,
                "group_size": self.group_size,
                "normalize_rewards": self.normalize_rewards
            }
            
            return training_results
            
        finally:
            self.engine.train_step = original_train_step
            self.engine.eval_step = original_eval_step
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        logger.info("Starting GRPO evaluation")
        eval_results = self.engine._eval_epoch(eval_loader)
        logger.info(f"GRPO evaluation completed: {eval_results}")
        return eval_results
    
    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        logger.info(f"Saved GRPO model to {path}")
    
    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded GRPO model from {path}")


def train_grpo(
    model: Transformer,
    reward_model: RewardModel,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
    num_epochs: int = 1
) -> Dict[str, Any]:
    if config is None:
        config = {}
    trainer = GRPOTrainer(model, reward_model, config)
    results = trainer.train(train_loader, eval_loader, num_epochs)
    return results
