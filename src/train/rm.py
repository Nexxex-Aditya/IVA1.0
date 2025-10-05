
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .engine import TrainingEngine, get_training_engine
from ..core.transformer import Transformer, TransformerConfig
from ..utils.logging import get_logger, log_metrics, log_training_metrics
from ..utils.distributed import get_rank, get_world_size, is_main_process


logger = get_logger(__name__)


class RewardModel(nn.Module):
    
    def __init__(
        self,
        base_model: Transformer,
        reward_head_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        self.reward_head_dim = reward_head_dim
        self.dropout = dropout
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.config.d_model, reward_head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reward_head_dim, 1)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_logits=False,
            return_value=False,
            **kwargs
        )
        hidden_states = base_outputs.get("hidden_states", base_outputs.get("last_hidden_state"))
        
        if hidden_states is None:
            if "logits" in base_outputs:
                hidden_states = base_outputs["logits"]
            else:
                raise ValueError("No hidden states available from base model")
        rewards = self.reward_head(hidden_states)
        
        return {
            "rewards": rewards,
            "hidden_states": hidden_states
        }


class RMTrainer:
    
    def __init__(
        self,
        model: RewardModel,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.engine = get_training_engine(self.model, config)
        self.reward_loss_weight = config.get("reward_loss_weight", 1.0)
        self.kl_loss_weight = config.get("kl_loss_weight", 0.01)
        self.entropy_loss_weight = config.get("entropy_loss_weight", 0.01)
        self.metrics_history: List[Dict[str, Any]] = []
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        if "rewards" in outputs:
            rewards = outputs["rewards"]
            reward_loss = self._compute_reward_loss(rewards, labels)
            losses["reward_loss"] = reward_loss * self.reward_loss_weight
        if "kl_loss" in outputs:
            kl_loss = outputs["kl_loss"]
            losses["kl_loss"] = kl_loss * self.kl_loss_weight
        if "entropy_loss" in outputs:
            entropy_loss = outputs["entropy_loss"]
            losses["entropy_loss"] = entropy_loss * self.entropy_loss_weight
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss
        
        return losses
    
    def _compute_reward_loss(
        self,
        rewards: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        batch_size = rewards.size(0)
        if rewards.size(1) == 2:
            rewards_chosen = rewards[:, 0]
            rewards_rejected = rewards[:, 1]
        else:
            rewards_chosen = rewards[::2]
            rewards_rejected = rewards[1::2]
        log_prob_chosen = F.logsigmoid(rewards_chosen)
        log_prob_rejected = F.logsigmoid(rewards_rejected)
        loss = -log_prob_chosen.mean() - log_prob_rejected.mean()
        
        return loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        outputs = self.model(**batch)
        losses = self.compute_loss(
            outputs,
            batch["labels"],
            batch.get("attention_mask")
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
        
        return step_results
    
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        with torch.no_grad():
            outputs = self.model(**batch)
            losses = self.compute_loss(
                outputs,
                batch["labels"],
                batch.get("attention_mask")
            )
            accuracy = self._compute_accuracy(outputs, batch["labels"])
            step_results = {
                "loss": losses["total_loss"].item(),
                "accuracy": accuracy,
                "step": self.engine.step,
                "global_step": self.engine.global_step,
                "epoch": self.engine.epoch
            }
            for key, value in losses.items():
                if key != "total_loss":
                    step_results[key] = value.item()
            
            return step_results
    
    def _compute_accuracy(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> float:
        if "rewards" not in outputs:
            return 0.0
        
        rewards = outputs["rewards"]
        if rewards.size(1) == 2:
            rewards_chosen = rewards[:, 0]
            rewards_rejected = rewards[:, 1]
        else:
            rewards_chosen = rewards[::2]
            rewards_rejected = rewards[1::2]
        predictions = (rewards_chosen > rewards_rejected).float()
        accuracy = predictions.mean().item()
        
        return accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 1
    ) -> Dict[str, Any]:
        logger.info(f"Starting RM training for {num_epochs} epochs")
        original_train_step = self.engine.train_step
        self.engine.train_step = self.train_step
        original_eval_step = self.engine.eval_step
        self.engine.eval_step = self.eval_step
        
        try:
            training_results = self.engine.train(train_loader, eval_loader, num_epochs)
            training_results["rm_metrics"] = {
                "reward_loss_weight": self.reward_loss_weight,
                "kl_loss_weight": self.kl_loss_weight,
                "entropy_loss_weight": self.entropy_loss_weight
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
        logger.info("Starting RM evaluation")
        eval_results = self.engine._eval_epoch(eval_loader)
        logger.info(f"RM evaluation completed: {eval_results}")
        return eval_results
    
    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved RM model to {path}")
    
    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded RM model from {path}")


def train_rm(
    model: RewardModel,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
    num_epochs: int = 1
) -> Dict[str, Any]:
    if config is None:
        config = {}
    trainer = RMTrainer(model, config)
    results = trainer.train(train_loader, eval_loader, num_epochs)
    
    return results
