
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
from ..utils.evals import evaluate_perplexity
from ..utils.distributed import get_rank, get_world_size, is_main_process


logger = get_logger(__name__)


class SFTTrainer:
    
    def __init__(
        self,
        model: Transformer,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.engine = get_training_engine(self.model, config)
        self.ce_weight = config.get("ce_weight", 1.0)
        self.kd_weight = config.get("kd_weight", 0.0)
        self.kd_temperature = config.get("kd_temperature", 3.0)
        self.hrm_loss_weight = config.get("hrm_loss_weight", 1.0)
        self.halt_penalty = config.get("halt_penalty", 0.1)
        self.sink_loss_weight = config.get("sink_loss_weight", 0.1)
        self.moe_aux_loss_weight = config.get("moe_aux_loss_weight", 0.01)
        self.router_kl_weight = config.get("router_kl_weight", 0.01)
        self.kd_enable = config.get("kd_enable", False)
        self.teacher_model = None
        self.metrics_history: List[Dict[str, Any]] = []
    
    def set_teacher_model(self, teacher_model: Transformer) -> None:
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        if "logits" in outputs:
            logits = outputs["logits"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            losses["ce_loss"] = ce_loss * self.ce_weight
        if self.kd_enable and self.teacher_model is not None:
            kd_loss = self._compute_kd_loss(outputs)
            losses["kd_loss"] = kd_loss * self.kd_weight
        if "aux_losses" in outputs:
            aux_losses = outputs["aux_losses"]
            if "halt_loss" in aux_losses:
                halt_loss = aux_losses["halt_loss"]
                losses["halt_loss"] = halt_loss * self.hrm_loss_weight
            if "sink_loss" in aux_losses:
                sink_loss = aux_losses["sink_loss"]
                losses["sink_loss"] = sink_loss * self.sink_loss_weight
            if "load_balance_loss" in aux_losses:
                moe_loss = aux_losses["load_balance_loss"]
                losses["moe_aux_loss"] = moe_loss * self.moe_aux_loss_weight
            if "router_kl_loss" in aux_losses:
                router_kl_loss = aux_losses["router_kl_loss"]
                losses["router_kl_loss"] = router_kl_loss * self.router_kl_weight
        
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss
        
        return losses
    
    def _compute_kd_loss(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "logits" not in outputs:
            return torch.tensor(0.0, device=self.device)
        student_logits = outputs["logits"]
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=outputs.get("input_ids"),
                attention_mask=outputs.get("attention_mask")
            )
            teacher_logits = teacher_outputs["logits"]
        student_probs = F.log_softmax(student_logits / self.kd_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.kd_temperature, dim=-1)
        
        kd_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction="batchmean"
        ) * (self.kd_temperature ** 2)
        
        return kd_loss
    
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
            perplexity = self._compute_perplexity(outputs, batch["labels"])
            step_results = {
                "loss": losses["total_loss"].item(),
                "perplexity": perplexity,
                "step": self.engine.step,
                "global_step": self.engine.global_step,
                "epoch": self.engine.epoch
            }
            for key, value in losses.items():
                if key != "total_loss":
                    step_results[key] = value.item()
            
            return step_results
    
    def _compute_perplexity(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> float:
        if "logits" not in outputs:
            return float('inf')
        
        logits = outputs["logits"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )
        perplexity = math.exp(ce_loss.item())
        
        return perplexity
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 1
    ) -> Dict[str, Any]:
        logger.info(f"Starting SFT training for {num_epochs} epochs")
        original_train_step = self.engine.train_step
        self.engine.train_step = self.train_step
        original_eval_step = self.engine.eval_step
        self.engine.eval_step = self.eval_step
        
        try:
            training_results = self.engine.train(train_loader, eval_loader, num_epochs)
            training_results["sft_metrics"] = {
                "ce_weight": self.ce_weight,
                "kd_weight": self.kd_weight,
                "kd_temperature": self.kd_temperature,
                "hrm_loss_weight": self.hrm_loss_weight,
                "halt_penalty": self.halt_penalty,
                "sink_loss_weight": self.sink_loss_weight,
                "moe_aux_loss_weight": self.moe_aux_loss_weight,
                "router_kl_weight": self.router_kl_weight
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
        logger.info("Starting SFT evaluation")
        eval_results = self.engine._eval_epoch(eval_loader)
        perplexity_results = evaluate_perplexity(
            self.model,
            eval_loader,
            self.device,
            max_batches=max_batches
        )
        combined_results = {
            **eval_results,
            **perplexity_results
        }
        
        logger.info(f"SFT evaluation completed: {combined_results}")
        
        return combined_results
    
    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        logger.info(f"Saved SFT model to {path}")
    
    def load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded SFT model from {path}")


def train_sft(
    model: Transformer,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
    num_epochs: int = 1
) -> Dict[str, Any]:
    if config is None:
        config = {}
    trainer = SFTTrainer(model, config)
    results = trainer.train(train_loader, eval_loader, num_epochs)  
    return results
