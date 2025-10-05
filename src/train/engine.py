
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..core.transformer import Transformer
from ..utils.logging import get_logger, log_metrics, log_training_metrics
from ..utils.checkpoints import save_checkpoint, load_checkpoint
from ..utils.distributed import get_rank, get_world_size, is_main_process
from ..utils.evals import evaluate_perplexity


logger = get_logger(__name__)


class TrainingEngine:
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        device: torch.device,
        config: Dict[str, Any]
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.epoch = 0
        self.step = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        self.metrics_history: List[Dict[str, Any]] = []
        
        self.use_amp = config.get("amp", "no") != "no"
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.grad_accum_steps = config.get("grad_accum", 1)
        self.grad_accum_count = 0
        
        self.grad_clip = config.get("grad_clip", 0.0)
        
        self.activation_checkpointing = config.get("activation_checkpointing", False)
        if self.activation_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.log_interval = config.get("log_interval", 10)
        self.eval_interval = config.get("eval_interval", 100)
        self.save_interval = config.get("save_interval", 500)
        
        self.checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
        self.max_checkpoints = config.get("max_checkpoints", 3)
        
        self.seed = config.get("seed", 42)
        self._set_seed()
    
    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = True
    ) -> Dict[str, Any]:
        self.model.train()
        
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.get("loss", 0.0)
        else:
            outputs = self.model(**batch)
            loss = outputs.get("loss", 0.0)
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.grad_accum_count += 1
        if self.grad_accum_count >= self.grad_accum_steps:
            if self.grad_clip > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.grad_accum_count = 0
            self.optimizer.zero_grad()
            
            self.step += 1
            self.global_step += 1
        
        results = {
            "loss": loss.item(),
            "step": self.step,
            "global_step": self.global_step,
            "epoch": self.epoch
        }
        
        if "aux_losses" in outputs:
            results.update(outputs["aux_losses"])
        
        return results
    
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self.model.eval()
        
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.get("loss", 0.0)
            else:
                outputs = self.model(**batch)
                loss = outputs.get("loss", 0.0)
        
        results = {
            "loss": loss.item(),
            "step": self.step,
            "global_step": self.global_step,
            "epoch": self.epoch
        }
        if "aux_losses" in outputs:
            results.update(outputs["aux_losses"])
        
        return results
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        self.epoch += 1
        epoch_start_time = time.time()
        
        train_results = self._train_epoch(train_loader)
        
        eval_results = {}
        if eval_loader is not None:
            eval_results = self._eval_epoch(eval_loader)
        
        epoch_time = time.time() - epoch_start_time
        epoch_results = {
            "epoch": self.epoch,
            "epoch_time": epoch_time,
            "train": train_results,
            "eval": eval_results
        }
        
        if is_main_process():
            self._log_epoch_metrics(epoch_results)
        
        if is_main_process() and self.epoch % self.save_interval == 0:
            self._save_checkpoint()
        
        return epoch_results
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, Any]:
        self.model.train()
        
        total_loss = 0.0
        total_steps = 0
        step_times = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            step_start_time = time.time()
            
            step_results = self.train_step(batch)
            
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            total_loss += step_results["loss"]
            total_steps += 1
            
            progress_bar.set_postfix({
                "loss": f"{step_results['loss']:.4f}",
                "avg_loss": f"{total_loss / total_steps:.4f}",
                "step_time": f"{step_time:.3f}s"
            })
            
            if self.step % self.log_interval == 0:
                self._log_step_metrics(step_results, step_time)
            
            if self.step % self.eval_interval == 0:
                self._eval_and_log()
        
        avg_loss = total_loss / total_steps
        avg_step_time = sum(step_times) / len(step_times)
        
        return {
            "loss": avg_loss,
            "step_time": avg_step_time,
            "total_steps": total_steps
        }
    
    def _eval_epoch(self, eval_loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                step_results = self.eval_step(batch)
                
                total_loss += step_results["loss"]
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        
        return {
            "loss": avg_loss,
            "total_steps": total_steps
        }
    
    def _log_step_metrics(self, step_results: Dict[str, Any], step_time: float) -> None:
        grad_norm = 0.0
        if self.grad_accum_count == 0:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
        
        lr = self.optimizer.param_groups[0]["lr"]
        
        memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        
        log_training_metrics(
            loss=step_results["loss"],
            learning_rate=lr,
            grad_norm=grad_norm,
            step=self.step
        )
        
        additional_metrics = {
            "step_time_ms": step_time * 1000,
            "memory_gb": memory_gb
        }
        
        for key, value in step_results.items():
            if key not in ["loss", "step", "global_step", "epoch"]:
                additional_metrics[key] = value
        
        log_metrics(additional_metrics, step=self.step, prefix="train")
    
    def _log_epoch_metrics(self, epoch_results: Dict[str, Any]) -> None:
        train_metrics = epoch_results["train"]
        log_metrics(train_metrics, step=self.epoch, prefix="epoch/train")
        
        if "eval" in epoch_results:
            eval_metrics = epoch_results["eval"]
            log_metrics(eval_metrics, step=self.epoch, prefix="epoch/eval")
        
        epoch_summary = {
            "epoch": epoch_results["epoch"],
            "epoch_time": epoch_results["epoch_time"],
            "train_loss": train_metrics["loss"],
            "eval_loss": epoch_results["eval"].get("loss", 0.0) if "eval" in epoch_results else 0.0
        }
        
        log_metrics(epoch_summary, step=self.epoch, prefix="epoch")
    
    def _eval_and_log(self) -> None:
        pass
    
    def _save_checkpoint(self) -> None:
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{self.epoch}_step_{self.step}.pt"
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.step,
            loss=self.best_loss,
            checkpoint_dir=self.checkpoint_dir,
            filename=f"checkpoint_epoch_{self.epoch}_step_{self.step}.pt"
        )
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_data, epoch, step, loss = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.epoch = epoch
        self.step = step
        self.best_loss = loss
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        num_epochs: int = 1
    ) -> Dict[str, Any]:
        logger.info(f"Starting training for {num_epochs} epochs")
        
        training_results = {
            "epochs": [],
            "best_loss": float('inf'),
            "total_time": 0.0
        }
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_results = self.train_epoch(train_loader, eval_loader)
            training_results["epochs"].append(epoch_results)
            current_loss = epoch_results["train"]["loss"]
            if current_loss < training_results["best_loss"]:
                training_results["best_loss"] = current_loss
                self.best_loss = current_loss
        
        training_results["total_time"] = time.time() - start_time
        
        logger.info(f"Training completed in {training_results['total_time']:.2f}s")
        logger.info(f"Best loss: {training_results['best_loss']:.4f}")
        
        return training_results


def get_training_engine(
    model: nn.Module,
    config: Dict[str, Any]
) -> TrainingEngine:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 0.01)
    )
    
    scheduler = None
    if config.get("lr_scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("max_steps", 1000)
        )
    
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    return engine
