
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        tensorboard: bool = True,
        jsonl: bool = True,
        console: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard = tensorboard
        self.jsonl = jsonl
        self.console = console
        self._setup_tensorboard()
        self._setup_jsonl()
        self._setup_console()
        self.metrics_history: Dict[str, list] = {}
        self.step = 0
        self.start_time = time.time()
    
    def _setup_tensorboard(self) -> None:
        if self.tensorboard:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(tb_dir)
        else:
            self.tb_writer = None
    
    def _setup_jsonl(self) -> None:
        if self.jsonl:
            self.jsonl_file = self.log_dir / "metrics.jsonl"
            self.jsonl_file.touch()
        else:
            self.jsonl_file = None
    
    def _setup_console(self) -> None:
        if self.console:
            self.console_logger = logging.getLogger("metrics")
            self.console_logger.setLevel(logging.INFO)
            
            if not self.console_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.console_logger.addHandler(handler)
        else:
            self.console_logger = None
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        if step is None:
            step = self.step
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
                elif isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.tb_writer.add_scalar(key, value.item(), step)
        
        if self.jsonl_file:
            log_entry = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            with open(self.jsonl_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        if self.console_logger:
            metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
            self.console_logger.info(f"Step {step}: {metrics_str}")
    
    def log_scalar(self, name: str, value: Union[float, int], step: Optional[int] = None) -> None:
        self.log_metrics({name: value}, step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None) -> None:
        if self.tb_writer and step is not None:
            self.tb_writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image: torch.Tensor, step: Optional[int] = None) -> None:
        if self.tb_writer and step is not None:
            self.tb_writer.add_image(name, image, step)
    
    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        if self.tb_writer and step is not None:
            self.tb_writer.add_text(name, text, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        if self.tb_writer:
            self.tb_writer.add_graph(model, input_tensor)
    
    def close(self) -> None:
        if self.tb_writer:
            self.tb_writer.close()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        summary = {}
        for key, values in self.metrics_history.items():
            if values:
                summary[key] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1]
                }
        return summary
    
    def increment_step(self) -> None:
        self.step += 1
    
    def set_step(self, step: int) -> None:
        self.step = step


def setup_logging(
    log_dir: Union[str, Path],
    log_level: str = "INFO",
    tensorboard: bool = True,
    jsonl: bool = True,
    console: bool = True
) -> MetricsLogger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    metrics_logger = MetricsLogger(
        log_dir=log_dir,
        tensorboard=tensorboard,
        jsonl=jsonl,
        console=console
    )
    
    return metrics_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = ""
) -> None:
    logger = get_logger("metrics")
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    logger.info(f"Step {step}: {metrics_str}")


def log_speed_metrics(
    tokens_per_sec: float,
    step_time_ms: float,
    step: Optional[int] = None
) -> None:
    metrics = {
        "speed.tokens_per_sec": tokens_per_sec,
        "speed.step_time_ms": step_time_ms
    }
    log_metrics(metrics, step, "speed")


def log_memory_metrics(
    vram_gb: float,
    step: Optional[int] = None
) -> None:
    metrics = {
        "mem.vram_gb": vram_gb
    }
    log_metrics(metrics, step, "mem")


def log_quality_metrics(
    ppl: float,
    eval_passk: float,
    step: Optional[int] = None
) -> None:
    metrics = {
        "quality.ppl": ppl,
        "eval.passk": eval_passk
    }
    log_metrics(metrics, step, "quality")


def log_hrm_metrics(
    halt_rate: float,
    avg_steps: float,
    step: Optional[int] = None
) -> None:
    metrics = {
        "hrm.halt_rate": halt_rate,
        "hrm.avg_steps": avg_steps
    }
    log_metrics(metrics, step, "hrm")


def log_moe_metrics(
    load_per_expert: list[float],
    overflow_pct: float,
    route_entropy: float,
    step: Optional[int] = None
) -> None:
    metrics = {
        "moe.load_per_expert": load_per_expert,
        "moe.overflow_pct": overflow_pct,
        "moe.route_entropy": route_entropy
    }
    log_metrics(metrics, step, "moe")


def log_quant_metrics(
    fp8_amax_drift: float,
    step: Optional[int] = None
) -> None:
    metrics = {
        "quant.fp8.amax_drift": fp8_amax_drift
    }
    log_metrics(metrics, step, "quant")


def log_training_metrics(
    loss: float,
    learning_rate: float,
    grad_norm: float,
    step: Optional[int] = None
) -> None:
    metrics = {
        "train.loss": loss,
        "train.learning_rate": learning_rate,
        "train.grad_norm": grad_norm
    }
    log_metrics(metrics, step, "train")


def log_evaluation_metrics(
    eval_loss: float,
    eval_ppl: float,
    eval_accuracy: float,
    step: Optional[int] = None
) -> None:
    metrics = {
        "eval.loss": eval_loss,
        "eval.ppl": eval_ppl,
        "eval.accuracy": eval_accuracy
    }
    log_metrics(metrics, step, "eval")
