
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .logging import get_logger


logger = get_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    checkpoint_dir: Union[str, Path],
    filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    save_optimizer: bool = True,
    save_scheduler: bool = True
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    
    checkpoint_path = checkpoint_dir / filename
    checkpoint_data = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "model_config": getattr(model, "config", None),
    }
    
    if save_optimizer:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint_data["optimizer_config"] = {
            "type": type(optimizer).__name__,
            "lr": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
        }
    
    if save_scheduler and scheduler is not None:
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        checkpoint_data["scheduler_config"] = {
            "type": type(scheduler).__name__,
        }
    
    if metadata:
        checkpoint_data["metadata"] = metadata
    torch.save(checkpoint_data, checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump({
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "filename": filename,
            "metadata": metadata or {}
        }, f, indent=2)
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Tuple[Dict[str, Any], int, int, float]:
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    if model is not None:
        if "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"], strict=strict)
        else:
            logger.warning("No model state dict found in checkpoint")
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    elif optimizer is not None:
        logger.warning("No optimizer state dict found in checkpoint")
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
    elif scheduler is not None:
        logger.warning("No scheduler state dict found in checkpoint")
    
    epoch = checkpoint_data.get("epoch", 0)
    step = checkpoint_data.get("step", 0)
    loss = checkpoint_data.get("loss", float("inf"))
    
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch={epoch}, step={step}, loss={loss})")
    
    return checkpoint_data, epoch, step, loss


def load_checkpoint_metadata(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".json")
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    else:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        return {
            "epoch": checkpoint_data.get("epoch", 0),
            "step": checkpoint_data.get("step", 0),
            "loss": checkpoint_data.get("loss", float("inf")),
            "filename": checkpoint_path.name,
            "metadata": checkpoint_data.get("metadata", {})
        }


def rotate_checkpoints(
    checkpoint_dir: Union[str, Path],
    max_checkpoints: int = 3,
    keep_best: bool = True
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    checkpoint_files = []
    for file_path in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            metadata = load_checkpoint_metadata(file_path)
            checkpoint_files.append((file_path, metadata))
        except Exception as e:
            logger.warning(f"Failed to load metadata for {file_path}: {e}")
            continue
    
    if not checkpoint_files:
        return
    checkpoint_files.sort(key=lambda x: x[1]["step"], reverse=True)
    if keep_best and len(checkpoint_files) > 1:
        best_checkpoint = min(checkpoint_files, key=lambda x: x[1]["loss"])
        checkpoint_files = [best_checkpoint] + [f for f in checkpoint_files if f != best_checkpoint]
    
    for checkpoint_file, _ in checkpoint_files[max_checkpoints:]:
        try:
            checkpoint_file.unlink()
            metadata_file = checkpoint_file.with_suffix(".json")
            if metadata_file.exists():
                metadata_file.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = []
    for file_path in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            metadata = load_checkpoint_metadata(file_path)
            checkpoint_files.append((file_path, metadata))
        except Exception:
            continue
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key=lambda x: x[1]["step"], reverse=True)
    
    return checkpoint_files[0][0]


def find_best_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = []
    for file_path in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            metadata = load_checkpoint_metadata(file_path)
            checkpoint_files.append((file_path, metadata))
        except Exception:
            continue
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key=lambda x: x[1]["loss"])
    
    return checkpoint_files[0][0]


def list_checkpoints(checkpoint_dir: Union[str, Path]) -> list[Dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for file_path in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            metadata = load_checkpoint_metadata(file_path)
            metadata["path"] = str(file_path)
            checkpoints.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {file_path}: {e}")
            continue
    
    checkpoints.sort(key=lambda x: x["step"], reverse=True)
    
    return checkpoints


def copy_checkpoint(
    source_path: Union[str, Path],
    dest_path: Union[str, Path]
) -> Path:
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_path}")
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    source_metadata = source_path.with_suffix(".json")
    dest_metadata = dest_path.with_suffix(".json")
    if source_metadata.exists():
        shutil.copy2(source_metadata, dest_metadata)
    
    logger.info(f"Copied checkpoint: {source_path} -> {dest_path}")
    return dest_path


def get_checkpoint_size(checkpoint_path: Union[str, Path]) -> int:
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return 0
    
    return checkpoint_path.stat().st_size


def cleanup_checkpoints(
    checkpoint_dir: Union[str, Path],
    max_age_days: int = 7,
    max_size_gb: float = 10.0
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    import time
    from datetime import datetime, timedelta
    
    age_threshold = time.time() - (max_age_days * 24 * 60 * 60)
    size_threshold = max_size_gb * 1024 * 1024 * 1024
    
    checkpoint_files = []
    total_size = 0
    
    for file_path in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            metadata = load_checkpoint_metadata(file_path)
            file_size = get_checkpoint_size(file_path)
            file_age = file_path.stat().st_mtime
            
            checkpoint_files.append((file_path, metadata, file_size, file_age))
            total_size += file_size
        except Exception as e:
            logger.warning(f"Failed to process checkpoint {file_path}: {e}")
            continue
    
    checkpoint_files.sort(key=lambda x: x[1]["step"])
    for checkpoint_file, metadata, file_size, file_age in checkpoint_files:
        if file_age < age_threshold:
            try:
                checkpoint_file.unlink()
                metadata_file = checkpoint_file.with_suffix(".json")
                if metadata_file.exists():
                    metadata_file.unlink()
                total_size -= file_size
                logger.info(f"Removed old checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")
    
    if total_size > size_threshold:
        checkpoint_files = [(f, m, s, a) for f, m, s, a in checkpoint_files if f.exists()]
        checkpoint_files.sort(key=lambda x: x[1]["step"])
        
        for checkpoint_file, metadata, file_size, file_age in checkpoint_files:
            if total_size <= size_threshold:
                break
            
            try:
                checkpoint_file.unlink()
                metadata_file = checkpoint_file.with_suffix(".json")
                if metadata_file.exists():
                    metadata_file.unlink()
                total_size -= file_size
                logger.info(f"Removed checkpoint due to size limit: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")
