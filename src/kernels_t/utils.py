

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import triton

from .autotune import get_autotuner
from ..utils.logging import get_logger


logger = get_logger(__name__)


def get_block_size(
    seq_len: int,
    head_dim: int,
    kernel_name: str = "attention"
) -> int:
    autotuner = get_autotuner()
    
    configs = [
        {"BLOCK_SIZE": 16},
        {"BLOCK_SIZE": 32},
        {"BLOCK_SIZE": 64},
        {"BLOCK_SIZE": 128},
        {"BLOCK_SIZE": 256},
    ]
    
    valid_configs = []
    for config in configs:
        block_size = config["BLOCK_SIZE"]
        if block_size <= seq_len and block_size <= head_dim:
            valid_configs.append(config)
    
    if not valid_configs:
        return 32
    
    optimal_config = autotuner.get_optimal_config(
        kernel_name,
        seq_len=seq_len,
        head_dim=head_dim
    )
    
    if optimal_config is not None:
        return optimal_config["BLOCK_SIZE"]
    
    return max(config["BLOCK_SIZE"] for config in valid_configs)


def get_num_blocks(
    seq_len: int,
    block_size: int
) -> int:
    return (seq_len + block_size - 1) // block_size


def get_threads_per_block(
    block_size: int,
    head_dim: int
) -> int:
    return min(block_size, head_dim)


def get_shared_memory_size(
    block_size: int,
    head_dim: int
) -> int:
    return block_size * head_dim * 4


def get_optimal_config(
    kernel_name: str,
    **kwargs
) -> Dict[str, Any]:
    autotuner = get_autotuner()
    return autotuner.get_optimal_config(kernel_name, **kwargs)


def set_optimal_config(
    kernel_name: str,
    config: Dict[str, Any],
    **kwargs
) -> None:
    autotuner = get_autotuner()
    autotuner.set_optimal_config(kernel_name, config, **kwargs)


def create_kernel_grid(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    block_size: int
) -> Tuple[int, int, int]:
    num_blocks = get_num_blocks(seq_len, block_size)
    return (batch_size, n_heads, num_blocks)


def get_kernel_launch_config(
    kernel_func,
    **kwargs
) -> Dict[str, Any]:
    seq_len = kwargs.get('seq_len', 512)
    head_dim = kwargs.get('head_dim', 64)
    block_size = get_block_size(seq_len, head_dim)
    
    batch_size = kwargs.get('batch_size', 32)
    n_heads = kwargs.get('n_heads', 16)
    grid = create_kernel_grid(batch_size, n_heads, seq_len, block_size)
    
    return {
        "grid": grid,
        "block_size": block_size,
        "num_blocks": get_num_blocks(seq_len, block_size),
        "threads_per_block": get_threads_per_block(block_size, head_dim),
        "shared_memory_size": get_shared_memory_size(block_size, head_dim)
    }


def validate_kernel_config(
    config: Dict[str, Any],
    **kwargs
) -> bool:
    try:
        required_params = ["BLOCK_SIZE"]
        for param in required_params:
            if param not in config:
                return False
        
        block_size = config["BLOCK_SIZE"]
        seq_len = kwargs.get('seq_len', 512)
        head_dim = kwargs.get('head_dim', 64)
        
        if block_size > seq_len or block_size > head_dim:
            return False
        
        if block_size <= 0:
            return False
        
        return True
        
    except Exception:
        return False


def get_kernel_memory_usage(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    block_size: int
) -> Dict[str, float]:
    q_size = batch_size * n_heads * seq_len * head_dim * 4
    k_size = batch_size * n_heads * seq_len * head_dim * 4
    v_size = batch_size * n_heads * seq_len * head_dim * 4
    
    o_size = batch_size * n_heads * seq_len * head_dim * 4
    scores_size = batch_size * n_heads * seq_len * seq_len * 4
    total_size = q_size + k_size + v_size + o_size + scores_size
    
    return {
        "input_memory_gb": (q_size + k_size + v_size) / 1e9,
        "output_memory_gb": o_size / 1e9,
        "intermediate_memory_gb": scores_size / 1e9,
        "total_memory_gb": total_size / 1e9,
        "block_size": block_size,
        "num_blocks": get_num_blocks(seq_len, block_size)
    }


def get_kernel_throughput(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    elapsed_time_ms: float
) -> Dict[str, float]:
    total_ops = batch_size * n_heads * seq_len * seq_len * head_dim
    throughput_ops_per_sec = total_ops / (elapsed_time_ms / 1000.0)
    throughput_ops_per_ms = total_ops / elapsed_time_ms
    
    return {
        "total_ops": total_ops,
        "throughput_ops_per_sec": throughput_ops_per_sec,
        "throughput_ops_per_ms": throughput_ops_per_ms,
        "elapsed_time_ms": elapsed_time_ms
    }


def get_kernel_efficiency(
    actual_throughput: float,
    theoretical_throughput: float
) -> float:
    if theoretical_throughput <= 0:
        return 0.0
    
    return (actual_throughput / theoretical_throughput) * 100.0


def get_kernel_roofline(
    arithmetic_intensity: float,
    memory_bandwidth: float,
    compute_peak: float
) -> float:
    memory_bound = arithmetic_intensity * memory_bandwidth
    compute_bound = compute_peak
    
    return min(memory_bound, compute_bound)


def get_kernel_bottleneck(
    arithmetic_intensity: float,
    memory_bandwidth: float,
    compute_peak: float
) -> str:
    memory_bound = arithmetic_intensity * memory_bandwidth
    compute_bound = compute_peak
    
    if memory_bound < compute_bound:
        return "memory"
    else:
        return "compute"
