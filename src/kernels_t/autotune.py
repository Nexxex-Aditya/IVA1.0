
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import triton

from ..utils.logging import get_logger


logger = get_logger(__name__)


class Autotuner:
    
    def __init__(
        self,
        cache_file: Optional[str] = None,
        max_cache_size: int = 1000
    ):
        self.cache_file = cache_file or "triton_autotune_cache.json"
        self.max_cache_size = max_cache_size
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load autotune cache: {e}")
        
        return {}
    
    def _save_cache(self) -> None:
        try:
            cache_path = Path(self.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save autotune cache: {e}")
    
    def _get_cache_key(
        self,
        kernel_name: str,
        **kwargs
    ) -> str:
        key_parts = [kernel_name]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (int, float, str, bool)):
                key_parts.append(f"{k}={v}")
            elif isinstance(v, torch.Tensor):
                key_parts.append(f"{k}={v.shape}")
        
        return "_".join(key_parts)
    
    def get_optimal_config(
        self,
        kernel_name: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        cache_key = self._get_cache_key(kernel_name, **kwargs)
        return self.cache.get(cache_key)
    
    def set_optimal_config(
        self,
        kernel_name: str,
        config: Dict[str, Any],
        **kwargs
    ) -> None:
        cache_key = self._get_cache_key(kernel_name, **kwargs)
        self.cache[cache_key] = config
        
        if len(self.cache) > self.max_cache_size:
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - self.max_cache_size]
            for key in keys_to_remove:
                del self.cache[key]
        
        self._save_cache()
    
    def autotune(
        self,
        kernel_func,
        kernel_name: str,
        configs: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        cached_config = self.get_optimal_config(kernel_name, **kwargs)
        if cached_config is not None:
            return cached_config
        
        logger.info(f"Autotuning {kernel_name} with {len(configs)} configurations")
        
        best_config = None
        best_time = float('inf')
        
        for config in configs:
            try:
                times = []
                for _ in range(5):
                    start_time = time.time()
                    kernel_func(config=config, **kwargs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = sum(times) / len(times)
                
                if avg_time < best_time:
                    best_time = avg_time
                    best_config = config
                
            except Exception as e:
                logger.warning(f"Configuration {config} failed: {e}")
                continue
        
        if best_config is None:
            raise RuntimeError(f"All configurations failed for {kernel_name}")
        
        self.set_optimal_config(kernel_name, best_config, **kwargs)
        
        logger.info(f"Best configuration for {kernel_name}: {best_config} (time: {best_time:.4f}s)")
        
        return best_config


def get_autotuner() -> Autotuner:
    if not hasattr(get_autotuner, '_instance'):
        get_autotuner._instance = Autotuner()
    return get_autotuner._instance


def autotune_kernel(
    kernel_func,
    kernel_name: str,
    configs: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    autotuner = get_autotuner()
    return autotuner.autotune(kernel_func, kernel_name, configs, **kwargs)


def get_optimal_block_size(
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


def get_optimal_num_blocks(
    seq_len: int,
    block_size: int
) -> int:
    return (seq_len + block_size - 1) // block_size


def get_optimal_threads_per_block(
    block_size: int,
    head_dim: int
) -> int:
    return min(block_size, head_dim)


def get_optimal_shared_memory_size(
    block_size: int,
    head_dim: int
) -> int:
    return block_size * head_dim * 4


def benchmark_kernel(
    kernel_func,
    config: Dict[str, Any],
    num_runs: int = 100,
    **kwargs
) -> Dict[str, float]:
    for _ in range(10):
        kernel_func(config=config, **kwargs)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        kernel_func(config=config, **kwargs)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / num_runs
    
    return {
        "elapsed_time_ms": elapsed_time * 1000,
        "throughput_ops_per_sec": 1.0 / elapsed_time,
    }
