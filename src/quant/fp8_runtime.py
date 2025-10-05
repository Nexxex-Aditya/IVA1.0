
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger


logger = get_logger(__name__)


class FP8AmaxTracker:
    
    def __init__(
        self,
        amax_history_len: int = 1024,
        amax_compute_algo: str = "max",
        scaling_factor_compute_algo: str = "max"
    ):
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.scaling_factor_compute_algo = scaling_factor_compute_algo
        
        self.amax_history: List[float] = []
        
        self.current_amax = 0.0
        self.current_scaling_factor = 1.0
        
        self.fp8_dtype = torch.float8_e4m3fn
        self.amax_dtype = torch.float32
    
    def update_amax(self, tensor: torch.Tensor) -> None:
        if self.amax_compute_algo == "max":
            amax = tensor.abs().max().item()
        elif self.amax_compute_algo == "mean":
            amax = tensor.abs().mean().item()
        else:
            raise ValueError(f"Unknown amax compute algorithm: {self.amax_compute_algo}")
        
        self.amax_history.append(amax)
        if len(self.amax_history) > self.amax_history_len:
            self.amax_history.pop(0)
        
        self.current_amax = max(self.current_amax, amax)
        
        self._update_scaling_factor()
    
    def _update_scaling_factor(self) -> None:
        if self.current_amax == 0:
            self.current_scaling_factor = 1.0
            return
        
        if self.scaling_factor_compute_algo == "max":
            max_amax = max(self.amax_history) if self.amax_history else self.current_amax
            self.current_scaling_factor = self._compute_scaling_factor(max_amax)
        elif self.scaling_factor_compute_algo == "mean":
            mean_amax = sum(self.amax_history) / len(self.amax_history) if self.amax_history else self.current_amax
            self.current_scaling_factor = self._compute_scaling_factor(mean_amax)
        else:
            raise ValueError(f"Unknown scaling factor compute algorithm: {self.scaling_factor_compute_algo}")
    
    def _compute_scaling_factor(self, amax: float) -> float:
        fp8_max = 448.0
        
        if amax == 0:
            return 1.0
        
        scaling_factor = fp8_max / amax
        
        scaling_factor = max(1e-8, min(scaling_factor, 1e8))
        
        return scaling_factor
    
    def get_scaling_factor(self) -> float:
        return self.current_scaling_factor
    
    def get_amax(self) -> float:
        return self.current_amax
    
    def reset(self) -> None:
        self.amax_history.clear()
        self.current_amax = 0.0
        self.current_scaling_factor = 1.0


class FP8Autocast:
    
    def __init__(
        self,
        ops: Optional[Set[str]] = None,
        amax_history_len: int = 1024,
        amax_compute_algo: str = "max",
        scaling_factor_compute_algo: str = "max"
    ):
        self.ops = ops or {"attention", "ffn", "linear", "matmul"}
        self.amax_tracker = FP8AmaxTracker(
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
            scaling_factor_compute_algo=scaling_factor_compute_algo
        )
        
        self.enabled_ops: Set[str] = set()
        
        self.original_functions: Dict[str, Any] = {}
    
    def __enter__(self):
        self._enable_fp8()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._disable_fp8()
    
    def _enable_fp8(self) -> None:
        for op in self.ops:
            if op in self.enabled_ops:
                continue
            
            if op == "attention":
                self._enable_attention_fp8()
            elif op == "ffn":
                self._enable_ffn_fp8()
            elif op == "linear":
                self._enable_linear_fp8()
            elif op == "matmul":
                self._enable_matmul_fp8()
            
            self.enabled_ops.add(op)
    
    def _disable_fp8(self) -> None:
        for op, original_func in self.original_functions.items():
            if op == "attention":
                self._disable_attention_fp8(original_func)
            elif op == "ffn":
                self._disable_ffn_fp8(original_func)
            elif op == "linear":
                self._disable_linear_fp8(original_func)
            elif op == "matmul":
                self._disable_matmul_fp8(original_func)
        
        self.original_functions.clear()
        self.enabled_ops.clear()
    
    def _enable_attention_fp8(self) -> None:
        self.original_functions["attention"] = F.linear
        
        def fp8_linear(*args, **kwargs):
            return self._fp8_linear(*args, **kwargs)
        
        F.linear = fp8_linear
    
    def _disable_attention_fp8(self, original_func) -> None:
        F.linear = original_func
    
    def _enable_ffn_fp8(self) -> None:
        pass
    
    def _disable_ffn_fp8(self, original_func) -> None:
        pass
    
    def _enable_linear_fp8(self) -> None:
        pass
    
    def _disable_linear_fp8(self, original_func) -> None:
        pass
    
    def _enable_matmul_fp8(self) -> None:
        pass
    
    def _disable_matmul_fp8(self, original_func) -> None:
        pass
    
    def _fp8_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.amax_tracker.update_amax(input)
        self.amax_tracker.update_amax(weight)
        scaling_factor = self.amax_tracker.get_scaling_factor()
        input_fp8 = self._quantize_to_fp8(input, scaling_factor)
        weight_fp8 = self._quantize_to_fp8(weight, scaling_factor)
        output = F.linear(input_fp8, weight_fp8, bias)
        output = self._dequantize_from_fp8(output, scaling_factor)
        return output
    
    def _quantize_to_fp8(self, tensor: torch.Tensor, scaling_factor: float) -> torch.Tensor:
        scaled_tensor = tensor * scaling_factor
        quantized_tensor = scaled_tensor.to(self.amax_tracker.fp8_dtype)      
        return quantized_tensor
    
    def _dequantize_from_fp8(self, tensor: torch.Tensor, scaling_factor: float) -> torch.Tensor:
        dequantized_tensor = tensor.float()
        dequantized_tensor = dequantized_tensor / scaling_factor
        return dequantized_tensor
    
    def get_amax_drift(self) -> float:
        if not self.amax_tracker.amax_history:
            return 0.0
        current_amax = self.amax_tracker.current_amax
        mean_amax = sum(self.amax_tracker.amax_history) / len(self.amax_tracker.amax_history)
        if mean_amax == 0:
            return 0.0
        
        return abs(current_amax - mean_amax) / mean_amax


_global_fp8_autocast: Optional[FP8Autocast] = None


@contextmanager
def fp8_autocast(
    ops: Optional[Set[str]] = None,
    amax_history_len: int = 1024,
    amax_compute_algo: str = "max",
    scaling_factor_compute_algo: str = "max"
):
    global _global_fp8_autocast
    
    fp8_autocast_instance = FP8Autocast(
        ops=ops,
        amax_history_len=amax_history_len,
        amax_compute_algo=amax_compute_algo,
        scaling_factor_compute_algo=scaling_factor_compute_algo
    )
    
    _global_fp8_autocast = fp8_autocast_instance
    
    try:
        with fp8_autocast_instance:
            yield fp8_autocast_instance
    finally:
        _global_fp8_autocast = None


def get_fp8_amax() -> float:
    global _global_fp8_autocast
    if _global_fp8_autocast is None:
        return 0.0
    return _global_fp8_autocast.amax_tracker.get_amax()


def set_fp8_amax(amax: float) -> None:
    global _global_fp8_autocast
    if _global_fp8_autocast is not None:
        _global_fp8_autocast.amax_tracker.current_amax = amax


def get_fp8_scaling_factor() -> float:
    global _global_fp8_autocast
    if _global_fp8_autocast is None:
        return 1.0
    return _global_fp8_autocast.amax_tracker.get_scaling_factor()


def set_fp8_scaling_factor(scaling_factor: float) -> None:
    global _global_fp8_autocast
    if _global_fp8_autocast is not None:
        _global_fp8_autocast.amax_tracker.current_scaling_factor = scaling_factor


def get_fp8_amax_drift() -> float:
    global _global_fp8_autocast
    if _global_fp8_autocast is None:
        return 0.0
    return _global_fp8_autocast.get_amax_drift()
