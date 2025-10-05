
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import torch
import triton
import triton.language as tl

from .autotune import get_autotuner
from .utils import get_block_size, get_num_blocks
from ..utils.logging import get_logger


logger = get_logger(__name__)


@triton.jit
def kv_gather_kernel(
    kv_ptr, indices_ptr, output_ptr,
    kv_stride_batch, kv_stride_head, kv_stride_seq, kv_stride_dim,
    output_stride_batch, output_stride_head, output_stride_seq, output_stride_dim,
    batch_size, n_heads, seq_len, head_dim, num_indices,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_idx = tl.program_id(2)
    
    if pid_batch >= batch_size or pid_head >= n_heads or pid_idx >= num_indices:
        return
    
    idx = tl.load(indices_ptr + pid_idx)
    
    kv_offset = pid_batch * kv_stride_batch + pid_head * kv_stride_head + idx * kv_stride_seq
    output_offset = pid_batch * output_stride_batch + pid_head * output_stride_head + pid_idx * output_stride_seq
    
    for i in range(0, head_dim, BLOCK_SIZE):
        kv_block = tl.load(kv_ptr + kv_offset + i * kv_stride_dim + tl.arange(0, BLOCK_SIZE))
        tl.store(output_ptr + output_offset + i * output_stride_dim + tl.arange(0, BLOCK_SIZE), kv_block)


@triton.jit
def kv_scatter_kernel(
    kv_ptr, indices_ptr, values_ptr,
    kv_stride_batch, kv_stride_head, kv_stride_seq, kv_stride_dim,
    values_stride_batch, values_stride_head, values_stride_seq, values_stride_dim,
    batch_size, n_heads, seq_len, head_dim, num_indices,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_idx = tl.program_id(2)
    
    if pid_batch >= batch_size or pid_head >= n_heads or pid_idx >= num_indices:
        return
    
    idx = tl.load(indices_ptr + pid_idx)
    
    kv_offset = pid_batch * kv_stride_batch + pid_head * kv_stride_head + idx * kv_stride_seq
    values_offset = pid_batch * values_stride_batch + pid_head * values_stride_head + pid_idx * values_stride_seq
    
    for i in range(0, head_dim, BLOCK_SIZE):
        values_block = tl.load(values_ptr + values_offset + i * values_stride_dim + tl.arange(0, BLOCK_SIZE))
        tl.store(kv_ptr + kv_offset + i * kv_stride_dim + tl.arange(0, BLOCK_SIZE), values_block)


@triton.jit
def kv_compact_kernel(
    kv_ptr, mask_ptr, output_ptr,
    kv_stride_batch, kv_stride_head, kv_stride_seq, kv_stride_dim,
    output_stride_batch, output_stride_head, output_stride_seq, output_stride_dim,
    batch_size, n_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)
    
    if pid_batch >= batch_size or pid_head >= n_heads or pid_seq >= seq_len:
        return
    
    mask = tl.load(mask_ptr + pid_batch * seq_len + pid_seq)
    if mask == 0:
        return
    
    kv_offset = pid_batch * kv_stride_batch + pid_head * kv_stride_head + pid_seq * kv_stride_seq
    output_offset = pid_batch * output_stride_batch + pid_head * output_stride_head + pid_seq * output_stride_seq
    
    for i in range(0, head_dim, BLOCK_SIZE):
        kv_block = tl.load(kv_ptr + kv_offset + i * kv_stride_dim + tl.arange(0, BLOCK_SIZE))
        tl.store(output_ptr + output_offset + i * output_stride_dim + tl.arange(0, BLOCK_SIZE), kv_block)


class KVOpsTriton(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        operation: str = "gather"
    ) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = kv_cache.shape
        num_indices = indices.shape[0]
        device = kv_cache.device
        
        block_size = get_block_size(head_dim, 1)
        
        if operation == "gather":
            output = torch.empty(batch_size, n_heads, num_indices, head_dim, device=device)
            
            grid = (batch_size, n_heads, num_indices)
            kv_gather_kernel[grid](
                kv_cache, indices, output,
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                batch_size, n_heads, seq_len, head_dim, num_indices,
                BLOCK_SIZE=block_size,
            )
            
            return output
        
        elif operation == "scatter":
            if values is None:
                raise ValueError("Values tensor required for scatter operation")
            
            output = kv_cache.clone()
            
            grid = (batch_size, n_heads, num_indices)
            kv_scatter_kernel[grid](
                output, indices, values,
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                values.stride(0), values.stride(1), values.stride(2), values.stride(3),
                batch_size, n_heads, seq_len, head_dim, num_indices,
                BLOCK_SIZE=block_size,
            )
            
            return output
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return None, None, None, None


def kv_ops_triton(
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    operation: str = "gather"
) -> torch.Tensor:
    return KVOpsTriton.apply(kv_cache, indices, values, operation)


def kv_compact_triton(
    kv_cache: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    batch_size, n_heads, seq_len, head_dim = kv_cache.shape
    device = kv_cache.device
    
    num_valid = mask.sum().item()
    
    output = torch.empty(batch_size, n_heads, num_valid, head_dim, device=device)
    
    block_size = get_block_size(head_dim, 1)
    
    grid = (batch_size, n_heads, seq_len)
    kv_compact_kernel[grid](
        kv_cache, mask, output,
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, n_heads, seq_len, head_dim,
        BLOCK_SIZE=block_size,
    )
    
    return output


def benchmark_kv_ops_triton(
    batch_size: int = 32,
    n_heads: int = 16,
    seq_len: int = 512,
    head_dim: int = 64,
    num_indices: int = 128,
    num_runs: int = 100
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    kv_cache = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    indices = torch.randint(0, seq_len, (num_indices,), device=device)
    values = torch.randn(batch_size, n_heads, num_indices, head_dim, device=device)
    
    for _ in range(10):
        _ = kv_ops_triton(kv_cache, indices, operation="gather")
        _ = kv_ops_triton(kv_cache, indices, values, operation="scatter")
    
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_runs):
        _ = kv_ops_triton(kv_cache, indices, operation="gather")
    end_time.record()
    
    torch.cuda.synchronize()
    gather_time = start_time.elapsed_time(end_time) / num_runs
    
    start_time.record()
    for _ in range(num_runs):
        _ = kv_ops_triton(kv_cache, indices, values, operation="scatter")
    end_time.record()
    
    torch.cuda.synchronize()
    scatter_time = start_time.elapsed_time(end_time) / num_runs
    
    return {
        "gather_time_ms": gather_time,
        "scatter_time_ms": scatter_time,
        "memory_usage_gb": (kv_cache.numel() + values.numel()) * 4 / 1e9
    }
