
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
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    x_stride_batch, x_stride_seq, x_stride_dim,
    output_stride_batch, output_stride_seq, output_stride_dim,
    batch_size, seq_len, d_model, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    if pid_batch >= batch_size or pid_seq >= seq_len:
        return
    
    x_offset = pid_batch * x_stride_batch + pid_seq * x_stride_seq
    output_offset = pid_batch * output_stride_batch + pid_seq * output_stride_seq
    
    x_block = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE) * x_stride_dim)
    
    rms = tl.sqrt(tl.sum(x_block * x_block) / d_model + eps)
    
    x_normalized = x_block / rms
    
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE))
    
    output = x_normalized * weight
    
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE) * output_stride_dim, output)


@triton.jit
def rmsnorm_backward_kernel(
    x_ptr, weight_ptr, grad_output_ptr, grad_input_ptr, grad_weight_ptr,
    x_stride_batch, x_stride_seq, x_stride_dim,
    grad_output_stride_batch, grad_output_stride_seq, grad_output_stride_dim,
    grad_input_stride_batch, grad_input_stride_seq, grad_input_stride_dim,
    batch_size, seq_len, d_model, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    if pid_batch >= batch_size or pid_seq >= seq_len:
        return
    
    x_offset = pid_batch * x_stride_batch + pid_seq * x_stride_seq
    grad_output_offset = pid_batch * grad_output_stride_batch + pid_seq * grad_output_stride_seq
    grad_input_offset = pid_batch * grad_input_stride_batch + pid_seq * grad_input_stride_seq
    
    x_block = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE) * x_stride_dim)
    grad_output_block = tl.load(
        grad_output_ptr + grad_output_offset + tl.arange(0, BLOCK_SIZE) * grad_output_stride_dim
    )
    
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE))
    rms = tl.sqrt(tl.sum(x_block * x_block) / d_model + eps)
    x_normalized = x_block / rms
    grad_input = grad_output_block * weight / rms - x_normalized * tl.sum(grad_output_block * weight * x_normalized) / d_model
    grad_weight = grad_output_block * x_normalized
    tl.store(grad_input_ptr + grad_input_offset + tl.arange(0, BLOCK_SIZE) * grad_input_stride_dim, grad_input)
    tl.store(grad_weight_ptr + tl.arange(0, BLOCK_SIZE), grad_weight)


class RMSNormTriton(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        block_size = get_block_size(d_model, 1)
        
        output = torch.empty_like(x)
        
        grid = (batch_size, seq_len)
        rmsnorm_kernel[grid](
            x, weight, output,
            x.stride(0), x.stride(1), x.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            batch_size, seq_len, d_model, eps,
            BLOCK_SIZE=block_size,
        )
        
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x, weight = ctx.saved_tensors
        batch_size, seq_len, d_model = x.shape
        
        block_size = get_block_size(d_model, 1)
        
        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros_like(weight)
        
        grid = (batch_size, seq_len)
        rmsnorm_backward_kernel[grid](
            x, weight, grad_output, grad_input, grad_weight,
            x.stride(0), x.stride(1), x.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            grad_input.stride(0), grad_input.stride(1), grad_input.stride(2),
            batch_size, seq_len, d_model, ctx.eps,
            BLOCK_SIZE=block_size,
        )
        
        return grad_input, grad_weight, None


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    return RMSNormTriton.apply(x, weight, eps)


def benchmark_rmsnorm_triton(
    batch_size: int = 32,
    seq_len: int = 512,
    d_model: int = 1024,
    num_runs: int = 100
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    weight = torch.ones(d_model, device=device)
    
    for _ in range(10):
        _ = rmsnorm_triton(x, weight)
    
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_runs):
        _ = rmsnorm_triton(x, weight)
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / num_runs
    
    total_ops = batch_size * seq_len * d_model
    throughput = total_ops / (elapsed_time / 1000.0)
    
    return {
        "elapsed_time_ms": elapsed_time,
        "throughput_ops_per_sec": throughput,
        "memory_usage_gb": (x.numel() + weight.numel()) * 4 / 1e9
    }
