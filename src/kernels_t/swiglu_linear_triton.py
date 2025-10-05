
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
def swiglu_linear_kernel(
    x_ptr, gate_weight_ptr, up_weight_ptr, down_weight_ptr,
    gate_bias_ptr, up_bias_ptr, down_bias_ptr, output_ptr,
    x_stride_batch, x_stride_seq, x_stride_dim,
    output_stride_batch, output_stride_seq, output_stride_dim,
    batch_size, seq_len, d_model, hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    if pid_batch >= batch_size or pid_seq >= seq_len:
        return
    
    x_offset = pid_batch * x_stride_batch + pid_seq * x_stride_seq
    output_offset = pid_batch * output_stride_batch + pid_seq * output_stride_seq
    
    x_block = tl.load(x_ptr + x_offset + tl.arange(0, BLOCK_SIZE) * x_stride_dim)
    
    gate = tl.zeros([hidden_dim], dtype=tl.float32)
    for i in range(0, hidden_dim, BLOCK_SIZE):
        gate_block = tl.load(gate_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * d_model)
        gate[i:i+BLOCK_SIZE] = tl.sum(x_block * gate_block, axis=0)
    
    up = tl.zeros([hidden_dim], dtype=tl.float32)
    for i in range(0, hidden_dim, BLOCK_SIZE):
        up_block = tl.load(up_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * d_model)
        up[i:i+BLOCK_SIZE] = tl.sum(x_block * up_block, axis=0)
    
    if gate_bias_ptr is not None:
        gate_bias = tl.load(gate_bias_ptr + tl.arange(0, hidden_dim))
        gate = gate + gate_bias
    
    if up_bias_ptr is not None:
        up_bias = tl.load(up_bias_ptr + tl.arange(0, hidden_dim))
        up = up + up_bias
    
    gate_swiglu = tl.where(gate > 0, gate, gate * 0.1)
    hidden = gate_swiglu * up
    
    output = tl.zeros([d_model], dtype=tl.float32)
    for i in range(0, d_model, BLOCK_SIZE):
        down_block = tl.load(down_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * hidden_dim)
        output[i:i+BLOCK_SIZE] = tl.sum(hidden * down_block, axis=0)
    
    if down_bias_ptr is not None:
        down_bias = tl.load(down_bias_ptr + tl.arange(0, d_model))
        output = output + down_bias
    
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE) * output_stride_dim, output)


@triton.jit
def swiglu_linear_backward_kernel(
    x_ptr, gate_weight_ptr, up_weight_ptr, down_weight_ptr,
    gate_bias_ptr, up_bias_ptr, down_bias_ptr,
    grad_output_ptr, grad_input_ptr, grad_gate_weight_ptr, grad_up_weight_ptr, grad_down_weight_ptr,
    grad_gate_bias_ptr, grad_up_bias_ptr, grad_down_bias_ptr,
    x_stride_batch, x_stride_seq, x_stride_dim,
    grad_output_stride_batch, grad_output_stride_seq, grad_output_stride_dim,
    grad_input_stride_batch, grad_input_stride_seq, grad_input_stride_dim,
    batch_size, seq_len, d_model, hidden_dim,
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
    
    gate = tl.zeros([hidden_dim], dtype=tl.float32)
    up = tl.zeros([hidden_dim], dtype=tl.float32)
    
    for i in range(0, hidden_dim, BLOCK_SIZE):
        gate_block = tl.load(gate_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * d_model)
        up_block = tl.load(up_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * d_model)
        gate[i:i+BLOCK_SIZE] = tl.sum(x_block * gate_block, axis=0)
        up[i:i+BLOCK_SIZE] = tl.sum(x_block * up_block, axis=0)
    
    if gate_bias_ptr is not None:
        gate_bias = tl.load(gate_bias_ptr + tl.arange(0, hidden_dim))
        gate = gate + gate_bias
    
    if up_bias_ptr is not None:
        up_bias = tl.load(up_bias_ptr + tl.arange(0, hidden_dim))
        up = up + up_bias
    
    gate_swiglu = tl.where(gate > 0, gate, gate * 0.1)
    hidden = gate_swiglu * up
    
    grad_hidden = tl.zeros([hidden_dim], dtype=tl.float32)
    for i in range(0, hidden_dim, BLOCK_SIZE):
        down_block = tl.load(down_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * hidden_dim)
        grad_hidden[i:i+BLOCK_SIZE] = tl.sum(grad_output_block * down_block, axis=0)
    
    grad_gate = grad_hidden * up
    grad_up = grad_hidden * gate_swiglu
    
    grad_input = tl.zeros([d_model], dtype=tl.float32)
    for i in range(0, d_model, BLOCK_SIZE):
        gate_weight_block = tl.load(gate_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * d_model)
        up_weight_block = tl.load(up_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * d_model)
        grad_input[i:i+BLOCK_SIZE] = tl.sum(grad_gate * gate_weight_block, axis=0) + tl.sum(grad_up * up_weight_block, axis=0)
    
    tl.store(grad_input_ptr + grad_input_offset + tl.arange(0, BLOCK_SIZE) * grad_input_stride_dim, grad_input)
    
    if grad_gate_weight_ptr is not None:
        for i in range(0, hidden_dim, BLOCK_SIZE):
            gate_weight_grad = tl.zeros([BLOCK_SIZE, d_model], dtype=tl.float32)
            for j in range(0, d_model, BLOCK_SIZE):
                gate_weight_grad[:, j:j+BLOCK_SIZE] = grad_gate[i:i+BLOCK_SIZE, None] * x_block[None, j:j+BLOCK_SIZE]
            tl.store(grad_gate_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * d_model, gate_weight_grad)
    
    if grad_up_weight_ptr is not None:
        for i in range(0, hidden_dim, BLOCK_SIZE):
            up_weight_grad = tl.zeros([BLOCK_SIZE, d_model], dtype=tl.float32)
            for j in range(0, d_model, BLOCK_SIZE):
                up_weight_grad[:, j:j+BLOCK_SIZE] = grad_up[i:i+BLOCK_SIZE, None] * x_block[None, j:j+BLOCK_SIZE]
            tl.store(grad_up_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * d_model, up_weight_grad)
    
    if grad_down_weight_ptr is not None:
        for i in range(0, d_model, BLOCK_SIZE):
            down_weight_grad = tl.zeros([BLOCK_SIZE, hidden_dim], dtype=tl.float32)
            for j in range(0, hidden_dim, BLOCK_SIZE):
                down_weight_grad[:, j:j+BLOCK_SIZE] = grad_output_block[i:i+BLOCK_SIZE, None] * hidden[None, j:j+BLOCK_SIZE]
            tl.store(grad_down_weight_ptr + i + tl.arange(0, BLOCK_SIZE) * hidden_dim, down_weight_grad)


class SwiGLULinearTriton(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_bias: Optional[torch.Tensor] = None,
        up_bias: Optional[torch.Tensor] = None,
        down_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        hidden_dim = gate_weight.shape[0]
        device = x.device
        
        block_size = get_block_size(d_model, 1)
        
        output = torch.empty_like(x)
        
        grid = (batch_size, seq_len)
        swiglu_linear_kernel[grid](
            x, gate_weight, up_weight, down_weight,
            gate_bias, up_bias, down_bias, output,
            x.stride(0), x.stride(1), x.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            batch_size, seq_len, d_model, hidden_dim,
            BLOCK_SIZE=block_size,
        )
        
        ctx.save_for_backward(x, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias = ctx.saved_tensors
        batch_size, seq_len, d_model = x.shape
        hidden_dim = gate_weight.shape[0]
        
        block_size = get_block_size(d_model, 1)
        
        grad_input = torch.empty_like(x)
        grad_gate_weight = torch.zeros_like(gate_weight)
        grad_up_weight = torch.zeros_like(up_weight)
        grad_down_weight = torch.zeros_like(down_weight)
        grad_gate_bias = torch.zeros_like(gate_bias) if gate_bias is not None else None
        grad_up_bias = torch.zeros_like(up_bias) if up_bias is not None else None
        grad_down_bias = torch.zeros_like(down_bias) if down_bias is not None else None
        
        grid = (batch_size, seq_len)
        swiglu_linear_backward_kernel[grid](
            x, gate_weight, up_weight, down_weight,
            gate_bias, up_bias, down_bias,
            grad_output, grad_input, grad_gate_weight, grad_up_weight, grad_down_weight,
            grad_gate_bias, grad_up_bias, grad_down_bias,
            x.stride(0), x.stride(1), x.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            grad_input.stride(0), grad_input.stride(1), grad_input.stride(2),
            batch_size, seq_len, d_model, hidden_dim,
            BLOCK_SIZE=block_size,
        )
        
        return grad_input, grad_gate_weight, grad_up_weight, grad_down_weight, grad_gate_bias, grad_up_bias, grad_down_bias


def swiglu_linear_triton(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor] = None,
    up_bias: Optional[torch.Tensor] = None,
    down_bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return SwiGLULinearTriton.apply(
        x, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias
    )


def benchmark_swiglu_linear_triton(
    batch_size: int = 32,
    seq_len: int = 512,
    d_model: int = 1024,
    hidden_dim: int = 4096,
    num_runs: int = 100
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    gate_weight = torch.randn(hidden_dim, d_model, device=device)
    up_weight = torch.randn(hidden_dim, d_model, device=device)
    down_weight = torch.randn(d_model, hidden_dim, device=device)
    gate_bias = torch.randn(hidden_dim, device=device)
    up_bias = torch.randn(hidden_dim, device=device)
    down_bias = torch.randn(d_model, device=device)
    
    for _ in range(10):
        _ = swiglu_linear_triton(x, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias)
    
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_runs):
        _ = swiglu_linear_triton(x, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias)
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / num_runs
    
    total_ops = batch_size * seq_len * d_model * hidden_dim * 3
    throughput = total_ops / (elapsed_time / 1000.0)
    
    return {
        "elapsed_time_ms": elapsed_time,
        "throughput_ops_per_sec": throughput,
        "memory_usage_gb": (x.numel() + gate_weight.numel() + up_weight.numel() + down_weight.numel()) * 4 / 1e9
    }
