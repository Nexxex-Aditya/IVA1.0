
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
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    q_stride_batch, q_stride_seq, q_stride_head, q_stride_dim,
    k_stride_batch, k_stride_seq, k_stride_head, k_stride_dim,
    v_stride_batch, v_stride_seq, v_stride_head, v_stride_dim,
    o_stride_batch, o_stride_seq, o_stride_head, o_stride_dim,
    batch_size, seq_len, n_heads, head_dim,
    scale, dropout_prob, training,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)
    
    if pid_batch >= batch_size or pid_head >= n_heads or pid_seq >= seq_len:
        return
    
    q_offset = pid_batch * q_stride_batch + pid_head * q_stride_head + pid_seq * q_stride_seq
    k_offset = pid_batch * k_stride_batch + pid_head * k_stride_head
    v_offset = pid_batch * v_stride_batch + pid_head * v_stride_head
    o_offset = pid_batch * o_stride_batch + pid_head * o_stride_head + pid_seq * o_stride_seq
    
    q = tl.load(q_ptr + q_offset + tl.arange(0, head_dim) * q_stride_dim)
    
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    for block_start in range(0, seq_len, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, seq_len)
        
        k_block = tl.load(
            k_ptr + k_offset + block_start * k_stride_seq + 
            tl.arange(0, head_dim) * k_stride_dim
        )
        
        score = tl.sum(q * k_block) * scale
        
        if block_start + tl.arange(0, BLOCK_SIZE)[0] > pid_seq:
            score = float('-inf')
        
        score = tl.exp(score - tl.max(score))
        
        if training and dropout_prob > 0:
            dropout_mask = tl.rand(0.0, 1.0) > dropout_prob
            score = score * dropout_mask
        
        v_block = tl.load(
            v_ptr + v_offset + block_start * v_stride_seq + 
            tl.arange(0, head_dim) * v_stride_dim
        )
        
        acc = acc + score * v_block
    
    tl.store(o_ptr + o_offset + tl.arange(0, head_dim) * o_stride_dim, acc)


@triton.jit
def attention_backward_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    q_stride_batch, q_stride_seq, q_stride_head, q_stride_dim,
    k_stride_batch, k_stride_seq, k_stride_head, k_stride_dim,
    v_stride_batch, v_stride_seq, v_stride_head, v_stride_dim,
    o_stride_batch, o_stride_seq, o_stride_head, o_stride_dim,
    do_stride_batch, do_stride_seq, do_stride_head, do_stride_dim,
    dq_stride_batch, dq_stride_seq, dq_stride_head, dq_stride_dim,
    dk_stride_batch, dk_stride_seq, dk_stride_head, dk_stride_dim,
    dv_stride_batch, dv_stride_seq, dv_stride_head, dv_stride_dim,
    batch_size, seq_len, n_heads, head_dim,
    scale, dropout_prob, training,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_seq = tl.program_id(2)
    
    if pid_batch >= batch_size or pid_head >= n_heads or pid_seq >= seq_len:
        return
    
    q_offset = pid_batch * q_stride_batch + pid_head * q_stride_head + pid_seq * q_stride_seq
    k_offset = pid_batch * k_stride_batch + pid_head * k_stride_head
    v_offset = pid_batch * v_stride_batch + pid_head * v_stride_head
    o_offset = pid_batch * o_stride_batch + pid_head * o_stride_head + pid_seq * o_stride_seq
    do_offset = pid_batch * do_stride_batch + pid_head * do_stride_head + pid_seq * do_stride_seq
    
    q = tl.load(q_ptr + q_offset + tl.arange(0, head_dim) * q_stride_dim)
    do = tl.load(do_ptr + do_offset + tl.arange(0, head_dim) * do_stride_dim)
    
    dq_acc = tl.zeros([head_dim], dtype=tl.float32)
    dk_acc = tl.zeros([head_dim], dtype=tl.float32)
    dv_acc = tl.zeros([head_dim], dtype=tl.float32)
    
    for block_start in range(0, seq_len, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, seq_len)
        
        k_block = tl.load(
            k_ptr + k_offset + block_start * k_stride_seq + 
            tl.arange(0, head_dim) * k_stride_dim
        )
        v_block = tl.load(
            v_ptr + v_offset + block_start * v_stride_seq + 
            tl.arange(0, head_dim) * v_stride_dim
        )
        
        score = tl.sum(q * k_block) * scale
        
        if block_start + tl.arange(0, BLOCK_SIZE)[0] > pid_seq:
            score = float('-inf')
        
        score = tl.exp(score - tl.max(score))
        
        if training and dropout_prob > 0:
            dropout_mask = tl.rand(0.0, 1.0) > dropout_prob
            score = score * dropout_mask
        
        dq_acc = dq_acc + score * v_block
        dk_acc = dk_acc + score * do
        dv_acc = dv_acc + score * q
    
    dq_offset = pid_batch * dq_stride_batch + pid_head * dq_stride_head + pid_seq * dq_stride_seq
    dk_offset = pid_batch * dk_stride_batch + pid_head * dk_stride_head + pid_seq * dk_stride_seq
    dv_offset = pid_batch * dv_stride_batch + pid_head * dv_stride_head + pid_seq * dv_stride_seq
    
    tl.store(dq_ptr + dq_offset + tl.arange(0, head_dim) * dq_stride_dim, dq_acc)
    tl.store(dk_ptr + dk_offset + tl.arange(0, head_dim) * dk_stride_dim, dk_acc)
    tl.store(dv_ptr + dv_offset + tl.arange(0, head_dim) * dv_stride_dim, dv_acc)


class AttentionTriton(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        dropout_prob: float = 0.0,
        training: bool = True
    ) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = q.shape
        device = q.device
        
        block_size = get_block_size(seq_len, head_dim)
        
        o = torch.empty_like(q)
        
        grid = (batch_size, n_heads, seq_len)
        attention_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),
            batch_size, seq_len, n_heads, head_dim,
            scale, dropout_prob, training,
            BLOCK_SIZE=block_size,
        )
        
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.dropout_prob = dropout_prob
        ctx.training = training
        
        return o
    
    @staticmethod
    def backward(ctx, do: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        q, k, v = ctx.saved_tensors
        batch_size, n_heads, seq_len, head_dim = q.shape
        
        block_size = get_block_size(seq_len, head_dim)
        
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        grid = (batch_size, n_heads, seq_len)
        attention_backward_kernel[grid](
            q, k, v, None, do,
            dq, dk, dv,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            None, None, None, None,
            do.stride(0), do.stride(2), do.stride(1), do.stride(3),
            dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3),
            dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3),
            dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3),
            batch_size, seq_len, n_heads, head_dim,
            ctx.scale, ctx.dropout_prob, ctx.training,
            BLOCK_SIZE=block_size,
        )
        
        return dq, dk, dv, None, None, None


def attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_prob: float = 0.0,
    training: bool = True
) -> torch.Tensor:
    return AttentionTriton.apply(q, k, v, scale, dropout_prob, training)


def benchmark_attention_triton(
    batch_size: int = 32,
    n_heads: int = 16,
    seq_len: int = 512,
    head_dim: int = 64,
    num_runs: int = 100
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    
    for _ in range(10):
        _ = attention_triton(q, k, v, 1.0 / math.sqrt(head_dim))
    
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_runs):
        _ = attention_triton(q, k, v, 1.0 / math.sqrt(head_dim))
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / num_runs
    
    total_ops = batch_size * n_heads * seq_len * seq_len * head_dim
    throughput = total_ops / (elapsed_time / 1000.0)
    
    return {
        "elapsed_time_ms": elapsed_time,
        "throughput_ops_per_sec": throughput,
        "memory_usage_gb": (q.numel() + k.numel() + v.numel()) * 4 / 1e9
    }
