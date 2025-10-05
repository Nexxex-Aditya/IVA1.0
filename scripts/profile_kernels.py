
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
import triton
import triton.language as tl

from src.kernels_t.attention_triton import attention_triton
from src.kernels_t.rmsnorm_triton import rmsnorm_triton
from src.kernels_t.swiglu_linear_triton import swiglu_linear_triton
from src.utils.logging import get_logger


logger = get_logger(__name__)


def benchmark_rmsnorm(
    batch_size: int = 32,
    seq_len: int = 2048,
    d_model: int = 1024,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict[str, float]:
    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)
    weight = torch.randn(d_model, dtype=dtype, device=device)
    
    for _ in range(10):
        _ = rmsnorm_triton(x, weight)
        _ = torch.nn.functional.rms_norm(x, weight, eps=1e-6)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        _ = rmsnorm_triton(x, weight)
    
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        _ = torch.nn.functional.rms_norm(x, weight, eps=1e-6)
    
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    
    speedup = torch_time / triton_time
    
    return {
        "triton_time": triton_time,
        "torch_time": torch_time,
        "speedup": speedup,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model
    }


def benchmark_attention(
    batch_size: int = 32,
    seq_len: int = 2048,
    n_heads: int = 16,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict[str, float]:
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    
    for _ in range(10):
        _ = attention_triton(q, k, v, mask)
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        _ = attention_triton(q, k, v, mask)
    
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    
    speedup = torch_time / triton_time
    
    return {
        "triton_time": triton_time,
        "torch_time": torch_time,
        "speedup": speedup,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_heads": n_heads,
        "head_dim": head_dim
    }


def benchmark_swiglu_linear(
    batch_size: int = 32,
    seq_len: int = 2048,
    d_model: int = 1024,
    hidden_dim: int = 4096,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict[str, float]:
    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)
    weight = torch.randn(d_model, hidden_dim, dtype=dtype, device=device)
    bias = torch.randn(hidden_dim, dtype=dtype, device=device)
    
    for _ in range(10):
        _ = swiglu_linear_triton(x, weight, bias)
        _ = torch.nn.functional.linear(x, weight, bias)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        _ = swiglu_linear_triton(x, weight, bias)
    
    torch.cuda.synchronize()
    triton_time = time.time() - start_time
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        _ = torch.nn.functional.linear(x, weight, bias)
    
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    
    speedup = torch_time / triton_time
    
    return {
        "triton_time": triton_time,
        "torch_time": torch_time,
        "speedup": speedup,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "hidden_dim": hidden_dim
    }


def run_comprehensive_benchmark(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Dict[str, Any]:
    results = {}
    
    configs = [
        {"batch_size": 1, "seq_len": 1024, "d_model": 512},
        {"batch_size": 8, "seq_len": 2048, "d_model": 1024},
        {"batch_size": 32, "seq_len": 4096, "d_model": 2048},
    ]
    
    logger.info("Benchmarking RMSNorm kernels")
    rmsnorm_results = []
    
    for config in configs:
        result = benchmark_rmsnorm(
            batch_size=config["batch_size"],
            seq_len=config["seq_len"],
            d_model=config["d_model"],
            dtype=dtype,
            device=device
        )
        rmsnorm_results.append(result)
        logger.info(f"RMSNorm {config}: {result['speedup']:.2f}x speedup")
    
    results["rmsnorm"] = rmsnorm_results
    
    logger.info("Benchmarking Attention kernels")
    attention_results = []
    
    for config in configs:
        result = benchmark_attention(
            batch_size=config["batch_size"],
            seq_len=config["seq_len"],
            n_heads=16,
            head_dim=64,
            dtype=dtype,
            device=device
        )
        attention_results.append(result)
        logger.info(f"Attention {config}: {result['speedup']:.2f}x speedup")
    
    results["attention"] = attention_results
    
    logger.info("Benchmarking SwiGLU Linear kernels")
    swiglu_results = []
    
    for config in configs:
        result = benchmark_swiglu_linear(
            batch_size=config["batch_size"],
            seq_len=config["seq_len"],
            d_model=config["d_model"],
            hidden_dim=config["d_model"] * 4,
            dtype=dtype,
            device=device
        )
        swiglu_results.append(result)
        logger.info(f"SwiGLU Linear {config}: {result['speedup']:.2f}x speedup")
    
    results["swiglu_linear"] = swiglu_results
    
    return results


def check_performance_assertions(results: Dict[str, Any]) -> bool:
    all_passed = True
    
    rmsnorm_speedups = [r["speedup"] for r in results["rmsnorm"]]
    min_rmsnorm_speedup = min(rmsnorm_speedups)
    
    if min_rmsnorm_speedup >= 1.25:
        logger.info(f"RMSNorm speedup assertion passed: {min_rmsnorm_speedup:.2f}x >= 1.25x")
    else:
        logger.error(f"RMSNorm speedup assertion failed: {min_rmsnorm_speedup:.2f}x < 1.25x")
        all_passed = False
    
    attention_speedups = [r["speedup"] for r in results["attention"]]
    min_attention_speedup = min(attention_speedups)
    
    if min_attention_speedup >= 1.1:
        logger.info(f"Attention speedup assertion passed: {min_attention_speedup:.2f}x >= 1.1x")
    else:
        logger.error(f"Attention speedup assertion failed: {min_attention_speedup:.2f}x < 1.1x")
        all_passed = False
    
    swiglu_speedups = [r["speedup"] for r in results["swiglu_linear"]]
    min_swiglu_speedup = min(swiglu_speedups)
    
    if min_swiglu_speedup >= 1.0:
        logger.info(f"SwiGLU Linear speedup assertion passed: {min_swiglu_speedup:.2f}x >= 1.0x")
    else:
        logger.error(f"SwiGLU Linear speedup assertion failed: {min_swiglu_speedup:.2f}x < 1.0x")
        all_passed = False
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Profile kernels")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--check-assertions", action="store_true", help="Check performance assertions")
    
    args = parser.parse_args()
    
    device = args.device
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info("Starting kernel profiling")
    results = run_comprehensive_benchmark(device=device, dtype=dtype)
    
    print("\nKernel Profiling Summary")
    
    rmsnorm_speedups = [r["speedup"] for r in results["rmsnorm"]]
    print(f"RMSNorm: {min(rmsnorm_speedups):.2f}x - {max(rmsnorm_speedups):.2f}x speedup")
    
    attention_speedups = [r["speedup"] for r in results["attention"]]
    print(f"Attention: {min(attention_speedups):.2f}x - {max(attention_speedups):.2f}x speedup")
    
    swiglu_speedups = [r["speedup"] for r in results["swiglu_linear"]]
    print(f"SwiGLU Linear: {min(swiglu_speedups):.2f}x - {max(swiglu_speedups):.2f}x speedup")
    
    if args.check_assertions:
        print("\nPerformance Assertions")
        all_passed = check_performance_assertions(results)
        
        if all_passed:
            print("All performance assertions passed")
        else:
            print("Some performance assertions failed")
            exit(1)
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    logger.info("Kernel profiling completed successfully")


if __name__ == "__main__":
    main()
