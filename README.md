# IVA: Hierarchical Reasoning + Efficiency Engine

IVA is a research grade reasoning and efficiency engine wrapping large pretrained models (Qwen/Mistral/Llama) with:
- Hierarchical Reasoning Model controller that plans and adaptively halts, modulates attention sink strength, and MoE router temperature.
- Triton fused kernels (attention, RMSNorm, SwiGLU/Linear, KV ops) with autotune and paged KV for long context memory efficiency.
- Mixture of Experts with Top 2 gating, capacity planning, load/importance aux loss, Expert Parallel hooks (optional).
- Reinforcement learning (PPO/GRPO) plus a new HRM RPO variant with reasoning aware reward shaping and memory regularization.
- Quantization (INT4 weight only inference; FP8 runtime on A100 with guards), sliding window and paged KV.

## Architecture

### Core Components

- **HRM Controller**: ACT (Adaptive Computation Time) + attention sink control + router temperature
- **Triton Kernels**: Fused RMSNorm, Attention, SwiGLU/Linear, paged KV operations
- **Mixture of Experts**: Top 2 routing with capacity planning and ExpertParallel support
- **Quantization**: FP8 runtime for matmuls, INT4 weight only inference
- **Training**: SFT > RM > PPO/GRPO + HRM-RPO with teacher KD option

### Key Features

- **Single GPU First**: Optimized for T4/Colab, multi GPU ready for A100
- **High Performance**: Triton kernels with autotuning and microbenchmarks

## Performance Benchmarks and Acceptance Criteria

### Kernel Speedups (vs PyTorch)

| Operation | Triton Speedup | Notes |
|-----------|----------------|-------|
| RMSNorm | ≥1.25× | Fused forward/backward | Expecting ( Exprimental )
| Attention | ≥1.1× | Train path with dropout | Expecting ( Exprimental )
| SwiGLU | ≥1.15× | GEMM+bias+activation fusion | Expecting ( Exprimental )

### Memory Efficiency

- **Paged KV Cache**: Efficient memory usage for long sequences
- **Activation Checkpointing**: Reduced memory footprint during training
- **Gradient Accumulation**: Support for large effective batch sizes

## Configuration

### Model Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `model_dense.yaml` | Dense transformer | General purpose |
| `model_moe.yaml` | Mixture of Experts | High capacity |
| `hrm.yaml` | HRM controller | Reasoning tasks |
| `quant_infer.yaml` | Quantized inference | Production deployment |

### Training Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `train_sft.yaml` | Supervised Fine Tuning | Base training |
| `train_ppo.yaml` | PPO training | RL fine tuning |
| `multimodal.yaml` | Multimodal adapter | Vision tasks |

## Telemetry & Monitoring

### Metrics Schema

```json
{
  "speed": {
    "tokens_per_sec": 1250.5,
    "step_time_ms": 45.2
  },
  "memory": {
    "vram_gb": 12.8
  },
  "quality": {
    "ppl": 2.34,
    "eval_passk": 0.85
  },
  "hrm": {
    "halt_rate": 0.45,
    "avg_steps": 3.2
  },
  "moe": {
    "load_per_expert": [0.23, 0.27, 0.25, 0.25],
    "overflow_pct": 0.02,
    "route_entropy": 1.89
  },
  "quant": {
    "fp8_amax_drift": 0.01
  }
}
```

## Development

### Setup

```bash
# Clone and install
git clone <repo-url>
cd iva
pip install -e .

# Install development dependencies
pip install -e ".[colab]"
```

### Kernel Benchmarking

```bash
# Profile Triton kernels
python scripts/profile_kernels.py

# Expected outputs:
# RMSNorm Triton: 1.25x speedup vs PyTorch 
# Attention Triton: 1.1x speedup vs PyTorch 
```

## API Reference

### Core Transformer

```python
from src.core.transformer import Transformer, TransformerConfig

# Create model
config = TransformerConfig(
    d_model=1024,
    n_layers=16,
    n_heads=16,
    vocab_size=32000,
    seq_len=4096,
    attn_impl="triton",
    norm_impl="triton",
    moe_enable=False,
    hrm_enable=True
)
model = Transformer(config)

# Forward pass
output = model.forward(
    input_ids=torch.tensor([[1, 2, 3, 4]]),
    attention_mask=None,
    kv_cache=None,
    return_logits=True,
    return_value=False
)
```

### Generation API

```python
from src.api.generate import generate

# Generate text
text = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="The capital of France is",
    max_length=100,
    temperature=0.8,
    top_p=0.9
)
```

### FastAPI Serving

```python
from src.api.serve import app

# Start server
uvicorn src.api.serve:app --host 0.0.0.0 --port 8000

## Mathematical Appendix

- ACT halting: `p_h = σ(w^T h + b)`, expected compute `E[S]` approximated via cumulative mass; penalty `λ ∑_t p_h(t)`.
- Sink modulation: `Â = A + α_s · 1_sink`.
- Router temperature: `softmax(x/τ)`.
- KD loss: `L_KD = τ^2 · KL(σ(z_T/τ) || σ(z_S/τ))`.
- PPO surrogate: `L_PPO = E[min(r_t(θ) Â_t, clip(r_t(θ),1−ε,1+ε) Â_t)]`.
- HRM-RPO reward shaping: `R'_t = R_t + β1 r_halt + β2 r_entropy + β3 r_mem`.
- Contrastive memory term: `L_mem = NT-Xent(z_H(i), z_H(j))`.
- MoE aux loss: `L_aux = ∑_e p̄_e · ℓ̄_e`.

## Validation (One shot)

Run the end to end smoke test with acceptance thresholds:

```bash
python scripts/validate_end_to_end.py --device auto
```

Checks enforced:
- RMSNorm ≥ 1.25× torch; attention ≥ 1.1× torch.
- HRM: halt_rate in [0.3, 0.7] on tiny task; no NaNs.
- MoE: per expert token load std/mean ≤ 1.0; overflow% ≤ 5% (capacity_factor=1.25).
- Quant: PPL drift ≤ 5% vs FP16; INT4 generation completes 64 tokens.
- Colab safe defaults: FP8 disabled on T4.
```

## Troubleshooting

### Common Issues

#### FP8/A100 Compatibility
- Ensure CUDA 12.1+ and compatible drivers
- Check Triton version compatibility
- Verify A100 availability: `nvidia-smi`

#### INT4 Weight-Only Inference
- Install compatible quantization libraries
- Check model compatibility with INT4 scheme
- Verify memory requirements

#### NCCL Multi GPU Issues
- Ensure proper NCCL installation
- Check network configuration
- Verify GPU visibility: `CUDA_VISIBLE_DEVICES`

#### Colab T4 Limitations
- Use `attn_impl="torch"` for T4
- Reduce `d_model` and `n_layers` if needed
- Enable gradient checkpointing

### Performance Tuning

#### Memory Optimization
```yaml
# configs/quant_infer.yaml
precision: fp16
kv_precision: fp8
paged_kv_cache: true
activation_checkpointing: true
```

#### Speed Optimization
```yaml
# configs/model_dense.yaml
attn_impl: triton
norm_impl: triton
moe_enable: false
hrm_enable: true
```
