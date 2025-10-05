
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Optional
import torch
import numpy as np
from transformers import AutoTokenizer

from src.core.transformer import Transformer
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.evals import calculate_perplexity, evaluate_tiny_math, evaluate_tiny_code


logger = get_logger(__name__)


def evaluate_model(
    model: Transformer,
    tokenizer: Any,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    results = {}
    
    model.to(device)
    model.eval()
    
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    ppl = calculate_perplexity(model, tokenizer, test_text, device)
    results["perplexity"] = ppl
    logger.info(f"Perplexity: {ppl:.2f}")
    
    math_results = evaluate_tiny_math(model, tokenizer, device, num_problems=10)
    results["math"] = math_results
    logger.info(f"Math accuracy: {math_results['accuracy']:.2%}")
    
    code_results = evaluate_tiny_code(model, tokenizer, device, num_problems=10)
    results["code"] = code_results
    logger.info(f"Code accuracy: {code_results['accuracy']:.2%}")
    
    generation_results = test_generation_quality(model, tokenizer, device)
    results["generation"] = generation_results
    logger.info(f"Generation quality: {generation_results['avg_length']:.1f} tokens, {generation_results['coherence_score']:.2f} coherence")
    
    if config.get("model", {}).get("hrm", {}).get("enable", False):
        hrm_results = test_hrm_controller(model, tokenizer, device)
        results["hrm"] = hrm_results
        logger.info(f"HRM halt rate: {hrm_results['halt_rate']:.2%}, avg steps: {hrm_results['avg_steps']:.1f}")
    
    if config.get("model", {}).get("moe", {}).get("enable", False):
        moe_results = test_moe_router(model, tokenizer, device)
        results["moe"] = moe_results
        logger.info(f"MoE load balance: {moe_results['load_balance']:.2f}, overflow: {moe_results['overflow_rate']:.2%}")
    
    return results


def test_generation_quality(
    model: Transformer,
    tokenizer: Any,
    device: torch.device
) -> Dict[str, Any]:
    prompts = [
        "The capital of France is",
        "In the year 2024,",
        "The following is a Python function:",
        "To solve this problem,",
        "The main advantage of"
    ]
    
    generated_lengths = []
    coherence_scores = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_length=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
        
        generated_lengths.append(len(generated_text.split()))
        
        coherence_score = calculate_coherence_score(generated_text)
        coherence_scores.append(coherence_score)
    
    return {
        "avg_length": np.mean(generated_lengths),
        "std_length": np.std(generated_lengths),
        "coherence_score": np.mean(coherence_scores),
        "coherence_std": np.std(coherence_scores)
    }


def calculate_coherence_score(text: str) -> float:
    if not text:
        return 0.0
    
    score = 0.0     # simple heuristics for coherence
    
    if text[0].isupper(): # check for proper sentence structure
        score += 0.2
    
    if text.endswith(('.', '!', '?')): # check for sentence endings
        score += 0.2
    
    # check for reasonable word count
    word_count = len(text.split())
    if 5 <= word_count <= 50:
        score += 0.2
    elif word_count > 50:
        score += 0.1
    
    # check for repeated words (penalty)
    words = text.lower().split()
    unique_words = set(words)
    if len(words) > 0:
        repetition_ratio = len(unique_words) / len(words)
        score += repetition_ratio * 0.4
    
    return min(score, 1.0)


def test_hrm_controller(
    model: Transformer,
    tokenizer: Any,
    device: torch.device
) -> Dict[str, Any]:
    # test prompts that should trigger different halt rates
    test_prompts = [
        "What is 2+2?",  # simple question
        "Explain the theory of relativity",  # complex question
        "The quick brown fox",  # simple completion
        "Implement a quicksort algorithm",  # Ccomplex task
    ]
    
    halt_rates = []
    avg_steps = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                return_logits=True,
                return_value=True
            )
        
        if "hrm_state" in outputs:
            hrm_state = outputs["hrm_state"]
            halt_rate = hrm_state["halted"].float().mean().item()
            avg_step = hrm_state["steps"].float().mean().item()
            
            halt_rates.append(halt_rate)
            avg_steps.append(avg_step)
    
    return {
        "halt_rate": np.mean(halt_rates) if halt_rates else 0.0,
        "halt_std": np.std(halt_rates) if halt_rates else 0.0,
        "avg_steps": np.mean(avg_steps) if avg_steps else 0.0,
        "steps_std": np.std(avg_steps) if avg_steps else 0.0
    }


def test_moe_router(
    model: Transformer,
    tokenizer: Any,
    device: torch.device
) -> Dict[str, Any]:
    # prompts that should trigger different expert usage
    test_prompts = [
        "The quick brown fox jumps over the lazy dog",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "The capital of France is Paris, the capital of Germany is Berlin",
        "import numpy as np; x = np.array([1, 2, 3])",
    ]
    
    expert_loads = []
    overflow_rates = []
    route_entropies = []
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                return_logits=True,
                return_value=True
            )
        
        if "moe_state" in outputs:
            moe_state = outputs["moe_state"]
            
            expert_load = moe_state["expert_load"].float()
            load_balance = 1.0 - expert_load.std() / expert_load.mean()
            expert_loads.append(load_balance.item())
            
            overflow_rate = moe_state["overflow"].float().mean().item()
            overflow_rates.append(overflow_rate)
            
            route_probs = moe_state["route_probs"]
            entropy = -(route_probs * torch.log(route_probs + 1e-8)).sum(dim=-1).mean().item()
            route_entropies.append(entropy)
    
    return {
        "load_balance": np.mean(expert_loads) if expert_loads else 0.0,
        "load_balance_std": np.std(expert_loads) if expert_loads else 0.0,
        "overflow_rate": np.mean(overflow_rates) if overflow_rates else 0.0,
        "overflow_std": np.std(overflow_rates) if overflow_rates else 0.0,
        "route_entropy": np.mean(route_entropies) if route_entropies else 0.0,
        "entropy_std": np.std(route_entropies) if route_entropies else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    if args.config:
        config = load_config(args.config)
    else:
        config = load_config("configs/train_sft.yaml")
    
    logger.info(f"Loading model from {args.model_path}")
    model = Transformer.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    logger.info("Starting evaluation")
    start_time = time.time()
    
    results = evaluate_model(model, tokenizer, device, config)
    
    evaluation_time = time.time() - start_time
    results["evaluation_time"] = evaluation_time
    
    print("\nEvaluation Results")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Math Accuracy: {results['math']['accuracy']:.2%}")
    print(f"Code Accuracy: {results['code']['accuracy']:.2%}")
    print(f"Generation Quality: {results['generation']['coherence_score']:.2f}")
    
    if "hrm" in results:
        print(f"HRM Halt Rate: {results['hrm']['halt_rate']:.2%}")
        print(f"HRM Avg Steps: {results['hrm']['avg_steps']:.1f}")
    
    if "moe" in results:
        print(f"MoE Load Balance: {results['moe']['load_balance']:.2f}")
        print(f"MoE Overflow Rate: {results['moe']['overflow_rate']:.2%}")
    
    print(f"Evaluation Time: {evaluation_time:.2f}s")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
