
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .logging import get_logger


logger = get_logger(__name__)


def evaluate_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch.get("labels", input_ids)
                if labels is not None:
                    labels = labels.to(device)
            else:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None
                labels = batch[2].to(device) if len(batch) > 2 else input_ids
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                if attention_mask is not None:
                    token_count = attention_mask[..., 1:].sum().item()
                else:
                    token_count = shift_labels.numel() - (shift_labels == -100).sum().item()
                
                total_loss += loss.item() * token_count
                total_tokens += token_count
            
            num_batches += 1
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        perplexity = float('inf')
    
    return {
        "perplexity": perplexity,
        "loss": avg_loss if total_tokens > 0 else float('inf'),
        "total_tokens": total_tokens,
        "num_batches": num_batches
    }


def evaluate_reasoning(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    max_batches: Optional[int] = None,
    max_length: int = 512
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    total_time = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating reasoning")):
            if max_batches and batch_idx >= max_batches:
                break
            
            if isinstance(batch, dict):
                prompts = batch["prompt"]
                answers = batch["answer"]
            else:
                prompts = batch[0]
                answers = batch[1]
            
            for prompt, answer in zip(prompts, answers):
                start_time = time.time()
                
                response = generate_response(
                    model, tokenizer, prompt, device, max_length=max_length
                )
                
                total_time += time.time() - start_time
                
                if is_answer_correct(response, answer):
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time": avg_time
    }


def evaluate_multimodal(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    max_batches: Optional[int] = None,
    max_length: int = 512
) -> Dict[str, float]:
    model.eval()
    vision_correct = 0
    audio_correct = 0
    total = 0
    total_time = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating multimodal")):
            if max_batches and batch_idx >= max_batches:
                break
            
            if isinstance(batch, dict):
                prompts = batch["prompt"]
                images = batch.get("image", None)
                audio = batch.get("audio", None)
                vision_answers = batch.get("vision_answer", None)
                audio_answers = batch.get("audio_answer", None)
            else:
                prompts = batch[0]
                images = batch[1] if len(batch) > 1 else None
                audio = batch[2] if len(batch) > 2 else None
                vision_answers = batch[3] if len(batch) > 3 else None
                audio_answers = batch[4] if len(batch) > 4 else None
            
            for i, prompt in enumerate(prompts):
                start_time = time.time()
                
                response = generate_multimodal_response(
                    model, tokenizer, prompt, device,
                    image=images[i] if images is not None else None,
                    audio=audio[i] if audio is not None else None,
                    max_length=max_length
                )
                
                total_time += time.time() - start_time
                
                if vision_answers is not None and is_answer_correct(response, vision_answers[i]):
                    vision_correct += 1
                
                if audio_answers is not None and is_answer_correct(response, audio_answers[i]):
                    audio_correct += 1
                
                total += 1
    
    vision_accuracy = vision_correct / total if total > 0 else 0.0
    audio_accuracy = audio_correct / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0
    
    return {
        "vision_accuracy": vision_accuracy,
        "audio_accuracy": audio_accuracy,
        "vision_correct": vision_correct,
        "audio_correct": audio_correct,
        "total": total,
        "avg_time": avg_time
    }


def generate_response(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_length: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response


def generate_multimodal_response(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    image: Optional[torch.Tensor] = None,
    audio: Optional[torch.Tensor] = None,
    max_length: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    kwargs = {}
    if image is not None:
        kwargs["image"] = image.to(device)
    if audio is not None:
        kwargs["audio"] = audio.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response


def is_answer_correct(response: str, answer: str) -> bool:
    response = response.strip().lower()
    answer = answer.strip().lower()
    
    if response == answer:
        return True
    
    if answer in response:
        return True
    
    if response in answer:
        return True
    
    return False


def calculate_bleu_score(
    predictions: List[str],
    references: List[str],
    n_gram: int = 4
) -> float:
    from collections import Counter
    import math
    
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        return Counter(ngrams)
    
    def precision(pred_ngrams: Counter, ref_ngrams: Counter) -> float:
        overlap = 0
        total = sum(pred_ngrams.values())
        
        for ngram, count in pred_ngrams.items():
            overlap += min(count, ref_ngrams[ngram])
        
        return overlap / total if total > 0 else 0.0
    
    precisions = []
    for n in range(1, n_gram + 1):
        pred_ngrams = Counter()
        ref_ngrams = Counter()
        
        for pred, ref in zip(predictions, references):
            pred_ngrams += get_ngrams(pred, n)
            ref_ngrams += get_ngrams(ref, n)
        
        precisions.append(precision(pred_ngrams, ref_ngrams))
    
    if any(p == 0 for p in precisions):
        return 0.0
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    pred_length = sum(len(p.split()) for p in predictions)
    ref_length = sum(len(r.split()) for r in references)
    
    if pred_length >= ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_length / pred_length)
    
    return bp * geometric_mean


def calculate_rouge_score(
    predictions: List[str],
    references: List[str],
    rouge_type: str = "rouge-l"
) -> float:
    def get_lcs(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def rouge_l(pred: str, ref: str) -> float:
        pred_words = pred.split()
        ref_words = ref.split()
        
        if not pred_words or not ref_words:
            return 0.0
        
        lcs_len = get_lcs(pred_words, ref_words)
        
        precision = lcs_len / len(pred_words)
        recall = lcs_len / len(ref_words)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    if rouge_type == "rouge-l":
        scores = [rouge_l(pred, ref) for pred, ref in zip(predictions, references)]
        return sum(scores) / len(scores) if scores else 0.0
    else:
        raise ValueError(f"Unsupported ROUGE type: {rouge_type}")


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    ppl_metrics = evaluate_perplexity(model, dataloader, device, max_batches)
    reasoning_metrics = evaluate_reasoning(model, dataloader, device, tokenizer, max_batches)
    all_metrics = {**ppl_metrics, **reasoning_metrics}
    return all_metrics
