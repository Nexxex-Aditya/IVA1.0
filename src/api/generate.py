"""Text generation API"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union, Iterator
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.transformer import Transformer
from ..utils.logging import get_logger
from ..utils.config import load_config


logger = get_logger(__name__)


class Generator:
    
    def __init__(
        self,
        model: Transformer,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = self.config.get("max_length", 2048)
        self.temperature = self.config.get("temperature", 0.8)
        self.top_p = self.config.get("top_p", 0.9)
        self.top_k = self.config.get("top_k", 50)
        self.repetition_penalty = self.config.get("repetition_penalty", 1.1)
        self.do_sample = self.config.get("do_sample", True)
        self.num_beams = self.config.get("num_beams", 1)
        self.early_stopping = self.config.get("early_stopping", False)
        self.pad_token_id = self.config.get("pad_token_id", 0)
        self.eos_token_id = self.config.get("eos_token_id", 2)
        self.bos_token_id = self.config.get("bos_token_id", 1)
        
        self.generation_count = 0
        self.total_tokens = 0
        self.total_time = 0.0
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        repetition_penalty = repetition_penalty or self.repetition_penalty
        do_sample = do_sample if do_sample is not None else self.do_sample
        num_beams = num_beams or self.num_beams
        early_stopping = early_stopping if early_stopping is not None else self.early_stopping
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                early_stopping=early_stopping,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                **kwargs
            )
        
        generation_time = time.time() - start_time
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break
        
        self.generation_count += 1
        self.total_tokens += generated_ids.size(1) - input_ids.size(1)
        self.total_time += generation_time
        
        return generated_text
    
    def generate_batch(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        generated_texts = []
        
        for prompt in prompts:
            generated_text = self.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                early_stopping=early_stopping,
                stop_sequences=stop_sequences,
                **kwargs
            )
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def generate_stream(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[str]:
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        repetition_penalty = repetition_penalty or self.repetition_penalty
        do_sample = do_sample if do_sample is not None else self.do_sample
        num_beams = num_beams or self.num_beams
        early_stopping = early_stopping if early_stopping is not None else self.early_stopping
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        generated_ids = input_ids.clone()
        generated_text = prompt
        
        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    return_logits=True,
                    return_value=False
                )
                
                next_token_logits = outputs["logits"][:, -1, :] / temperature
                
                if repetition_penalty != 1.0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, generated_ids, repetition_penalty
                    )
                
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p < 1.0:
                    next_token_logits = self._apply_top_p_filtering(next_token_logits, top_p)
                
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                if next_token.item() == self.eos_token_id:
                    break
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(attention_mask.size(0), 1, device=self.device)
                    ], dim=1)
                
                new_token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                generated_text += new_token_text
                
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            yield generated_text[len(prompt):]
                            return
                
                yield new_token_text
        
        yield generated_text[len(prompt):]
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        unique_tokens = torch.unique(generated_ids)
        
        for token in unique_tokens:
            logits[:, token] = torch.where(
                logits[:, token] < 0,
                logits[:, token] * penalty,
                logits[:, token] / penalty
            )
        
        return logits
    
    def _apply_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "generation_count": self.generation_count,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_tokens_per_sec": self.total_tokens / self.total_time if self.total_time > 0 else 0.0,
            "avg_time_per_generation": self.total_time / self.generation_count if self.generation_count > 0 else 0.0
        }
    
    def reset_stats(self) -> None:
        self.generation_count = 0
        self.total_tokens = 0
        self.total_time = 0.0


def generate(
    model: Transformer,
    tokenizer: Any,
    prompt: str,
    max_length: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    num_beams: int = 1,
    early_stopping: bool = False,
    stop_sequences: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    generator = Generator(model, tokenizer, device, config)
    
    generated_text = generator.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_beams=num_beams,
        early_stopping=early_stopping,
        stop_sequences=stop_sequences
    )
    
    return generated_text


def generate_batch(
    model: Transformer,
    tokenizer: Any,
    prompts: List[str],
    max_length: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    num_beams: int = 1,
    early_stopping: bool = False,
    stop_sequences: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    generator = Generator(model, tokenizer, device, config)
    
    generated_texts = generator.generate_batch(
        prompts=prompts,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_beams=num_beams,
        early_stopping=early_stopping,
        stop_sequences=stop_sequences
    )
    
    return generated_texts


def generate_stream(
    model: Transformer,
    tokenizer: Any,
    prompt: str,
    max_length: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    num_beams: int = 1,
    early_stopping: bool = False,
    stop_sequences: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None
) -> Iterator[str]:
    generator = Generator(model, tokenizer, device, config)
    
    for chunk in generator.generate_stream(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_beams=num_beams,
        early_stopping=early_stopping,
        stop_sequences=stop_sequences
    ):
        yield chunk


def get_generator(
    model: Transformer,
    tokenizer: Any,
    device: Optional[torch.device] = None,
    config: Optional[Dict[str, Any]] = None
) -> Generator:
    return Generator(model, tokenizer, device, config)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text using iva")
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--stop_sequences", type=str, nargs="*", help="Stop sequences")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--config", type=str, help="Configuration file")
    
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    from transformers import AutoTokenizer
    
    model = Transformer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        early_stopping=args.early_stopping,
        stop_sequences=args.stop_sequences,
        device=device,
        config=config
    )
    
    print(generated_text)


if __name__ == "__main__":
    main()
