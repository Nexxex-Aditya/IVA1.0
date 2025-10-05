from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    GPT2Tokenizer,
    GPT2TokenizerFast
)

from ..utils.logging import get_logger


logger = get_logger(__name__)


def load_tokenizer(
    tokenizer_name: str,
    trust_remote_code: bool = False,
    **kwargs
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        logger.info(f"Loaded tokenizer: {tokenizer_name}")
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
        return create_tokenizer()


def create_tokenizer(
    vocab_size: int = 32000,
    model_max_length: int = 4096
) -> PreTrainedTokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    if vocab_size != len(tokenizer):
        tokenizer.add_tokens([f"<token_{i}>" for i in range(len(tokenizer), vocab_size)])
    
    tokenizer.model_max_length = model_max_length
    
    special_tokens = {
        "pad_token": "<pad>",
        "eos_token": "<eos>",
        "bos_token": "<bos>",
        "unk_token": "<unk>",
        "mask_token": "<mask>"
    }
    
    tokenizer.add_special_tokens(special_tokens)
    
    logger.info(f"Created tokenizer with vocab size: {len(tokenizer)}")
    return tokenizer


def tokenize_text(
    text: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    max_length: Optional[int] = None,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    if max_length is None:
        max_length = tokenizer.model_max_length
    
    return tokenizer(
        text,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )


def detokenize_text(
    token_ids: torch.Tensor,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    skip_special_tokens: bool = True
) -> str:
    if token_ids.dim() > 1:
        token_ids = token_ids.squeeze(0)
    
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def tokenize_batch(
    texts: List[str],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    max_length: Optional[int] = None,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt"
) -> Dict[str, torch.Tensor]:
    if max_length is None:
        max_length = tokenizer.model_max_length
    
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors
    )


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    return (input_ids != pad_token_id).long()


def create_labels(
    input_ids: torch.Tensor,
    ignore_index: int = -100,
    shift: bool = True
) -> torch.Tensor:
    if shift:
        labels = input_ids.clone()
        labels[..., :-1] = input_ids[..., 1:]
        labels[..., -1] = ignore_index
    else:
        labels = input_ids.clone()
    
    return labels


def create_packed_sequence(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    max_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len = input_ids.shape
    
    flat_input_ids = input_ids.view(-1)
    flat_attention_mask = attention_mask.view(-1)
    flat_labels = labels.view(-1)
    
    non_pad_mask = flat_attention_mask.bool()
    packed_input_ids = flat_input_ids[non_pad_mask]
    packed_labels = flat_labels[non_pad_mask]
    
    num_chunks = (len(packed_input_ids) + max_length - 1) // max_length
    
    total_length = num_chunks * max_length
    padded_input_ids = torch.full((total_length,), 0, dtype=packed_input_ids.dtype)
    padded_labels = torch.full((total_length,), -100, dtype=packed_labels.dtype)
    
    padded_input_ids[:len(packed_input_ids)] = packed_input_ids
    padded_labels[:len(packed_labels)] = packed_labels
    
    packed_input_ids = padded_input_ids.view(num_chunks, max_length)
    packed_labels = padded_labels.view(num_chunks, max_length)
    packed_attention_mask = (packed_input_ids != 0).long()
    
    return packed_input_ids, packed_attention_mask, packed_labels


def add_special_tokens(
    input_ids: torch.Tensor,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    add_bos: bool = True,
    add_eos: bool = True
) -> torch.Tensor:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    batch_size, seq_len = input_ids.shape
    
    new_length = seq_len
    if add_bos:
        new_length += 1
    if add_eos:
        new_length += 1
    
    new_input_ids = torch.zeros(batch_size, new_length, dtype=input_ids.dtype)
    
    start_idx = 0
    if add_bos:
        new_input_ids[:, 0] = tokenizer.bos_token_id
        start_idx = 1
    
    new_input_ids[:, start_idx:start_idx + seq_len] = input_ids
    
    if add_eos:
        new_input_ids[:, start_idx + seq_len] = tokenizer.eos_token_id
    
    return new_input_ids


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        end = i + 1
        mask[i, start:end] = 1
    
    return mask


def create_causal_mask(
    seq_len: int,
    device: torch.device
) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def create_attention_sink_mask(
    seq_len: int,
    sink_tokens: int = 4,
    device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, device=device)
    
    for i in range(sink_tokens):
        mask[i, :i+1] = 1
    
    for i in range(sink_tokens, seq_len):
        mask[i, :sink_tokens] = 1
        mask[i, sink_tokens:i+1] = 1
    
    return mask


def save_tokenizer(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    path: Union[str, Path]
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    tokenizer.save_pretrained(path)
    logger.info(f"Saved tokenizer to {path}")


def load_tokenizer_from_path(
    path: Union[str, Path],
    **kwargs
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {path}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(path), **kwargs)
    logger.info(f"Loaded tokenizer from {path}")
    return tokenizer


def get_tokenizer_info(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Dict[str, Any]:
    return {
        "vocab_size": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
    }


def create_byte_bpe_tokenizer(
    vocab_size: int = 32000,
    model_max_length: int = 4096
) -> PreTrainedTokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    if vocab_size != len(tokenizer):
        tokenizer.add_tokens([f"<byte_{i}>" for i in range(len(tokenizer), vocab_size)])
    
    tokenizer.model_max_length = model_max_length
    
    special_tokens = {
        "pad_token": "<pad>",
        "eos_token": "<eos>",
        "bos_token": "<bos>",
        "unk_token": "<unk>",
        "mask_token": "<mask>"
    }
    
    tokenizer.add_special_tokens(special_tokens)
    
    logger.info(f"Created Byte-BPE tokenizer with vocab size: {len(tokenizer)}")
    return tokenizer
