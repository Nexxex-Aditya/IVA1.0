"""Data collation utilities for T_HRM_OPT_ADV."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..utils.logging import get_logger


logger = get_logger(__name__)


class DataCollator:
    
    def __init__(
        self,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        padding_side: str = "right"
    ):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.padding_side = padding_side
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [sample["input_ids"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        labels = [sample.get("labels", sample["input_ids"]) for sample in batch]
        
        input_ids = self._pad_sequences(input_ids, self.pad_token_id)
        attention_mask = self._pad_sequences(attention_mask, 0)
        labels = self._pad_sequences(labels, self.ignore_index)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _pad_sequences(
        self,
        sequences: List[torch.Tensor],
        pad_value: int
    ) -> torch.Tensor:
        if self.padding_side == "right":
            return pad_sequence(sequences, batch_first=True, padding_value=pad_value)
        else:
            # Left padding
            max_len = max(seq.size(0) for seq in sequences)
            padded = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
            
            for i, seq in enumerate(sequences):
                if self.padding_side == "left":
                    padded[i, -len(seq):] = seq
                else:
                    padded[i, :len(seq)] = seq
            
            return padded


class PackedDataCollator(DataCollator):
    
    def __init__(
        self,
        max_length: int = 512,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        packing: bool = True
    ):
        super().__init__(pad_token_id, ignore_index)
        self.max_length = max_length
        self.packing = packing
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not self.packing:
            return super().__call__(batch)
        
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        for sample in batch:
            all_input_ids.append(sample["input_ids"])
            all_attention_mask.append(sample["attention_mask"])
            all_labels.append(sample.get("labels", sample["input_ids"]))
        
        input_ids = torch.cat(all_input_ids)
        attention_mask = torch.cat(all_attention_mask)
        labels = torch.cat(all_labels)
        
        packed_input_ids, packed_attention_mask, packed_labels = self._pack_sequences(
            input_ids, attention_mask, labels
        )
        
        return {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "labels": packed_labels
        }
    
    def _pack_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        non_pad_mask = attention_mask.bool()
        packed_input_ids = input_ids[non_pad_mask]
        packed_labels = labels[non_pad_mask]
        
        num_chunks = (len(packed_input_ids) + self.max_length - 1) // self.max_length
        
        total_length = num_chunks * self.max_length
        padded_input_ids = torch.full((total_length,), self.pad_token_id, dtype=packed_input_ids.dtype)
        padded_labels = torch.full((total_length,), self.ignore_index, dtype=packed_labels.dtype)
        
        padded_input_ids[:len(packed_input_ids)] = packed_input_ids
        padded_labels[:len(packed_labels)] = packed_labels
        
        packed_input_ids = padded_input_ids.view(num_chunks, self.max_length)
        packed_labels = padded_labels.view(num_chunks, self.max_length)
        packed_attention_mask = (packed_input_ids != self.pad_token_id).long()
        
        return packed_input_ids, packed_attention_mask, packed_labels


class MultimodalDataCollator(DataCollator):
    
    def __init__(
        self,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        image_processor: Optional[Any] = None
    ):
        super().__init__(pad_token_id, ignore_index)
        self.image_processor = image_processor
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [sample["input_ids"] for sample in batch]
        attention_mask = [sample["attention_mask"] for sample in batch]
        labels = [sample.get("labels", sample["input_ids"]) for sample in batch]
        
        input_ids = self._pad_sequences(input_ids, self.pad_token_id)
        attention_mask = self._pad_sequences(attention_mask, 0)
        labels = self._pad_sequences(labels, self.ignore_index)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        if "image" in batch[0]:
            images = [sample["image"] for sample in batch]
            if self.image_processor:
                images = self.image_processor(images)
            else:
                images = torch.stack(images)
            result["image"] = images
        
        if "audio" in batch[0]:
            audio = [sample["audio"] for sample in batch]
            audio = torch.stack(audio)
            result["audio"] = audio
        
        return result


class AttentionSinkCollator(DataCollator):
    
    def __init__(
        self,
        sink_tokens: int = 4,
        pad_token_id: int = 0,
        ignore_index: int = -100
    ):
        super().__init__(pad_token_id, ignore_index)
        self.sink_tokens = sink_tokens
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        processed_batch = []
        for sample in batch:
            processed_sample = self._add_sink_tokens(sample)
            processed_batch.append(processed_sample)
        
        # Use parent collator
        return super().__call__(processed_batch)
    
    def _add_sink_tokens(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample.get("labels", input_ids)
        
        sink_input_ids = torch.full((self.sink_tokens,), self.pad_token_id, dtype=input_ids.dtype)
        sink_attention_mask = torch.ones(self.sink_tokens, dtype=attention_mask.dtype)
        sink_labels = torch.full((self.sink_tokens,), self.ignore_index, dtype=labels.dtype)
        
        new_input_ids = torch.cat([sink_input_ids, input_ids])
        new_attention_mask = torch.cat([sink_attention_mask, attention_mask])
        new_labels = torch.cat([sink_labels, labels])
        
        return {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "labels": new_labels
        }


class SlidingWindowCollator(DataCollator):
    
    def __init__(
        self,
        window_size: int = 512,
        pad_token_id: int = 0,
        ignore_index: int = -100
    ):
        super().__init__(pad_token_id, ignore_index)
        self.window_size = window_size
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        processed_batch = []
        for sample in batch:
            processed_sample = self._apply_sliding_window(sample)
            processed_batch.append(processed_sample)
        
        return super().__call__(processed_batch)
    
    def _apply_sliding_window(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample.get("labels", input_ids)
        
        seq_len = len(input_ids)
        
        if seq_len <= self.window_size:
            return sample
        
        truncated_input_ids = input_ids[-self.window_size:]
        truncated_attention_mask = attention_mask[-self.window_size:]
        truncated_labels = labels[-self.window_size:]
        
        return {
            "input_ids": truncated_input_ids,
            "attention_mask": truncated_attention_mask,
            "labels": truncated_labels
        }


def create_data_collator(
    collator_type: str = "default",
    **kwargs
) -> DataCollator:
    if collator_type == "default":
        return DataCollator(**kwargs)
    elif collator_type == "packed":
        return PackedDataCollator(**kwargs)
    elif collator_type == "multimodal":
        return MultimodalDataCollator(**kwargs)
    elif collator_type == "attention_sink":
        return AttentionSinkCollator(**kwargs)
    elif collator_type == "sliding_window":
        return SlidingWindowCollator(**kwargs)
    else:
        raise ValueError(f"Unknown collator type: {collator_type}")


def create_data_loader(
    dataset: torch.utils.data.Dataset,
    collator: DataCollator,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator
    )
