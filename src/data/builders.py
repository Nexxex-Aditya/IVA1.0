from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset as hf_load_dataset, Dataset as HFDataset

from .tokenize import load_tokenizer
from ..utils.logging import get_logger


logger = get_logger(__name__)


class SyntheticDataset(Dataset):
    
    def __init__(
        self,
        size: int = 1000,
        max_length: int = 512,
        vocab_size: int = 32000,
        task_type: str = "language_modeling"
    ):
        self.size = size
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.task_type = task_type
        
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        for i in range(self.size):
            if self.task_type == "language_modeling":
                sample = self._generate_lm_sample()
            elif self.task_type == "reasoning":
                sample = self._generate_reasoning_sample()
            elif self.task_type == "multimodal":
                sample = self._generate_multimodal_sample()
            else:
                sample = self._generate_lm_sample()
            
            samples.append(sample)
        
        return samples
    
    def _generate_lm_sample(self) -> Dict[str, Any]:
        length = random.randint(10, self.max_length)
        input_ids = torch.randint(0, self.vocab_size, (length,))
        
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(length, dtype=torch.bool),
            "labels": input_ids.clone()
        }
    
    def _generate_reasoning_sample(self) -> Dict[str, Any]:
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        operation = random.choice(["+", "-", "*"])
        
        if operation == "+":
            result = a + b
        elif operation == "-":
            result = a - b
        else:
            result = a * b
        
        prompt = f"What is {a} {operation} {b}?"
        answer = str(result)
        
        prompt_ids = torch.randint(0, self.vocab_size, (len(prompt.split()),))
        answer_ids = torch.randint(0, self.vocab_size, (len(answer.split()),))
        
        input_ids = torch.cat([prompt_ids, answer_ids])
        labels = torch.cat([torch.full_like(prompt_ids, -100), answer_ids])
        
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(len(input_ids), dtype=torch.bool),
            "labels": labels,
            "prompt": prompt,
            "answer": answer
        }
    
    def _generate_multimodal_sample(self) -> Dict[str, Any]:
        text = f"Describe this image: {random.randint(1, 1000)}"
        text_ids = torch.randint(0, self.vocab_size, (len(text.split()),))
        
        image = torch.randn(3, 224, 224)
        
        return {
            "input_ids": text_ids,
            "attention_mask": torch.ones(len(text_ids), dtype=torch.bool),
            "image": image,
            "text": text
        }
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class ReasoningDataset(Dataset):
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        prompt = sample["prompt"]
        answer = sample["answer"]
        
        prompt_encoded = self.tokenizer(
            prompt,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        answer_encoded = self.tokenizer(
            answer,
            max_length=self.max_length // 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = torch.cat([
            prompt_encoded["input_ids"][0],
            answer_encoded["input_ids"][0]
        ])
        
        attention_mask = torch.cat([
            prompt_encoded["attention_mask"][0],
            answer_encoded["attention_mask"][0]
        ])
        
        labels = torch.cat([
            torch.full_like(prompt_encoded["input_ids"][0], -100),
            answer_encoded["input_ids"][0]
        ])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt": prompt,
            "answer": answer
        }


class MultimodalDataset(Dataset):
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        image_processor: Any,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        text = sample["text"]
        text_encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        image = sample["image"]
        if self.image_processor:
            image = self.image_processor(image)
        
        return {
            "input_ids": text_encoded["input_ids"][0],
            "attention_mask": text_encoded["attention_mask"][0],
            "image": image,
            "text": text
        }


def load_dataset(
    dataset_name: str,
    split: str = "train",
    tokenizer: Optional[Any] = None,
    max_length: int = 512,
    **kwargs
) -> Dataset:
    if dataset_name == "tiny-reasoning-mix":
        return create_reasoning_dataset(
            size=kwargs.get("size", 1000),
            tokenizer=tokenizer,
            max_length=max_length
        )
    elif dataset_name == "multimodal-reasoning-mix":
        return create_multimodal_dataset(
            size=kwargs.get("size", 1000),
            tokenizer=tokenizer,
            max_length=max_length
        )
    else:
        try:
            dataset = hf_load_dataset(dataset_name, split=split, **kwargs)
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load dataset {dataset_name}: {e}")
            return create_synthetic_dataset(
                size=kwargs.get("size", 1000),
                max_length=max_length
            )


def create_synthetic_dataset(
    size: int = 1000,
    max_length: int = 512,
    vocab_size: int = 32000,
    task_type: str = "language_modeling"
) -> SyntheticDataset:
    return SyntheticDataset(
        size=size,
        max_length=max_length,
        vocab_size=vocab_size,
        task_type=task_type
    )


def create_reasoning_dataset(
    size: int = 1000,
    tokenizer: Optional[Any] = None,
    max_length: int = 512
) -> ReasoningDataset:
    if tokenizer is None:
        tokenizer = load_tokenizer("gpt2")
    
    data = []
    for i in range(size):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        operation = random.choice(["+", "-", "*"])
        
        if operation == "+":
            result = a + b
        elif operation == "-":
            result = a - b
        else:
            result = a * b
        
        prompt = f"What is {a} {operation} {b}?"
        answer = str(result)
        
        data.append({
            "prompt": prompt,
            "answer": answer,
            "type": "arithmetic"
        })
    
    return ReasoningDataset(data, tokenizer, max_length)


def create_multimodal_dataset(
    size: int = 1000,
    tokenizer: Optional[Any] = None,
    max_length: int = 512
) -> MultimodalDataset:
    if tokenizer is None:
        tokenizer = load_tokenizer("gpt2")
    
    data = []
    for i in range(size):
        text = f"Describe this image: {random.randint(1, 1000)}"
        image = torch.randn(3, 224, 224)
        
        data.append({
            "text": text,
            "image": image,
            "type": "image_description"
        })
    
    return MultimodalDataset(data, tokenizer, None, max_length)


def create_data_loader(
    dataset: Dataset,
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
        drop_last=drop_last
    )


def create_train_eval_loaders(
    dataset: Dataset,
    train_split: float = 0.8,
    batch_size: int = 32,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    train_size = int(len(dataset) * train_split)
    eval_size = len(dataset) - train_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    train_loader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    
    eval_loader = create_data_loader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs
    )
    
    return train_loader, eval_loader


def save_dataset(dataset: Dataset, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        serialized_sample = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                serialized_sample[key] = value.tolist()
            else:
                serialized_sample[key] = value
        samples.append(serialized_sample)
    
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    
    logger.info(f"Saved dataset to {path}")


def load_dataset_from_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    with open(path, "r") as f:
        samples = json.load(f)
    
    # Convert lists back to tensors
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, list) and key in ["input_ids", "attention_mask", "labels"]:
                sample[key] = torch.tensor(value)
    
    return samples
