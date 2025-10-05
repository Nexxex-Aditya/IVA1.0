"""Head implementations"""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger


logger = get_logger(__name__)


class LMHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.tie_embeddings = config.tie_embeddings
        
        if self.tie_embeddings:
            self.weight = None
        else:
            self.weight = nn.Parameter(torch.randn(self.vocab_size, self.d_model))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tie_embeddings:
            if hasattr(self, 'token_embedding'):
                weight = self.token_embedding.weight
            else:
                weight = nn.Parameter(torch.randn(self.vocab_size, self.d_model))
                nn.init.normal_(weight, mean=0.0, std=0.02)
        else:
            weight = self.weight
        
        logits = F.linear(x, weight)
        
        return logits
    
    def set_tied_embeddings(self, token_embedding: nn.Embedding) -> None:
        self.token_embedding = token_embedding
        self.tie_embeddings = True


class ValueHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.value_dim = getattr(config, 'value_dim', 1)
        
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.value_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x)


class RewardHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.reward_dim = getattr(config, 'reward_dim', 1)
        
        self.reward_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.reward_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reward_head(x)


class ClassificationHead(nn.Module):
    
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_labels = num_labels
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, self.num_labels)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiTaskHead(nn.Module):
    
    def __init__(self, config, task_configs: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.task_configs = task_configs
        
        self.heads = nn.ModuleDict()
        for task_name, task_config in task_configs.items():
            task_type = task_config.get("type", "classification")
            if task_type == "classification":
                self.heads[task_name] = ClassificationHead(
                    config, task_config["num_labels"]
                )
            elif task_type == "regression":
                self.heads[task_name] = nn.Linear(
                    self.d_model, task_config.get("output_dim", 1)
                )
            elif task_type == "value":
                self.heads[task_name] = ValueHead(config)
            elif task_type == "reward":
                self.heads[task_name] = RewardHead(config)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        task_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        if task_name is not None:
            return {task_name: self.heads[task_name](x)}
        
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(x)
        
        return outputs


def get_lm_head(config) -> nn.Module:
    return LMHead(config)


def get_value_head(config) -> nn.Module:
    return ValueHead(config)


def get_reward_head(config) -> nn.Module:
    return RewardHead(config)


def get_classification_head(config, num_labels: int) -> nn.Module:
    return ClassificationHead(config, num_labels)


def get_multi_task_head(config, task_configs: Dict[str, Dict[str, Any]]) -> nn.Module:
    return MultiTaskHead(config, task_configs)
