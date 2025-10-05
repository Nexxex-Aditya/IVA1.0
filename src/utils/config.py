
from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, ValidationError


class Config(BaseModel):
    
    class Config:
        extra = "allow"
        validate_assignment = True
        use_enum_values = True
        frozen = True
    
    def merge(self, other: Union[Dict[str, Any], Config]) -> Config:
        if isinstance(other, Config):
            other_dict = other.dict()
        else:
            other_dict = other
        
        current_dict = self.dict()
        merged_dict = _deep_merge(current_dict, other_dict)
        
        return Config(**merged_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()
    
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    config = Config.from_yaml(config_path)
    
    if overrides:
        config = config.merge(overrides)
    
    return config


def merge_configs(*configs: Union[Config, Dict[str, Any]]) -> Config:
    if not configs:
        return Config()
    
    result = configs[0]
    for config in configs[1:]:
        result = result.merge(config)
    
    return result


def parse_cli_overrides(overrides: list[str]) -> Dict[str, Any]:
    result = {}
    
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}")
        
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        if value.lower() in ('true', 'false'):
            parsed_value = value.lower() == 'true'
        elif value.lower() == 'null':
            parsed_value = None
        elif value.isdigit():
            parsed_value = int(value)
        elif value.replace('.', '').isdigit():
            parsed_value = float(value)
        else:
            parsed_value = value
        
        current = result
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = parsed_value
    
    return result


def get_config_path(config_name: str) -> Path:
    config_dir = Path(__file__).parent.parent.parent / "configs"
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return config_path


def load_config_with_cli(
    config_name: str,
    cli_args: Optional[list[str]] = None
) -> Config:
    config_path = get_config_path(config_name)
    config = Config.from_yaml(config_path)
    
    if cli_args:
        overrides = parse_cli_overrides(cli_args)
        config = config.merge(overrides)
    
    return config


def validate_config(config: Config, schema: Optional[Dict[str, Any]] = None) -> None:
    if schema:
        class SchemaModel(BaseModel):
            __config__ = type('Config', (), {'extra': 'forbid'})
        for field_name, field_info in schema.items():
            setattr(SchemaModel, field_name, Field(**field_info))
        try:
            SchemaModel(**config.dict())
        except ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {e}")


def get_env_config() -> Dict[str, Any]:
    config = {}
    env_mappings = {
        'iva_LOG_LEVEL': 'logging.log_level',
        'iva_SEED': 'training.seed',
        'iva_DEVICE': 'model.device',
        'iva_BATCH_SIZE': 'training.batch_size',
        'iva_LEARNING_RATE': 'training.lr',
        'iva_MAX_LENGTH': 'data.max_length',
        'iva_VOCAB_SIZE': 'model.vocab_size',
        'iva_D_MODEL': 'model.d_model',
        'iva_N_LAYERS': 'model.n_layers',
        'iva_N_HEADS': 'model.n_heads',
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            if value.lower() in ('true', 'false'):
                parsed_value = value.lower() == 'true'
            elif value.isdigit():
                parsed_value = int(value)
            elif value.replace('.', '').isdigit():
                parsed_value = float(value)
            else:
                parsed_value = value
            keys = config_path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = parsed_value
    
    return config
