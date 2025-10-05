
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger


logger = get_logger(__name__)


class Int4WeightOnlyQuantizer:
    
    def __init__(
        self,
        scheme: str = "awq",
        group_size: int = 128,
        zero_point: bool = True,
        scale: bool = True,
        quantize_weights: bool = True,
        quantize_biases: bool = False
    ):
        self.scheme = scheme
        self.group_size = group_size
        self.zero_point = zero_point
        self.scale = scale
        self.quantize_weights = quantize_weights
        self.quantize_biases = quantize_biases
        
        self.quantized_weights: Dict[str, torch.Tensor] = {}
        self.scales: Dict[str, torch.Tensor] = {}
        self.zero_points: Dict[str, torch.Tensor] = {}
        
        self.original_weights: Dict[str, torch.Tensor] = {}
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        name: str
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.quantize_weights:
            return tensor, torch.ones(1), None
        
        self.original_weights[name] = tensor.clone()
        
        original_shape = tensor.shape
        if len(original_shape) == 2:
            tensor_2d = tensor.view(-1, tensor.size(-1))
        else:
            tensor_2d = tensor.view(-1, tensor.size(-1))
        quantized_tensor, scale, zero_point = self._quantize_group_wise(tensor_2d)
        
        self.quantized_weights[name] = quantized_tensor
        self.scales[name] = scale
        if zero_point is not None:
            self.zero_points[name] = zero_point
        
        return quantized_tensor, scale, zero_point
    
    def _quantize_group_wise(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_groups, group_size = tensor.shape
        scales = torch.zeros(num_groups, device=tensor.device)
        zero_points = torch.zeros(num_groups, device=tensor.device) if self.zero_point else None
        
        quantized_groups = []
        
        for i in range(num_groups):
            group = tensor[i]
            
            if self.scheme == "awq":
                scale, zero_point = self._quantize_awq(group)
            elif self.scheme == "gptq":
                scale, zero_point = self._quantize_gptq(group)
            elif self.scheme == "smoothquant":
                scale, zero_point = self._quantize_smoothquant(group)
            else:
                raise ValueError(f"Unknown quantization scheme: {self.scheme}")
            
            scales[i] = scale
            if zero_points is not None:
                zero_points[i] = zero_point
            
            quantized_group = self._quantize_group(group, scale, zero_point)
            quantized_groups.append(quantized_group)
        
        quantized_tensor = torch.stack(quantized_groups)
        
        return quantized_tensor, scales, zero_points
    
    def _quantize_awq(
        self,
        group: torch.Tensor
    ) -> Tuple[float, Optional[float]]:
        max_val = group.abs().max().item()
        
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 7.0
        
        zero_point = 0.0 if self.zero_point else None
        
        return scale, zero_point
    
    def _quantize_gptq(
        self,
        group: torch.Tensor
    ) -> Tuple[float, Optional[float]]:
        mean_abs = group.abs().mean().item()
        
        if mean_abs == 0:
            scale = 1.0
        else:
            scale = mean_abs / 3.5
        zero_point = 0.0 if self.zero_point else None
        return scale, zero_point
    
    def _quantize_smoothquant(
        self,
        group: torch.Tensor
    ) -> Tuple[float, Optional[float]]:
        max_val = group.abs().max().item()
        
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 7.0
        
        zero_point = 0.0 if self.zero_point else None
        return scale, zero_point
    
    def _quantize_group(
        self,
        group: torch.Tensor,
        scale: float,
        zero_point: Optional[float]
    ) -> torch.Tensor:
        scaled_group = group / scale
        if zero_point is not None:
            scaled_group = scaled_group + zero_point
        quantized_group = torch.clamp(scaled_group, -8, 7)
        quantized_group = torch.round(quantized_group)
        quantized_group = quantized_group.to(torch.int8)
        
        return quantized_group
    
    def dequantize_tensor(
        self,
        quantized_tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        original_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        dequantized_groups = []
        
        for i in range(quantized_tensor.size(0)):
            group = quantized_tensor[i]
            group_scale = scale[i]
            group_zero_point = zero_point[i] if zero_point is not None else 0.0
            dequantized_group = group.float() - group_zero_point
            dequantized_group = dequantized_group * group_scale
            dequantized_groups.append(dequantized_group)
        
        dequantized_tensor = torch.stack(dequantized_groups)
        dequantized_tensor = dequantized_tensor.view(original_shape)
        
        return dequantized_tensor
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_linear(module, name)
            elif isinstance(module, nn.Embedding):
                self._quantize_embedding(module, name)
        
        return model
    
    def _quantize_linear(self, module: nn.Linear, name: str) -> None:
        if self.quantize_weights:
            quantized_weight, scale, zero_point = self.quantize_tensor(
                module.weight, f"{name}.weight"
            )
            module.weight = nn.Parameter(quantized_weight)
            module.register_buffer("_quant_scale", scale)
            if zero_point is not None:
                module.register_buffer("_quant_zero_point", zero_point)
        if self.quantize_biases and module.bias is not None:
            quantized_bias, bias_scale, bias_zero_point = self.quantize_tensor(
                module.bias, f"{name}.bias"
            )
            module.bias = nn.Parameter(quantized_bias)
            module.register_buffer("_bias_quant_scale", bias_scale)
            if bias_zero_point is not None:
                module.register_buffer("_bias_quant_zero_point", bias_zero_point)
    
    def _quantize_embedding(self, module: nn.Embedding, name: str) -> None:
        if self.quantize_weights:
            quantized_weight, scale, zero_point = self.quantize_tensor(
                module.weight, f"{name}.weight"
            )
            module.weight = nn.Parameter(quantized_weight)
            module.register_buffer("_quant_scale", scale)
            if zero_point is not None:
                module.register_buffer("_quant_zero_point", zero_point)
    
    def dequantize_model(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._dequantize_linear(module, name)
            elif isinstance(module, nn.Embedding):
                self._dequantize_embedding(module, name)
        return model
    
    def _dequantize_linear(self, module: nn.Linear, name: str) -> None:
        if hasattr(module, "_quant_scale"):
            scale = module._quant_scale
            zero_point = getattr(module, "_quant_zero_point", None)
            dequantized_weight = self.dequantize_tensor(
                module.weight, scale, zero_point, module.weight.shape
            )
            module.weight = nn.Parameter(dequantized_weight)
            
            delattr(module, "_quant_scale")
            if hasattr(module, "_quant_zero_point"):
                delattr(module, "_quant_zero_point")
        
        if hasattr(module, "_bias_quant_scale"):
            bias_scale = module._bias_quant_scale
            bias_zero_point = getattr(module, "_bias_quant_zero_point", None)
            
            dequantized_bias = self.dequantize_tensor(
                module.bias, bias_scale, bias_zero_point, module.bias.shape
            )
            
            module.bias = nn.Parameter(dequantized_bias)
            
            delattr(module, "_bias_quant_scale")
            if hasattr(module, "_bias_quant_zero_point"):
                delattr(module, "_bias_quant_zero_point")
    
    def _dequantize_embedding(self, module: nn.Embedding, name: str) -> None:
        if hasattr(module, "_quant_scale"):
            scale = module._quant_scale
            zero_point = getattr(module, "_quant_zero_point", None)
            dequantized_weight = self.dequantize_tensor(
                module.weight, scale, zero_point, module.weight.shape
            )
            module.weight = nn.Parameter(dequantized_weight)
            delattr(module, "_quant_scale")
            if hasattr(module, "_quant_zero_point"):
                delattr(module, "_quant_zero_point")
    
    def get_quantization_info(self) -> Dict[str, Any]:
        return {
            "scheme": self.scheme,
            "group_size": self.group_size,
            "zero_point": self.zero_point,
            "scale": self.scale,
            "quantize_weights": self.quantize_weights,
            "quantize_biases": self.quantize_biases,
            "num_quantized_tensors": len(self.quantized_weights),
            "quantized_tensor_names": list(self.quantized_weights.keys())
        }


def quantize_model_int4(
    model: nn.Module,
    scheme: str = "awq",
    group_size: int = 128,
    zero_point: bool = True,
    scale: bool = True,
    quantize_weights: bool = True,
    quantize_biases: bool = False
) -> Tuple[nn.Module, Int4WeightOnlyQuantizer]:
    quantizer = Int4WeightOnlyQuantizer(
        scheme=scheme,
        group_size=group_size,
        zero_point=zero_point,
        scale=scale,
        quantize_weights=quantize_weights,
        quantize_biases=quantize_biases
    )
    
    quantized_model = quantizer.quantize_model(model)
    
    return quantized_model, quantizer


def dequantize_model_int4(
    model: nn.Module,
    quantizer: Int4WeightOnlyQuantizer
) -> nn.Module:
    dequantized_model = quantizer.dequantize_model(model)
    
    return dequantized_model


def get_int4_quantizer(
    scheme: str = "awq",
    group_size: int = 128,
    zero_point: bool = True,
    scale: bool = True,
    quantize_weights: bool = True,
    quantize_biases: bool = False
) -> Int4WeightOnlyQuantizer:
    return Int4WeightOnlyQuantizer(
        scheme=scheme,
        group_size=group_size,
        zero_point=zero_point,
        scale=scale,
        quantize_weights=quantize_weights,
        quantize_biases=quantize_biases
    )
