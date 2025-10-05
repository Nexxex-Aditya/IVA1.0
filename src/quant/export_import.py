
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from safetensors import safe_open, save_file

from .fp8_runtime import FP8Autocast, get_fp8_amax, get_fp8_scaling_factor
from .int4_weightonly import Int4WeightOnlyQuantizer
from ..utils.logging import get_logger


logger = get_logger(__name__)


def save_quantized_model(
    model: nn.Module,
    quantizer: Optional[Int4WeightOnlyQuantizer] = None,
    fp8_autocast: Optional[FP8Autocast] = None,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    model_path = path / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    
    quant_info = {}
    
    if quantizer is not None:
        quant_info["int4"] = quantizer.get_quantization_info()
    
    if fp8_autocast is not None:
        quant_info["fp8"] = {
            "amax": fp8_autocast.amax_tracker.get_amax(),
            "scaling_factor": fp8_autocast.amax_tracker.get_scaling_factor(),
            "amax_drift": fp8_autocast.get_amax_drift()
        }
    
    quant_info_path = path / "quantization_info.json"
    with open(quant_info_path, "w") as f:
        json.dump(quant_info, f, indent=2)
    
    if metadata is not None:
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved quantized model to {path}")


def load_quantized_model(
    model: nn.Module,
    path: Union[str, Path],
    quantizer: Optional[Int4WeightOnlyQuantizer] = None,
    fp8_autocast: Optional[FP8Autocast] = None
) -> Tuple[nn.Module, Optional[Int4WeightOnlyQuantizer], Optional[FP8Autocast]]:
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    
    model_path = path / "pytorch_model.bin"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    quant_info_path = path / "quantization_info.json"
    if quant_info_path.exists():
        with open(quant_info_path, "r") as f:
            quant_info = json.load(f)
        
        if "int4" in quant_info and quantizer is not None:
            int4_info = quant_info["int4"]
            quantizer.scheme = int4_info.get("scheme", "awq")
            quantizer.group_size = int4_info.get("group_size", 128)
            quantizer.zero_point = int4_info.get("zero_point", True)
            quantizer.scale = int4_info.get("scale", True)
            quantizer.quantize_weights = int4_info.get("quantize_weights", True)
            quantizer.quantize_biases = int4_info.get("quantize_biases", False)
        
        if "fp8" in quant_info and fp8_autocast is not None:
            fp8_info = quant_info["fp8"]
            fp8_autocast.amax_tracker.current_amax = fp8_info.get("amax", 0.0)
            fp8_autocast.amax_tracker.current_scaling_factor = fp8_info.get("scaling_factor", 1.0)
    
    logger.info(f"Loaded quantized model from {path}")
    
    return model, quantizer, fp8_autocast


def export_to_safetensors(
    model: nn.Module,
    quantizer: Optional[Int4WeightOnlyQuantizer] = None,
    fp8_autocast: Optional[FP8Autocast] = None,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    state_dict = model.state_dict()
    
    if quantizer is not None:
        for name, scale in quantizer.scales.items():
            state_dict[f"{name}_quant_scale"] = scale
        
        if quantizer.zero_point:
            for name, zero_point in quantizer.zero_points.items():
                state_dict[f"{name}_quant_zero_point"] = zero_point
    
    if fp8_autocast is not None:
        state_dict["fp8_amax"] = torch.tensor(fp8_autocast.amax_tracker.get_amax())
        state_dict["fp8_scaling_factor"] = torch.tensor(fp8_autocast.amax_tracker.get_scaling_factor())
    
    safetensors_path = path / "model.safetensors"
    save_file(state_dict, safetensors_path)
    
    quant_info = {}
    
    if quantizer is not None:
        quant_info["int4"] = quantizer.get_quantization_info()
    
    if fp8_autocast is not None:
        quant_info["fp8"] = {
            "amax": fp8_autocast.amax_tracker.get_amax(),
            "scaling_factor": fp8_autocast.amax_tracker.get_scaling_factor(),
            "amax_drift": fp8_autocast.get_amax_drift()
        }
    
    quant_info_path = path / "quantization_info.json"
    with open(quant_info_path, "w") as f:
        json.dump(quant_info, f, indent=2)
    
    if metadata is not None:
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    logger.info(f"Exported quantized model to SafeTensors format at {path}")


def import_from_safetensors(
    model: nn.Module,
    path: Union[str, Path],
    quantizer: Optional[Int4WeightOnlyQuantizer] = None,
    fp8_autocast: Optional[FP8Autocast] = None
) -> Tuple[nn.Module, Optional[Int4WeightOnlyQuantizer], Optional[FP8Autocast]]:
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    
    safetensors_path = path / "model.safetensors"
    if safetensors_path.exists():
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            state_dict = {}
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        raise FileNotFoundError(f"SafeTensors file not found: {safetensors_path}")
    
    quant_params = {}
    model_params = {}
    
    for key, value in state_dict.items():
        if key.endswith("_quant_scale") or key.endswith("_quant_zero_point"):
            quant_params[key] = value
        elif key.startswith("fp8_"):
            quant_params[key] = value
        else:
            model_params[key] = value
    
    model.load_state_dict(model_params)
    
    if quantizer is not None:
        for key, value in quant_params.items():
            if key.endswith("_quant_scale"):
                name = key[:-len("_quant_scale")]
                quantizer.scales[name] = value
            elif key.endswith("_quant_zero_point"):
                name = key[:-len("_quant_zero_point")]
                quantizer.zero_points[name] = value
    
    if fp8_autocast is not None:
        if "fp8_amax" in quant_params:
            fp8_autocast.amax_tracker.current_amax = quant_params["fp8_amax"].item()
        if "fp8_scaling_factor" in quant_params:
            fp8_autocast.amax_tracker.current_scaling_factor = quant_params["fp8_scaling_factor"].item()
    
    quant_info_path = path / "quantization_info.json"
    if quant_info_path.exists():
        with open(quant_info_path, "r") as f:
            quant_info = json.load(f)
        
        if "int4" in quant_info and quantizer is not None:
            int4_info = quant_info["int4"]
            quantizer.scheme = int4_info.get("scheme", "awq")
            quantizer.group_size = int4_info.get("group_size", 128)
            quantizer.zero_point = int4_info.get("zero_point", True)
            quantizer.scale = int4_info.get("scale", True)
            quantizer.quantize_weights = int4_info.get("quantize_weights", True)
            quantizer.quantize_biases = int4_info.get("quantize_biases", False)
        
        if "fp8" in quant_info and fp8_autocast is not None:
            fp8_info = quant_info["fp8"]
            fp8_autocast.amax_tracker.current_amax = fp8_info.get("amax", 0.0)
            fp8_autocast.amax_tracker.current_scaling_factor = fp8_info.get("scaling_factor", 1.0)
    
    logger.info(f"Imported quantized model from SafeTensors format at {path}")
    
    return model, quantizer, fp8_autocast


def get_quantization_metadata(
    model: nn.Module,
    quantizer: Optional[Int4WeightOnlyQuantizer] = None,
    fp8_autocast: Optional[FP8Autocast] = None
) -> Dict[str, Any]:
    metadata = {
        "model_type": type(model).__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_quantized_parameters": 0,
        "quantization_schemes": []
    }
    
    if quantizer is not None:
        quant_info = quantizer.get_quantization_info()
        metadata["int4_quantization"] = quant_info
        metadata["quantization_schemes"].append("int4")
        metadata["num_quantized_parameters"] += len(quantizer.quantized_weights)
    
    if fp8_autocast is not None:
        metadata["fp8_quantization"] = {
            "amax": fp8_autocast.amax_tracker.get_amax(),
            "scaling_factor": fp8_autocast.amax_tracker.get_scaling_factor(),
            "amax_drift": fp8_autocast.get_amax_drift()
        }
        metadata["quantization_schemes"].append("fp8")
    
    return metadata


def validate_quantized_model(
    model: nn.Module,
    quantizer: Optional[Int4WeightOnlyQuantizer] = None,
    fp8_autocast: Optional[FP8Autocast] = None
) -> bool:
    try:
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                logger.warning(f"Non-finite parameter found: {name}")
                return False
        
        if quantizer is not None:
            for name, scale in quantizer.scales.items():
                if not torch.isfinite(scale).all():
                    logger.warning(f"Non-finite scale found: {name}")
                    return False
            
            if quantizer.zero_point:
                for name, zero_point in quantizer.zero_points.items():
                    if not torch.isfinite(zero_point).all():
                        logger.warning(f"Non-finite zero point found: {name}")
                        return False
        
        if fp8_autocast is not None:
            amax = fp8_autocast.amax_tracker.get_amax()
            scaling_factor = fp8_autocast.amax_tracker.get_scaling_factor()
            
            if not torch.isfinite(torch.tensor(amax)):
                logger.warning("Non-finite FP8 amax")
                return False
            
            if not torch.isfinite(torch.tensor(scaling_factor)):
                logger.warning("Non-finite FP8 scaling factor")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating quantized model: {e}")
        return False
