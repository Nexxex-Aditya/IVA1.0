"""FastAPI serving"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Iterator
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core.transformer import Transformer
from ..utils.logging import get_logger
from ..utils.config import load_config
from .generate import Generator, generate, generate_batch, generate_stream


logger = get_logger(__name__)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_length: Optional[int] = Field(2048, description="Maximum generation length")
    temperature: Optional[float] = Field(0.8, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling")
    top_k: Optional[int] = Field(50, description="Top-k sampling")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")
    do_sample: Optional[bool] = Field(True, description="Whether to sample")
    num_beams: Optional[int] = Field(1, description="Number of beams")
    early_stopping: Optional[bool] = Field(False, description="Whether to stop early")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream response")


class GenerateResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text")
    generation_time: float = Field(..., description="Generation time in seconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    tokens_per_second: float = Field(..., description="Tokens per second")


class GenerateBatchRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of input prompts")
    max_length: Optional[int] = Field(2048, description="Maximum generation length")
    temperature: Optional[float] = Field(0.8, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling")
    top_k: Optional[int] = Field(50, description="Top-k sampling")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")
    do_sample: Optional[bool] = Field(True, description="Whether to sample")
    num_beams: Optional[int] = Field(1, description="Number of beams")
    early_stopping: Optional[bool] = Field(False, description="Whether to stop early")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")


class GenerateBatchResponse(BaseModel):
    generated_texts: List[str] = Field(..., description="List of generated texts")
    generation_time: float = Field(..., description="Generation time in seconds")
    total_tokens: int = Field(..., description="Total tokens generated")
    tokens_per_second: float = Field(..., description="Tokens per second")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage information")


class StatsResponse(BaseModel):
    generation_count: int = Field(..., description="Total number of generations")
    total_tokens: int = Field(..., description="Total tokens generated")
    total_time: float = Field(..., description="Total generation time")
    avg_tokens_per_sec: float = Field(..., description="Average tokens per second")
    avg_time_per_generation: float = Field(..., description="Average time per generation")


app = FastAPI(
    title="iva API",
    description="Text generation API for iva",
    version="1.0.0"
)

generator: Optional[Generator] = None
model: Optional[Transformer] = None
tokenizer: Optional[Any] = None
device: Optional[torch.device] = None
config: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    global generator, model, tokenizer, device, config
    
    try:
        config = load_config("configs/quant_infer.yaml")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from transformers import AutoTokenizer
        
        model_path = config.get("model_path", "Qwen/Qwen2.5-0.5B")
        model = Transformer.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        generator = Generator(model, tokenizer, device, config)
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    global generator, model, tokenizer
    
    generator = None
    model = None
    tokenizer = None
    
    logger.info("Model unloaded successfully")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    global generator, model, device
    
    memory_usage = {}
    if torch.cuda.is_available():
        memory_usage = {
            "cuda_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "cuda_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            "cuda_max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    
    return HealthResponse(
        status="healthy" if generator is not None else "unhealthy",
        model_loaded=generator is not None,
        device=str(device) if device else "unknown",
        memory_usage=memory_usage
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    global generator
    
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    stats = generator.get_stats()
    return StatsResponse(**stats)


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    global generator
    
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        if request.stream:
            generated_text = ""
            for chunk in generator.generate_stream(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                num_beams=request.num_beams,
                early_stopping=request.early_stopping,
                stop_sequences=request.stop_sequences
            ):
                generated_text += chunk
        else:
            generated_text = generator.generate(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                num_beams=request.num_beams,
                early_stopping=request.early_stopping,
                stop_sequences=request.stop_sequences
            )
        
        generation_time = time.time() - start_time
        tokens_generated = len(generated_text.split())
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0.0
        
        return GenerateResponse(
            generated_text=generated_text,
            generation_time=generation_time,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_batch", response_model=GenerateBatchResponse)
async def generate_text_batch(request: GenerateBatchRequest):
    global generator
    
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        generated_texts = generator.generate_batch(
            prompts=request.prompts,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            num_beams=request.num_beams,
            early_stopping=request.early_stopping,
            stop_sequences=request.stop_sequences
        )
        
        generation_time = time.time() - start_time
        total_tokens = sum(len(text.split()) for text in generated_texts)
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0.0
        
        return GenerateBatchResponse(
            generated_texts=generated_texts,
            generation_time=generation_time,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second
        )
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_stream")
async def generate_text_stream(request: GenerateRequest):
    global generator
    
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream must be enabled for streaming endpoint")
    
    def generate_streaming():
        try:
            for chunk in generator.generate_stream(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                num_beams=request.num_beams,
                early_stopping=request.early_stopping,
                stop_sequences=request.stop_sequences
            ):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"data: ERROR: {str(e)}\n\n"
    
    return StreamingResponse(
        generate_streaming(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/reset_stats")
async def reset_stats():
    global generator
    
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    generator.reset_stats()
    return {"message": "Statistics reset successfully"}


def create_app(
    model_path: str = "Qwen/Qwen2.5-0.5B",
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> FastAPI:
    global generator, model, tokenizer, device, config
    
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config("configs/quant_infer.yaml")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from transformers import AutoTokenizer
    
    model = Transformer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    generator = Generator(model, tokenizer, device, config)
    
    return app


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Start iva API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model path or name")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    app = create_app(
        model_path=args.model,
        config_path=args.config,
        device=device
    )
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
