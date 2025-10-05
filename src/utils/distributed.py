
from __future__ import annotations

import os
import torch
import torch.distributed as dist
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training")
    
    if init_method is None:
        init_method = os.getenv("MASTER_ADDR", "localhost")
        init_method = f"tcp://{init_method}:{os.getenv('MASTER_PORT', '12355')}"
    
    if world_size is None:
        world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if rank is None:
        rank = int(os.getenv("RANK", "0"))
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def is_distributed() -> bool:
    return dist.is_initialized()


def get_expert_group(
    expert_id: int,
    n_experts: int,
    world_size: Optional[int] = None
) -> Optional[torch.distributed.ProcessGroup]:
    if not is_distributed():
        return None
    
    if world_size is None:
        world_size = get_world_size()
    
    expert_group_size = world_size // n_experts
    if expert_group_size < 1:
        raise ValueError(f"Not enough processes for {n_experts} experts")
    
    start_rank = expert_id * expert_group_size
    end_rank = min(start_rank + expert_group_size, world_size)
    expert_ranks = list(range(start_rank, end_rank))
    
    return dist.new_group(expert_ranks)


def get_data_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    if not is_distributed():
        return None
    
    return dist.group.WORLD


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    if not is_distributed():
        return tensor
    
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported reduction operation: {op}")
    
    dist.all_reduce(tensor, op=op_map[op])
    
    if op == "mean":
        tensor = tensor / get_world_size()
    
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor.unsqueeze(0)
    
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered)


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if not is_distributed():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def reduce_scatter(
    input_list: List[torch.Tensor],
    op: str = "sum"
) -> torch.Tensor:
    if not is_distributed():
        return input_list[0]
    
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported reduction operation: {op}")
    
    output = torch.zeros_like(input_list[0])
    dist.reduce_scatter(output, input_list, op=op_map[op])
    
    if op == "mean":
        output = output / get_world_size()
    
    return output


@contextmanager
def distributed_context(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
):
    try:
        init_distributed(backend, init_method, world_size, rank)
        yield
    finally:
        cleanup_distributed()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_rank()}")
    return torch.device("cpu")


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def get_node_rank() -> int:
    return int(os.getenv("NODE_RANK", "0"))


def get_num_nodes() -> int:
    return int(os.getenv("NNODES", "1"))


def get_processes_per_node() -> int:
    return int(os.getenv("NPROC_PER_NODE", "1"))


def setup_distributed_environment() -> Dict[str, Any]:
    env_info = {
        "RANK": get_rank(),
        "WORLD_SIZE": get_world_size(),
        "LOCAL_RANK": get_local_rank(),
        "NODE_RANK": get_node_rank(),
        "NNODES": get_num_nodes(),
        "NPROC_PER_NODE": get_processes_per_node(),
        "MASTER_ADDR": os.getenv("MASTER_ADDR", "localhost"),
        "MASTER_PORT": os.getenv("MASTER_PORT", "12355"),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
    }
    
    return env_info


def check_distributed_setup() -> bool:
    try:
        if not is_distributed():
            return True
        
        rank = get_rank()
        world_size = get_world_size()
        
        if rank < 0 or rank >= world_size:
            return False
        
        if world_size < 1:
            return False
        
        if not torch.cuda.is_available():
            return False
        
        device = get_device()
        if device.type != "cuda":
            return False
        
        return True
        
    except Exception:
        return False


def synchronize() -> None:
    if is_distributed():
        dist.barrier()


def get_backend() -> str:
    if not is_distributed():
        return "none"
    
    return dist.get_backend()
