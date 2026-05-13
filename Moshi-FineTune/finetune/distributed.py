import logging
import os
from functools import lru_cache
from typing import List, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("distributed")

BACKEND = "nccl"


@lru_cache()
def get_rank() -> int:
    if dist.is_initialized():  # Ensure the distributed process group is initialized
        return dist.get_rank()
    return 0  # Default rank for non-distributed setup


@lru_cache()
def get_world_size() -> int:
    if dist.is_initialized():  # Ensure the distributed process group is initialized
        return dist.get_world_size()
    return 1  # Default world size for non-distributed setup


def visible_devices() -> List[int]:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    return [0]  # Default to device 0 if no environment variable


def set_device():
    logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        logger.info("CUDA_VISIBLE_DEVICES not set, defaulting to device 0")

    assert torch.cuda.is_available()

    if torch.cuda.device_count() == 1:
        # Single GPU setup
        logger.info("Using single GPU.")
        torch.cuda.set_device(0)
        return

    # Multi-GPU setup
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        logger.info(f"Set cuda device to {local_rank}")
        assert 0 <= local_rank < torch.cuda.device_count(), (
            local_rank,
            torch.cuda.device_count(),
        )
        torch.cuda.set_device(local_rank)
    else:
        logger.warning("LOCAL_RANK not found. Defaulting to device 0.")
        torch.cuda.set_device(0)


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    if dist.is_initialized():
        buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        return buffer[0].item() / get_world_size()
    return metric  # For non-distributed, just return the metric as is


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ