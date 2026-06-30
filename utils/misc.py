"""misc.py

Miscellaneous utilities: seed management and directory counting.
"""

import os
import random

import numpy as np
import torch


def set_seeds(seed_value: int, device: torch.device) -> None:
    """Set random seeds for reproducibility across all relevant libraries.

    Args:
        seed_value: Integer seed.
        device: Target torch device (used to enable CUDA-specific settings).
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function for reproducible data loading."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_prefix_dirs(directory: str, prefix: str) -> int:
    """Count subdirectories in *directory* whose names start with *prefix*.

    A new experiment index is returned (existing count + 1), so the first
    experiment gets index 1.

    Args:
        directory: Path to the directory to scan.
        prefix:   Prefix string to match at the start of folder names.

    Returns:
        Next available experiment index (>= 1).
    """
    os.makedirs(directory, exist_ok=True)

    count = 1
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if entry.startswith(prefix) and os.path.isdir(entry_path):
            count += 1
    return count