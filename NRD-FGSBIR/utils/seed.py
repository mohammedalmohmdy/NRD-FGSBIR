"""
Seed and reproducibility utilities for Neural Retrieval Dynamics (NRD)
This module provides functions for setting random seeds and ensuring reproducibility.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    This sets the seed for random, numpy, and torch to ensure
    reproducible results across runs.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_reproducibility(config: dict) -> None:
    """
    Set reproducibility settings from configuration.
    
    Args:
        config: Configuration dictionary containing reproducibility settings
    """
    repro_config = config.get('reproducibility', {})
    
    # Set seed
    seed = repro_config.get('seed', 42)
    set_seed(seed)
    
    # Set deterministic flag
    deterministic = repro_config.get('deterministic', True)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f"Reproducibility settings:")
    print(f"  Seed: {seed}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Benchmark: {torch.backends.cudnn.benchmark}")


def get_worker_init_fn(seed: int = 42):
    """
    Get worker initialization function for DataLoader.
    
    This ensures that each worker in a DataLoader has a different
    random seed for reproducibility.
    
    Args:
        seed: Base seed value
        
    Returns:
        Function: Worker initialization function
    """
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn
