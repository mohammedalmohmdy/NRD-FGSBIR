"""
Logging utilities for Neural Retrieval Dynamics (NRD)
This module provides logging functionality for training and evaluation.
"""

import logging
import os
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'NRD',
    log_dir: str = './logs',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Specific log file name (auto-generated if None)
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_{timestamp}.log'
    
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    
    Args:
        name: Name of the metric
    """
    
    def __init__(self, name: str = ''):
        """
        Initialize the average meter.
        
        Args:
            name: Name of the metric
        """
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics with new value.
        
        Args:
            val: New value
            n: Number of samples for this value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        """Return string representation."""
        return f'{self.name}: {self.avg:.4f}'
