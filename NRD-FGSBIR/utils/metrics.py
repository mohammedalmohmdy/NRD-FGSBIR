"""
Metrics utilities for Neural Retrieval Dynamics (NRD)
This module provides metric computation functions.
"""

import numpy as np
from typing import Dict, List, Tuple
import torch


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics.
    
    Args:
        predictions: Predicted similarity scores or rankings
        targets: Ground truth labels
        k_values: List of K values for precision/recall
        
    Returns:
        Dict containing computed metrics
    """
    metrics = {}
    
    # Compute precision@K
    for k in k_values:
        metrics[f'precision@{k}'] = compute_precision_at_k(predictions, targets, k)
    
    # Compute recall@K
    for k in k_values:
        metrics[f'recall@{k}'] = compute_recall_at_k(predictions, targets, k)
    
    # Compute top-K accuracy
    metrics['top1'] = compute_top_k_accuracy(predictions, targets, 1)
    metrics['top5'] = compute_top_k_accuracy(predictions, targets, 5)
    metrics['top10'] = compute_top_k_accuracy(predictions, targets, 10)
    
    return metrics


def compute_precision_at_k(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int
) -> float:
    """
    Compute Precision at K.
    
    Args:
        predictions: Predicted rankings or similarities
        targets: Ground truth labels
        k: K value
        
    Returns:
        float: Precision@K score
    """
    num_queries = predictions.shape[0]
    precisions = []
    
    for i in range(num_queries):
        # Get top-k predictions
        if len(predictions.shape) == 2:
            # Similarity matrix
            top_k_indices = np.argsort(predictions[i])[::-1][:k]
        else:
            # Already rankings
            top_k_indices = predictions[i][:k]
        
        # Get labels of top-k
        retrieved_labels = targets[top_k_indices]
        
        # Compute precision
        query_label = targets[i]
        relevant = (retrieved_labels == query_label).sum()
        precision = relevant / k
        
        precisions.append(precision)
    
    return np.mean(precisions)


def compute_recall_at_k(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int
) -> float:
    """
    Compute Recall at K.
    
    Args:
        predictions: Predicted rankings or similarities
        targets: Ground truth labels
        k: K value
        
    Returns:
        float: Recall@K score
    """
    num_queries = predictions.shape[0]
    recalls = []
    
    for i in range(num_queries):
        # Get top-k predictions
        if len(predictions.shape) == 2:
            top_k_indices = np.argsort(predictions[i])[::-1][:k]
        else:
            top_k_indices = predictions[i][:k]
        
        # Get labels of top-k
        retrieved_labels = targets[top_k_indices]
        
        # Compute recall
        query_label = targets[i]
        relevant_retrieved = (retrieved_labels == query_label).sum()
        relevant_total = (targets == query_label).sum()
        
        if relevant_total > 0:
            recall = relevant_retrieved / relevant_total
        else:
            recall = 0.0
        
        recalls.append(recall)
    
    return np.mean(recalls)


def compute_top_k_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int
) -> float:
    """
    Compute Top-K accuracy.
    
    Args:
        predictions: Predicted rankings or similarities
        targets: Ground truth labels
        k: K value
        
    Returns:
        float: Top-K accuracy
    """
    num_queries = predictions.shape[0]
    correct = 0
    
    for i in range(num_queries):
        # Get top-k predictions
        if len(predictions.shape) == 2:
            top_k_indices = np.argsort(predictions[i])[::-1][:k]
        else:
            top_k_indices = predictions[i][:k]
        
        # Get labels of top-k
        retrieved_labels = targets[top_k_indices]
        
        # Check if correct label is in top-k
        query_label = targets[i]
        if query_label in retrieved_labels:
            correct += 1
    
    return correct / num_queries


def compute_map_at_k(
    predictions: np.ndarray,
    targets: np.ndarray,
    k: int = 200
) -> float:
    """
    Compute Mean Average Precision at K.
    
    Args:
        predictions: Predicted similarities or rankings
        targets: Ground truth labels
        k: K value for mAP@K
        
    Returns:
        float: mAP@K score
    """
    num_queries = predictions.shape[0]
    average_precisions = []
    
    for i in range(num_queries):
        # Get similarities for query i
        if len(predictions.shape) == 2:
            similarities = predictions[i]
        else:
            similarities = predictions
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Get labels of top-k results
        retrieved_labels = targets[top_k_indices]
        
        # Compute precision at each position
        query_label = targets[i]
        relevant = (retrieved_labels == query_label).astype(int)
        
        if relevant.sum() == 0:
            average_precisions.append(0.0)
            continue
        
        precisions = []
        num_relevant = 0
        for j in range(k):
            if relevant[j] == 1:
                num_relevant += 1
                precision = num_relevant / (j + 1)
                precisions.append(precision)
        
        if len(precisions) > 0:
            average_precision = np.mean(precisions)
        else:
            average_precision = 0.0
        
        average_precisions.append(average_precision)
    
    return np.mean(average_precisions)


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
