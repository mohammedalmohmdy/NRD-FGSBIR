"""
Validation module for Neural Retrieval Dynamics (NRD)
This module implements validation and evaluation metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from ..models import NRDModel


class Validator:
    """
    Validator class for Neural Retrieval Dynamics.
    
    This class implements validation and evaluation with metrics:
    - mAP@200
    - Precision@K
    - Recall@K
    - Top-1, Top-5, Top-10
    
    Args:
        model (NRDModel): NRD model to validate
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to validate on
        config (Dict): Configuration dictionary
    """
    
    def __init__(
        self,
        model: NRDModel,
        val_loader: DataLoader,
        device: torch.device = torch.device('cuda'),
        config: Optional[Dict] = None
    ):
        """
        Initialize the validator.
        
        Args:
            model: NRD model to validate
            val_loader: Validation data loader
            device: Device to validate on
            config: Configuration dictionary
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        
        # Evaluation settings
        self.k_values = [1, 5, 10]
        self.map_at = 200
        
        # Move model to device
        self.model.to(self.device)
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dict containing validation metrics
        """
        self.model.eval()
        
        # Extract features
        sketch_features, image_features, labels = self._extract_features()
        
        # Compute similarity matrix
        similarity_matrix = self.model.compute_similarity_matrix(
            sketch_features, image_features
        )
        
        # Compute metrics
        metrics = self._compute_metrics(similarity_matrix, labels)
        
        return metrics
    
    def _extract_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from validation set.
        
        Returns:
            Tuple containing:
                - sketch_features: Sketch features (N, feature_dim)
                - image_features: Image features (M, feature_dim)
                - labels: Labels (N,)
        """
        sketch_features_list = []
        image_features_list = []
        labels_list = []
        
        for batch in tqdm(self.val_loader, desc='Extracting features'):
            sketch = batch['sketch'].to(self.device)
            image = batch['image'].to(self.device)
            label = batch['label']
            
            # Encode features
            sketch_feat = self.model.encode_sketch(sketch)
            image_feat = self.model.encode_image(image)
            
            sketch_features_list.append(sketch_feat.cpu())
            image_features_list.append(image_feat.cpu())
            labels_list.append(label)
        
        # Concatenate features
        sketch_features = torch.cat(sketch_features_list, dim=0)
        image_features = torch.cat(image_features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return sketch_features, image_features, labels
    
    def _compute_metrics(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            similarity_matrix: Similarity matrix of shape (N, M)
            labels: Labels of shape (N,)
            
        Returns:
            Dict containing metrics
        """
        # Convert to numpy
        similarity_matrix = similarity_matrix.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Compute metrics
        metrics = {}
        
        # mAP@200
        metrics['map@200'] = self._compute_map_at_k(similarity_matrix, labels, self.map_at)
        
        # Precision@K and Recall@K
        for k in self.k_values:
            metrics[f'precision@{k}'] = self._compute_precision_at_k(similarity_matrix, labels, k)
            metrics[f'recall@{k}'] = self._compute_recall_at_k(similarity_matrix, labels, k)
        
        # Top-K accuracy
        metrics['top1'] = self._compute_top_k_accuracy(similarity_matrix, labels, 1)
        metrics['top5'] = self._compute_top_k_accuracy(similarity_matrix, labels, 5)
        metrics['top10'] = self._compute_top_k_accuracy(similarity_matrix, labels, 10)
        
        return metrics
    
    def _compute_map_at_k(
        self,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        Compute Mean Average Precision at K.
        
        Args:
            similarity_matrix: Similarity matrix of shape (N, M)
            labels: Labels of shape (N,)
            k: K value for mAP@K
            
        Returns:
            float: mAP@K score
        """
        num_queries = similarity_matrix.shape[0]
        average_precisions = []
        
        for i in range(num_queries):
            # Get similarities for query i
            similarities = similarity_matrix[i]
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            # Get labels of top-k results
            retrieved_labels = labels[top_k_indices]
            
            # Compute precision at each position
            query_label = labels[i]
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
        
        map_score = np.mean(average_precisions)
        return map_score
    
    def _compute_precision_at_k(
        self,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        Compute Precision at K.
        
        Args:
            similarity_matrix: Similarity matrix of shape (N, M)
            labels: Labels of shape (N,)
            k: K value
            
        Returns:
            float: Precision@K score
        """
        num_queries = similarity_matrix.shape[0]
        precisions = []
        
        for i in range(num_queries):
            # Get similarities for query i
            similarities = similarity_matrix[i]
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            # Get labels of top-k results
            retrieved_labels = labels[top_k_indices]
            
            # Compute precision
            query_label = labels[i]
            relevant = (retrieved_labels == query_label).sum()
            precision = relevant / k
            
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def _compute_recall_at_k(
        self,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        Compute Recall at K.
        
        Args:
            similarity_matrix: Similarity matrix of shape (N, M)
            labels: Labels of shape (N,)
            k: K value
            
        Returns:
            float: Recall@K score
        """
        num_queries = similarity_matrix.shape[0]
        recalls = []
        
        for i in range(num_queries):
            # Get similarities for query i
            similarities = similarity_matrix[i]
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            # Get labels of top-k results
            retrieved_labels = labels[top_k_indices]
            
            # Compute recall
            query_label = labels[i]
            relevant_retrieved = (retrieved_labels == query_label).sum()
            relevant_total = (labels == query_label).sum()
            
            if relevant_total > 0:
                recall = relevant_retrieved / relevant_total
            else:
                recall = 0.0
            
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def _compute_top_k_accuracy(
        self,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """
        Compute Top-K accuracy.
        
        Args:
            similarity_matrix: Similarity matrix of shape (N, M)
            labels: Labels of shape (N,)
            k: K value
            
        Returns:
            float: Top-K accuracy
        """
        num_queries = similarity_matrix.shape[0]
        correct = 0
        
        for i in range(num_queries):
            # Get similarities for query i
            similarities = similarity_matrix[i]
            
            # Get top-k indices
            top_k_indices = np.argsort(similarities)[::-1][:k]
            
            # Get labels of top-k results
            retrieved_labels = labels[top_k_indices]
            
            # Check if correct label is in top-k
            query_label = labels[i]
            if query_label in retrieved_labels:
                correct += 1
        
        top_k_acc = correct / num_queries
        return top_k_acc
    
    def compute_per_class_metrics(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute metrics per class.
        
        Args:
            similarity_matrix: Similarity matrix of shape (N, M)
            labels: Labels of shape (N,)
            
        Returns:
            Dict mapping class labels to metrics
        """
        similarity_matrix = similarity_matrix.cpu().numpy()
        labels = labels.cpu().numpy()
        
        unique_labels = np.unique(labels)
        per_class_metrics = {}
        
        for label in unique_labels:
            # Get indices for this class
            class_indices = np.where(labels == label)[0]
            
            # Get similarities for this class
            class_similarities = similarity_matrix[class_indices]
            class_labels = labels[class_indices]
            
            # Compute metrics for this class
            class_metrics = {
                'map@200': self._compute_map_at_k(class_similarities, class_labels, self.map_at),
                'precision@1': self._compute_precision_at_k(class_similarities, class_labels, 1),
                'precision@5': self._compute_precision_at_k(class_similarities, class_labels, 5),
                'precision@10': self._compute_precision_at_k(class_similarities, class_labels, 10),
                'recall@1': self._compute_recall_at_k(class_similarities, class_labels, 1),
                'recall@5': self._compute_recall_at_k(class_similarities, class_labels, 5),
                'recall@10': self._compute_recall_at_k(class_similarities, class_labels, 10),
            }
            
            per_class_metrics[int(label)] = class_metrics
        
        return per_class_metrics
