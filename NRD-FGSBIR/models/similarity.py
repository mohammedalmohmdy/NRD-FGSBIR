"""
Progressive Similarity Optimization for Neural Retrieval Dynamics (NRD)
This module implements the progressive similarity computation for retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class SimilarityAccumulator(nn.Module):
    """
    Similarity Accumulator for progressive similarity computation.
    
    This module accumulates similarity scores across trajectory steps
    to provide a comprehensive similarity measure between modalities.
    
    The accumulation follows:
        S_total = Σ_{t=0}^{T} w_t * sim(z_s(t), z_i(t))
        
    where w_t are learnable weights and sim is the similarity function.
    
    Args:
        num_steps (int): Number of trajectory steps
        similarity_type (str): Type of similarity ('cosine', 'euclidean', 'dot')
    """
    
    def __init__(
        self,
        num_steps: int = 10,
        similarity_type: str = 'cosine'
    ):
        """
        Initialize the similarity accumulator.
        
        Args:
            num_steps: Number of trajectory steps
            similarity_type: Type of similarity measure
        """
        super(SimilarityAccumulator, self).__init__()
        
        self.num_steps = num_steps
        self.similarity_type = similarity_type
        
        # Learnable weights for each step
        self.step_weights = nn.Parameter(torch.ones(num_steps) / num_steps)
        
    def forward(
        self,
        sketch_trajectory: torch.Tensor,
        image_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Accumulate similarity across trajectory steps.
        
        Args:
            sketch_trajectory: Sketch trajectory of shape (B, num_steps, feature_dim)
            image_trajectory: Image trajectory of shape (B, num_steps, feature_dim)
            
        Returns:
            torch.Tensor: Accumulated similarity scores of shape (B,)
        """
        batch_size = sketch_trajectory.size(0)
        device = sketch_trajectory.device
        
        # Normalize weights
        normalized_weights = F.softmax(self.step_weights, dim=0)
        
        # Compute similarity at each step
        similarities = []
        for t in range(self.num_steps):
            sketch_t = sketch_trajectory[:, t, :]  # (B, feature_dim)
            image_t = image_trajectory[:, t, :]    # (B, feature_dim)
            
            if self.similarity_type == 'cosine':
                sim = F.cosine_similarity(sketch_t, image_t, dim=1)
            elif self.similarity_type == 'euclidean':
                sim = -F.pairwise_distance(sketch_t, image_t, p=2)
            elif self.similarity_type == 'dot':
                sim = torch.sum(sketch_t * image_t, dim=1)
            else:
                raise ValueError(f"Unknown similarity type: {self.similarity_type}")
            
            similarities.append(sim)
        
        # Stack similarities
        similarities = torch.stack(similarities, dim=1)  # (B, num_steps)
        
        # Weighted sum
        accumulated_similarity = torch.sum(similarities * normalized_weights.unsqueeze(0), dim=1)
        
        return accumulated_similarity
    
    def get_step_similarities(
        self,
        sketch_trajectory: torch.Tensor,
        image_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Get similarity scores at each trajectory step.
        
        Args:
            sketch_trajectory: Sketch trajectory of shape (B, num_steps, feature_dim)
            image_trajectory: Image trajectory of shape (B, num_steps, feature_dim)
            
        Returns:
            torch.Tensor: Similarity scores at each step of shape (B, num_steps)
        """
        batch_size = sketch_trajectory.size(0)
        
        # Compute similarity at each step
        similarities = []
        for t in range(self.num_steps):
            sketch_t = sketch_trajectory[:, t, :]
            image_t = image_trajectory[:, t, :]
            
            if self.similarity_type == 'cosine':
                sim = F.cosine_similarity(sketch_t, image_t, dim=1)
            elif self.similarity_type == 'euclidean':
                sim = -F.pairwise_distance(sketch_t, image_t, p=2)
            elif self.similarity_type == 'dot':
                sim = torch.sum(sketch_t * image_t, dim=1)
            else:
                raise ValueError(f"Unknown similarity type: {self.similarity_type}")
            
            similarities.append(sim)
        
        return torch.stack(similarities, dim=1)


class ProgressiveSimilarity(nn.Module):
    """
    Progressive Similarity Optimization module.
    
    This module implements progressive similarity optimization by computing
    similarity at multiple stages of the trajectory evolution and combining
    them in a learned manner. This allows the model to focus on different
    aspects of similarity at different stages of the cross-modal evolution.
    
    The progressive similarity follows:
        S_progressive = Σ_{k=1}^{K} α_k * S_k
        
    where S_k is the similarity at stage k and α_k are learnable weights.
    
    Args:
        feature_dim (int): Dimension of the feature space
        num_steps (int): Number of trajectory steps
        progressive_steps (int): Number of progressive stages
        similarity_type (str): Type of similarity ('cosine', 'euclidean', 'dot')
        alpha (float): Weight for progressive combination
        beta (float): Weight for trajectory alignment
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        num_steps: int = 10,
        progressive_steps: int = 5,
        similarity_type: str = 'cosine',
        alpha: float = 0.5,
        beta: float = 0.3
    ):
        """
        Initialize the progressive similarity module.
        
        Args:
            feature_dim: Feature dimension
            num_steps: Number of trajectory steps
            progressive_steps: Number of progressive stages
            similarity_type: Type of similarity measure
            alpha: Weight for progressive combination
            beta: Weight for trajectory alignment
        """
        super(ProgressiveSimilarity, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        self.progressive_steps = progressive_steps
        self.similarity_type = similarity_type
        self.alpha = alpha
        self.beta = beta
        
        # Similarity accumulator
        self.similarity_accumulator = SimilarityAccumulator(
            num_steps=num_steps,
            similarity_type=similarity_type
        )
        
        # Progressive stage weights
        self.stage_weights = nn.Parameter(torch.ones(progressive_steps) / progressive_steps)
        
        # Stage-specific similarity networks
        self.stage_networks = nn.ModuleList()
        for _ in range(progressive_steps):
            stage_net = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, 1)
            )
            self.stage_networks.append(stage_net)
        
    def forward(
        self,
        sketch_trajectory: torch.Tensor,
        image_trajectory: torch.Tensor,
        sketch_features: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute progressive similarity.
        
        Args:
            sketch_trajectory: Sketch trajectory of shape (B, num_steps, feature_dim)
            image_trajectory: Image trajectory of shape (B, num_steps, feature_dim)
            sketch_features: Original sketch features of shape (B, feature_dim)
            image_features: Original image features of shape (B, feature_dim)
            
        Returns:
            Tuple containing:
                - progressive_similarity: Progressive similarity scores (B,)
                - trajectory_similarity: Trajectory-based similarity scores (B,)
        """
        # Compute trajectory-based similarity
        trajectory_similarity = self.similarity_accumulator(sketch_trajectory, image_trajectory)
        
        # Compute progressive stage similarities
        stage_similarities = []
        step_indices = torch.linspace(0, self.num_steps - 1, self.progressive_steps, dtype=torch.long)
        
        for k, step_idx in enumerate(step_indices):
            # Get features at this step
            sketch_t = sketch_trajectory[:, step_idx, :]
            image_t = image_trajectory[:, step_idx, :]
            
            # Concatenate features
            combined = torch.cat([sketch_t, image_t], dim=1)
            
            # Compute stage-specific similarity
            stage_sim = self.stage_networks[k](combined).squeeze(1)
            stage_similarities.append(stage_sim)
        
        # Stack stage similarities
        stage_similarities = torch.stack(stage_similarities, dim=1)  # (B, progressive_steps)
        
        # Normalize stage weights
        normalized_weights = F.softmax(self.stage_weights, dim=0)
        
        # Weighted combination
        progressive_similarity = torch.sum(stage_similarities * normalized_weights.unsqueeze(0), dim=1)
        
        # Combine trajectory and progressive similarities
        combined_similarity = self.alpha * progressive_similarity + self.beta * trajectory_similarity
        
        return combined_similarity, trajectory_similarity
    
    def compute_pairwise_similarity(
        self,
        sketch_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise similarity between sketch and image features.
        
        Args:
            sketch_features: Sketch features of shape (B, feature_dim)
            image_features: Image features of shape (B, feature_dim)
            
        Returns:
            torch.Tensor: Pairwise similarity scores of shape (B,)
        """
        if self.similarity_type == 'cosine':
            similarity = F.cosine_similarity(sketch_features, image_features, dim=1)
        elif self.similarity_type == 'euclidean':
            similarity = -F.pairwise_distance(sketch_features, image_features, p=2)
        elif self.similarity_type == 'dot':
            similarity = torch.sum(sketch_features * image_features, dim=1)
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        
        return similarity
    
    def compute_similarity_matrix(
        self,
        query_features: torch.Tensor,
        gallery_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity matrix between queries and gallery.
        
        Args:
            query_features: Query features of shape (N, feature_dim)
            gallery_features: Gallery features of shape (M, feature_dim)
            
        Returns:
            torch.Tensor: Similarity matrix of shape (N, M)
        """
        if self.similarity_type == 'cosine':
            # Normalize features
            query_norm = F.normalize(query_features, p=2, dim=1)
            gallery_norm = F.normalize(gallery_features, p=2, dim=1)
            
            # Compute cosine similarity
            similarity_matrix = torch.mm(query_norm, gallery_norm.t())
            
        elif self.similarity_type == 'euclidean':
            # Compute negative Euclidean distance
            similarity_matrix = -torch.cdist(query_features, gallery_features, p=2)
            
        elif self.similarity_type == 'dot':
            # Compute dot product
            similarity_matrix = torch.mm(query_features, gallery_features.t())
            
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
        
        return similarity_matrix
