"""
Dynamic Retrieval Interaction Module for Neural Retrieval Dynamics (NRD)
This module implements the dynamic interaction between modalities for enhanced retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for modality interaction.
    
    This module implements multi-head attention to enable interaction
    between sketch and image modalities. It allows each modality to
    attend to relevant features from the other modality.
    
    The attention mechanism follows:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        
    where Q, K, V are query, key, and value projections.
    
    Args:
        feature_dim (int): Dimension of the feature space
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize the cross-modal attention module.
        
        Args:
            feature_dim: Input feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossModalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, \
            "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for scaled dot-product attention
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            query: Query tensor of shape (B, feature_dim)
            key: Key tensor of shape (B, feature_dim)
            value: Value tensor of shape (B, feature_dim)
            mask: Optional attention mask
            
        Returns:
            torch.Tensor: Attention output of shape (B, feature_dim)
        """
        batch_size = query.size(0)
        
        # Project to Q, K, V
        Q = self.query_proj(query)  # (B, feature_dim)
        K = self.key_proj(key)      # (B, feature_dim)
        V = self.value_proj(value)  # (B, feature_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        # (B, num_heads, head_dim) x (B, num_heads, head_dim) -> (B, num_heads)
        attention_scores = torch.sum(Q * K, dim=2) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (B, num_heads) x (B, num_heads, head_dim) -> (B, num_heads, head_dim)
        attended = attention_weights.unsqueeze(2) * V
        
        # Concatenate heads
        attended = attended.view(batch_size, self.feature_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output


class DynamicInteractionModule(nn.Module):
    """
    Dynamic Retrieval Interaction Module for enhanced cross-modal retrieval.
    
    This module implements dynamic interaction between sketch and image
    modalities through multi-layer cross-modal attention. It enables
    the model to learn complex relationships between modalities and
    adaptively focus on relevant features during retrieval.
    
    The interaction follows:
        h_s^l = Attention(h_s^{l-1}, h_i^{l-1}, h_i^{l-1})
        h_i^l = Attention(h_i^{l-1}, h_s^{l-1}, h_s^{l-1})
        
    where h_s^l and h_i^l are the sketch and image representations at layer l.
    
    Args:
        feature_dim (int): Dimension of the feature space
        hidden_dim (int): Dimension of the hidden layers
        num_heads (int): Number of attention heads
        num_layers (int): Number of interaction layers
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize the dynamic interaction module.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of interaction layers
            dropout: Dropout probability
        """
        super(DynamicInteractionModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = InteractionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.interaction_layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(
        self,
        sketch_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply dynamic interaction between modalities.
        
        Args:
            sketch_features: Sketch features of shape (B, feature_dim)
            image_features: Image features of shape (B, feature_dim)
            
        Returns:
            Tuple containing:
                - interacted_sketch: Interacted sketch features (B, feature_dim)
                - interacted_image: Interacted image features (B, feature_dim)
        """
        # Project to hidden dimension
        sketch_hidden = self.input_proj(sketch_features)
        image_hidden = self.input_proj(image_features)
        
        # Apply interaction layers
        for layer in self.interaction_layers:
            sketch_hidden, image_hidden = layer(sketch_hidden, image_hidden)
        
        # Project back to feature dimension
        interacted_sketch = self.output_proj(sketch_hidden)
        interacted_image = self.output_proj(image_hidden)
        
        # Apply layer normalization
        interacted_sketch = self.layer_norm(interacted_sketch)
        interacted_image = self.layer_norm(interacted_image)
        
        # L2 normalize
        interacted_sketch = F.normalize(interacted_sketch, p=2, dim=1)
        interacted_image = F.normalize(interacted_image, p=2, dim=1)
        
        return interacted_sketch, interacted_image
    
    def compute_interaction_score(
        self,
        sketch_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the interaction score between modalities.
        
        Args:
            sketch_features: Sketch features of shape (B, feature_dim)
            image_features: Image features of shape (B, feature_dim)
            
        Returns:
            torch.Tensor: Interaction scores of shape (B,)
        """
        # Apply interaction
        interacted_sketch, interacted_image = self.forward(sketch_features, image_features)
        
        # Compute cosine similarity
        interaction_score = F.cosine_similarity(interacted_sketch, interacted_image, dim=1)
        
        return interaction_score


class InteractionLayer(nn.Module):
    """
    Single interaction layer with cross-modal attention and feed-forward networks.
    
    This layer implements bidirectional cross-modal attention followed by
    position-wise feed-forward networks with residual connections and layer norm.
    
    Args:
        hidden_dim (int): Dimension of the hidden layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize the interaction layer.
        
        Args:
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(InteractionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Cross-modal attention for sketch
        self.sketch_attention = CrossModalAttention(
            feature_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-modal attention for image
        self.image_attention = CrossModalAttention(
            feature_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward networks
        self.sketch_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.sketch_norm1 = nn.LayerNorm(hidden_dim)
        self.sketch_norm2 = nn.LayerNorm(hidden_dim)
        self.image_norm1 = nn.LayerNorm(hidden_dim)
        self.image_norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        sketch_hidden: torch.Tensor,
        image_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply interaction layer.
        
        Args:
            sketch_hidden: Sketch hidden features of shape (B, hidden_dim)
            image_hidden: Image hidden features of shape (B, hidden_dim)
            
        Returns:
            Tuple containing:
                - sketch_output: Updated sketch hidden features (B, hidden_dim)
                - image_output: Updated image hidden features (B, hidden_dim)
        """
        # Sketch attends to image
        sketch_attention_out = self.sketch_attention(
            query=sketch_hidden,
            key=image_hidden,
            value=image_hidden
        )
        sketch_hidden = self.sketch_norm1(sketch_hidden + sketch_attention_out)
        
        # Image attends to sketch
        image_attention_out = self.image_attention(
            query=image_hidden,
            key=sketch_hidden,
            value=sketch_hidden
        )
        image_hidden = self.image_norm1(image_hidden + image_attention_out)
        
        # Feed-forward networks
        sketch_ffn_out = self.sketch_ffn(sketch_hidden)
        sketch_hidden = self.sketch_norm2(sketch_hidden + sketch_ffn_out)
        
        image_ffn_out = self.image_ffn(image_hidden)
        image_hidden = self.image_norm2(image_hidden + image_ffn_out)
        
        return sketch_hidden, image_hidden
