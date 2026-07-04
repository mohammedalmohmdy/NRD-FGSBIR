"""
Neural Retrieval Dynamics (NRD) Main Model
This module implements the complete NRD model combining all components.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from .encoder import DualRepresentationEncoder
from .trajectory import CrossModalTrajectory
from .interaction import DynamicInteractionModule
from .similarity import ProgressiveSimilarity
from .loss import (
    TripletRankingLoss,
    TrajectoryAlignmentLoss,
    SmoothnessLoss,
    L2Regularization,
    NRDLoss
)


class NRDModel(nn.Module):
    """
    Neural Retrieval Dynamics (NRD) Model for Fine-Grained Sketch-Based Image Retrieval.
    
    This model implements the complete NRD framework with:
    1. Dual Representation Encoder for sketch and image encoding
    2. Continuous Cross-Modal Trajectory Learning
    3. Dynamic Retrieval Interaction Module
    4. Progressive Similarity Optimization
    
    The model follows the architecture:
        Input (sketch, image) → Encoder → Trajectory → Interaction → Similarity → Loss
    
    Args:
        backbone (str): Backbone architecture for encoders
        pretrained (bool): Whether to use pretrained weights
        feature_dim (int): Dimension of the feature space
        hidden_dim (int): Dimension of hidden layers
        trajectory_dim (int): Dimension of trajectory representation
        num_steps (int): Number of trajectory steps
        num_heads (int): Number of attention heads
        num_layers (int): Number of interaction layers
        progressive_steps (int): Number of progressive similarity stages
        dropout (float): Dropout probability
        temperature (float): Temperature for trajectory learning
        momentum (float): Momentum for momentum encoders
        similarity_type (str): Type of similarity measure
        alpha (float): Weight for progressive similarity
        beta (float): Weight for trajectory similarity
        margin (float): Margin for triplet loss
        mining_type (str): Type of triplet mining
        loss_weights (Dict): Dictionary of loss weights
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        trajectory_dim: int = 256,
        num_steps: int = 10,
        num_heads: int = 8,
        num_layers: int = 2,
        progressive_steps: int = 5,
        dropout: float = 0.5,
        temperature: float = 0.07,
        momentum: float = 0.999,
        similarity_type: str = 'cosine',
        alpha: float = 0.5,
        beta: float = 0.3,
        margin: float = 0.3,
        mining_type: str = 'semihard',
        loss_weights: Optional[Dict] = None
    ):
        """
        Initialize the NRD model.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            feature_dim: Feature dimension
            hidden_dim: Hidden layer dimension
            trajectory_dim: Trajectory representation dimension
            num_steps: Number of trajectory steps
            num_heads: Number of attention heads
            num_layers: Number of interaction layers
            progressive_steps: Number of progressive stages
            dropout: Dropout probability
            temperature: Temperature for trajectory learning
            momentum: Momentum for momentum encoders
            similarity_type: Type of similarity measure
            alpha: Weight for progressive similarity
            beta: Weight for trajectory similarity
            margin: Margin for triplet loss
            mining_type: Type of triplet mining
            loss_weights: Dictionary of loss weights
        """
        super(NRDModel, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        
        # Dual Representation Encoder
        self.encoder = DualRepresentationEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            dropout=dropout,
            share_weights=False
        )
        
        # Cross-Modal Trajectory Learning
        self.trajectory = CrossModalTrajectory(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            trajectory_dim=trajectory_dim,
            num_steps=num_steps,
            temperature=temperature,
            momentum=momentum
        )
        
        # Dynamic Retrieval Interaction Module
        self.interaction = DynamicInteractionModule(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Progressive Similarity Optimization
        self.similarity = ProgressiveSimilarity(
            feature_dim=feature_dim,
            num_steps=num_steps,
            progressive_steps=progressive_steps,
            similarity_type=similarity_type,
            alpha=alpha,
            beta=beta
        )
        
        # Loss functions
        self.triplet_loss = TripletRankingLoss(
            margin=margin,
            mining_type=mining_type
        )
        
        self.trajectory_alignment_loss = TrajectoryAlignmentLoss(
            num_steps=num_steps
        )
        
        self.smoothness_loss = SmoothnessLoss(
            num_steps=num_steps
        )
        
        self.l2_regularization = L2Regularization(
            model=self,
            coefficient=loss_weights.get('l2_regularization', 0.001) if loss_weights else 0.001
        )
        
        # Combined loss
        self.nrd_loss = NRDLoss(
            triplet_loss=self.triplet_loss,
            trajectory_alignment_loss=self.trajectory_alignment_loss,
            smoothness_loss=self.smoothness_loss,
            l2_regularization=self.l2_regularization,
            loss_weights=loss_weights
        )
        
    def forward(
        self,
        sketch: torch.Tensor,
        image: torch.Tensor,
        return_trajectories: bool = False,
        return_interactions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the NRD model.
        
        This implements the complete forward pass:
        1. Encode sketches and images
        2. Generate cross-modal trajectories
        3. Apply dynamic interaction
        4. Compute progressive similarity
        
        Args:
            sketch: Input sketch tensor of shape (B, C, H, W)
            image: Input image tensor of shape (B, C, H, W)
            return_trajectories: Whether to return trajectory information
            return_interactions: Whether to return interaction information
            
        Returns:
            Dict containing:
                - 'sketch_features': Encoded sketch features (B, feature_dim)
                - 'image_features': Encoded image features (B, feature_dim)
                - 'similarity': Progressive similarity scores (B,)
                - 'sketch_to_image_traj': Sketch-to-image trajectory (B, num_steps, feature_dim) (optional)
                - 'image_to_sketch_traj': Image-to-sketch trajectory (B, num_steps, feature_dim) (optional)
                - 'interacted_sketch': Interacted sketch features (B, feature_dim) (optional)
                - 'interacted_image': Interacted image features (B, feature_dim) (optional)
        """
        # Step 1: Encode sketches and images
        sketch_features, image_features = self.encoder(sketch, image)
        
        # Step 2: Generate cross-modal trajectories
        sketch_to_image_traj, image_to_sketch_traj = self.trajectory(
            sketch_features, image_features
        )
        
        # Step 3: Apply dynamic interaction
        interacted_sketch, interacted_image = self.interaction(
            sketch_features, image_features
        )
        
        # Step 4: Compute progressive similarity
        progressive_sim, trajectory_sim = self.similarity(
            sketch_to_image_traj,
            image_to_sketch_traj,
            interacted_sketch,
            interacted_image
        )
        
        # Build output dictionary
        output = {
            'sketch_features': sketch_features,
            'image_features': image_features,
            'similarity': progressive_sim,
            'trajectory_similarity': trajectory_sim
        }
        
        if return_trajectories:
            output['sketch_to_image_traj'] = sketch_to_image_traj
            output['image_to_sketch_traj'] = image_to_sketch_traj
        
        if return_interactions:
            output['interacted_sketch'] = interacted_sketch
            output['interacted_image'] = interacted_image
        
        return output
    
    def compute_loss(
        self,
        sketch: torch.Tensor,
        positive_image: torch.Tensor,
        negative_image: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the training loss.
        
        This implements the complete training loop:
        1. Forward pass
        2. Trajectory evolution
        3. Similarity accumulation
        4. Loss computation
        
        Args:
            sketch: Input sketch tensor of shape (B, C, H, W)
            positive_image: Positive (matching) image tensor of shape (B, C, H, W)
            negative_image: Negative (non-matching) image tensor of shape (B, C, H, W)
            labels: Optional labels for triplet mining
            
        Returns:
            Tuple containing:
                - total_loss: Combined loss scalar
                - loss_dict: Dictionary of individual loss values
        """
        # Forward pass for positive pair
        positive_output = self.forward(
            sketch,
            positive_image,
            return_trajectories=True,
            return_interactions=True
        )
        
        # Forward pass for negative pair
        negative_output = self.forward(
            sketch,
            negative_image,
            return_trajectories=True,
            return_interactions=False
        )
        
        # Extract features
        anchor_features = positive_output['sketch_features']
        positive_features = positive_output['image_features']
        negative_features = negative_output['image_features']
        
        # Extract trajectories
        sketch_to_image_traj = positive_output['sketch_to_image_traj']
        image_to_sketch_traj = positive_output['image_to_sketch_traj']
        
        # Compute loss
        total_loss, loss_dict = self.nrd_loss(
            anchor_features=anchor_features,
            positive_features=positive_features,
            negative_features=negative_features,
            sketch_to_image_traj=sketch_to_image_traj,
            image_to_sketch_traj=image_to_sketch_traj
        )
        
        return total_loss, loss_dict
    
    def encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Encode sketch images.
        
        Args:
            sketch: Input sketch tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded sketch features of shape (B, feature_dim)
        """
        sketch_features, _ = self.encoder(sketch=sketch)
        return sketch_features
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode photo images.
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded image features of shape (B, feature_dim)
        """
        _, image_features = self.encoder(image=image)
        return image_features
    
    def compute_similarity(
        self,
        sketch_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between sketch and image features.
        
        Args:
            sketch_features: Sketch features of shape (B, feature_dim)
            image_features: Image features of shape (B, feature_dim)
            
        Returns:
            torch.Tensor: Similarity scores of shape (B,)
        """
        return self.similarity.compute_pairwise_similarity(sketch_features, image_features)
    
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
        return self.similarity.compute_similarity_matrix(query_features, gallery_features)
    
    def get_feature_dim(self) -> int:
        """
        Get the feature dimension.
        
        Returns:
            int: Feature dimension
        """
        return self.feature_dim
    
    def get_num_steps(self) -> int:
        """
        Get the number of trajectory steps.
        
        Returns:
            int: Number of trajectory steps
        """
        return self.num_steps
