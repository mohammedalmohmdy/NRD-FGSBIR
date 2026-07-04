"""
Loss Functions for Neural Retrieval Dynamics (NRD)
This module implements all loss functions for training the NRD model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TripletRankingLoss(nn.Module):
    """
    Triplet Ranking Loss for sketch-based image retrieval.
    
    This loss encourages the distance between anchor and positive samples
    to be smaller than the distance between anchor and negative samples
    by a margin.
    
    The loss follows:
        L_triplet = max(0, d(a, p) - d(a, n) + margin)
        
    where a is anchor (sketch), p is positive (matching image), and n is negative (non-matching image).
    
    Args:
        margin (float): Margin for triplet loss
        mining_type (str): Type of triplet mining ('random', 'semihard', 'hard')
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        mining_type: str = 'semihard'
    ):
        """
        Initialize the triplet ranking loss.
        
        Args:
            margin: Margin for triplet loss
            mining_type: Type of triplet mining strategy
        """
        super(TripletRankingLoss, self).__init__()
        
        self.margin = margin
        self.mining_type = mining_type
        
    def forward(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet ranking loss.
        
        Args:
            anchor_features: Anchor features (sketch) of shape (B, feature_dim)
            positive_features: Positive features (matching image) of shape (B, feature_dim)
            negative_features: Negative features (non-matching image) of shape (B, feature_dim)
            
        Returns:
            torch.Tensor: Triplet loss scalar
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor_features, positive_features, p=2)
        neg_dist = F.pairwise_distance(anchor_features, negative_features, p=2)
        
        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()
    
    def compute_loss_with_mining(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss with mining strategy.
        
        Args:
            anchor_features: Anchor features of shape (B, feature_dim)
            positive_features: Positive features of shape (B, feature_dim)
            negative_features: Negative features of shape (B, feature_dim)
            labels: Labels for mining of shape (B,)
            
        Returns:
            torch.Tensor: Triplet loss scalar
        """
        if self.mining_type == 'random':
            # Random mining (use provided negatives)
            return self.forward(anchor_features, positive_features, negative_features)
        
        elif self.mining_type == 'semihard':
            # Semi-hard mining: negatives that are farther than positive but within margin
            pos_dist = F.pairwise_distance(anchor_features, positive_features, p=2)
            neg_dist = F.pairwise_distance(anchor_features, negative_features, p=2)
            
            # Find semi-hard negatives
            mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
            
            if mask.any():
                # Use semi-hard negatives
                semi_hard_negatives = negative_features[mask]
                semi_hard_anchors = anchor_features[mask]
                semi_hard_positives = positive_features[mask]
                
                return self.forward(semi_hard_anchors, semi_hard_positives, semi_hard_negatives)
            else:
                # Fall back to random mining
                return self.forward(anchor_features, positive_features, negative_features)
        
        elif self.mining_type == 'hard':
            # Hard mining: use the hardest negatives
            pos_dist = F.pairwise_distance(anchor_features, positive_features, p=2)
            neg_dist = F.pairwise_distance(anchor_features, negative_features, p=2)
            
            # Find hardest negatives (closest to anchor)
            hardest_idx = torch.argmax(neg_dist - pos_dist)
            
            hard_negative = negative_features[hardest_idx:hardest_idx+1]
            hard_anchor = anchor_features[hardest_idx:hardest_idx+1]
            hard_positive = positive_features[hardest_idx:hardest_idx+1]
            
            return self.forward(hard_anchor, hard_positive, hard_negative)
        
        else:
            raise ValueError(f"Unknown mining type: {self.mining_type}")


class TrajectoryAlignmentLoss(nn.Module):
    """
    Trajectory Alignment Loss for cross-modal trajectory learning.
    
    This loss encourages the trajectories from sketch to image and from
    image to sketch to be aligned and consistent. It measures the
    discrepancy between corresponding points on the bidirectional trajectories.
    
    The loss follows:
        L_align = Σ_{t=0}^{T} ||z_s→i(t) - z_i→s(1-t)||^2
        
    where z_s→i(t) is the point at time t on the sketch-to-image trajectory
    and z_i→s(1-t) is the corresponding point on the image-to-sketch trajectory.
    
    Args:
        num_steps (int): Number of trajectory steps
    """
    
    def __init__(self, num_steps: int = 10):
        """
        Initialize the trajectory alignment loss.
        
        Args:
            num_steps: Number of trajectory steps
        """
        super(TrajectoryAlignmentLoss, self).__init__()
        
        self.num_steps = num_steps
        
    def forward(
        self,
        sketch_to_image_traj: torch.Tensor,
        image_to_sketch_traj: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute trajectory alignment loss.
        
        Args:
            sketch_to_image_traj: Sketch-to-image trajectory of shape (B, num_steps, feature_dim)
            image_to_sketch_traj: Image-to-sketch trajectory of shape (B, num_steps, feature_dim)
            
        Returns:
            torch.Tensor: Trajectory alignment loss scalar
        """
        batch_size = sketch_to_image_traj.size(0)
        
        # Compute alignment loss at each step
        alignment_losses = []
        for t in range(self.num_steps):
            # Get corresponding points
            sketch_to_image_t = sketch_to_image_traj[:, t, :]
            image_to_sketch_t = image_to_sketch_traj[:, self.num_steps - 1 - t, :]
            
            # Compute L2 distance
            alignment_loss = F.mse_loss(sketch_to_image_t, image_to_sketch_t)
            alignment_losses.append(alignment_loss)
        
        # Average across steps
        total_loss = torch.stack(alignment_losses).mean()
        
        return total_loss
    
    def compute_trajectory_consistency(
        self,
        trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute trajectory consistency loss.
        
        This loss encourages the trajectory to be smooth and consistent.
        
        Args:
            trajectory: Trajectory of shape (B, num_steps, feature_dim)
            
        Returns:
            torch.Tensor: Trajectory consistency loss scalar
        """
        # Compute differences between consecutive steps
        diffs = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        
        # Compute variance of differences (encourages smoothness)
        consistency_loss = torch.var(diffs, dim=1).mean()
        
        return consistency_loss


class SmoothnessLoss(nn.Module):
    """
    Smoothness Loss for trajectory regularization.
    
    This loss encourages the trajectory to be smooth by penalizing
    large changes between consecutive trajectory points. This helps
    the model learn continuous and smooth cross-modal evolution.
    
    The loss follows:
        L_smooth = Σ_{t=1}^{T-1} ||z(t) - z(t-1)||^2
        
    where z(t) is the trajectory point at time t.
    
    Args:
        num_steps (int): Number of trajectory steps
    """
    
    def __init__(self, num_steps: int = 10):
        """
        Initialize the smoothness loss.
        
        Args:
            num_steps: Number of trajectory steps
        """
        super(SmoothnessLoss, self).__init__()
        
        self.num_steps = num_steps
        
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness loss for a trajectory.
        
        Args:
            trajectory: Trajectory of shape (B, num_steps, feature_dim)
            
        Returns:
            torch.Tensor: Smoothness loss scalar
        """
        # Compute differences between consecutive steps
        diffs = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        
        # Compute L2 norm of differences
        smoothness_loss = torch.norm(diffs, p=2, dim=2).mean()
        
        return smoothness_loss
    
    def compute_bidirectional_smoothness(
        self,
        sketch_to_image_traj: torch.Tensor,
        image_to_sketch_traj: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute smoothness loss for bidirectional trajectories.
        
        Args:
            sketch_to_image_traj: Sketch-to-image trajectory
            image_to_sketch_traj: Image-to-sketch trajectory
            
        Returns:
            torch.Tensor: Combined smoothness loss scalar
        """
        smoothness_loss_1 = self.forward(sketch_to_image_traj)
        smoothness_loss_2 = self.forward(image_to_sketch_traj)
        
        return (smoothness_loss_1 + smoothness_loss_2) / 2


class L2Regularization(nn.Module):
    """
    L2 Regularization loss for model parameters.
    
    This loss applies L2 regularization to model parameters to prevent
    overfitting and encourage smaller weights.
    
    The loss follows:
        L_l2 = λ * Σ ||θ||^2
        
    where θ are the model parameters and λ is the regularization coefficient.
    
    Args:
        model (nn.Module): Model to regularize
        coefficient (float): L2 regularization coefficient
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        coefficient: float = 0.001
    ):
        """
        Initialize the L2 regularization loss.
        
        Args:
            model: Model to regularize (can be set later)
            coefficient: L2 regularization coefficient
        """
        super(L2Regularization, self).__init__()
        
        self.model = model
        self.coefficient = coefficient
        
    def set_model(self, model: nn.Module) -> None:
        """
        Set the model to regularize.
        
        Args:
            model: Model to regularize
        """
        self.model = model
        
    def forward(self) -> torch.Tensor:
        """
        Compute L2 regularization loss.
        
        Returns:
            torch.Tensor: L2 regularization loss scalar
        """
        if self.model is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        l2_loss = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, p=2) ** 2
        
        return self.coefficient * l2_loss
    
    def compute_parameter_norm(self) -> torch.Tensor:
        """
        Compute the total L2 norm of model parameters.
        
        Returns:
            torch.Tensor: Total parameter norm
        """
        if self.model is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        total_norm = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                total_norm += torch.norm(param, p=2) ** 2
        
        return torch.sqrt(total_norm)


class NRDLoss(nn.Module):
    """
    Combined loss for Neural Retrieval Dynamics.
    
    This module combines all loss functions with their respective weights
    to compute the total training loss for the NRD model.
    
    The total loss follows:
        L_total = w_triplet * L_triplet + 
                  w_align * L_align + 
                  w_smooth * L_smooth + 
                  w_l2 * L_l2
    
    Args:
        triplet_loss (TripletRankingLoss): Triplet ranking loss
        trajectory_alignment_loss (TrajectoryAlignmentLoss): Trajectory alignment loss
        smoothness_loss (SmoothnessLoss): Smoothness loss
        l2_regularization (L2Regularization): L2 regularization
        loss_weights (Dict): Dictionary of loss weights
    """
    
    def __init__(
        self,
        triplet_loss: TripletRankingLoss,
        trajectory_alignment_loss: TrajectoryAlignmentLoss,
        smoothness_loss: SmoothnessLoss,
        l2_regularization: L2Regularization,
        loss_weights: Optional[dict] = None
    ):
        """
        Initialize the combined NRD loss.
        
        Args:
            triplet_loss: Triplet ranking loss module
            trajectory_alignment_loss: Trajectory alignment loss module
            smoothness_loss: Smoothness loss module
            l2_regularization: L2 regularization module
            loss_weights: Dictionary of loss weights
        """
        super(NRDLoss, self).__init__()
        
        self.triplet_loss = triplet_loss
        self.trajectory_alignment_loss = trajectory_alignment_loss
        self.smoothness_loss = smoothness_loss
        self.l2_regularization = l2_regularization
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'triplet_ranking': 1.0,
                'trajectory_alignment': 0.5,
                'smoothness': 0.1,
                'l2_regularization': 0.001
            }
        
        self.loss_weights = loss_weights
        
    def forward(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: torch.Tensor,
        sketch_to_image_traj: torch.Tensor,
        image_to_sketch_traj: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the total NRD loss.
        
        Args:
            anchor_features: Anchor features (sketch) of shape (B, feature_dim)
            positive_features: Positive features (matching image) of shape (B, feature_dim)
            negative_features: Negative features (non-matching image) of shape (B, feature_dim)
            sketch_to_image_traj: Sketch-to-image trajectory of shape (B, num_steps, feature_dim)
            image_to_sketch_traj: Image-to-sketch trajectory of shape (B, num_steps, feature_dim)
            
        Returns:
            Tuple containing:
                - total_loss: Combined loss scalar
                - loss_dict: Dictionary of individual loss values
        """
        # Compute individual losses
        triplet_loss = self.triplet_loss(anchor_features, positive_features, negative_features)
        alignment_loss = self.trajectory_alignment_loss(sketch_to_image_traj, image_to_sketch_traj)
        smoothness_loss = self.smoothness_loss.compute_bidirectional_smoothness(
            sketch_to_image_traj, image_to_sketch_traj
        )
        l2_loss = self.l2_regularization()
        
        # Combine losses with weights
        total_loss = (
            self.loss_weights['triplet_ranking'] * triplet_loss +
            self.loss_weights['trajectory_alignment'] * alignment_loss +
            self.loss_weights['smoothness'] * smoothness_loss +
            self.loss_weights['l2_regularization'] * l2_loss
        )
        
        # Create loss dictionary
        loss_dict = {
            'total_loss': total_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'l2_loss': l2_loss.item()
        }
        
        return total_loss, loss_dict
