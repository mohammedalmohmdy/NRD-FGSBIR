"""
Continuous Cross-Modal Trajectory Learning for Neural Retrieval Dynamics (NRD)
This module implements the trajectory learning component for continuous cross-modal evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class TrajectoryLearner(nn.Module):
    """
    Trajectory Learner for generating continuous cross-modal trajectories.
    
    This module learns to generate a trajectory of feature representations
    that evolve from one modality to another. The trajectory is parameterized
    by a continuous parameter t in [0, 1], where t=0 corresponds to the source
    modality and t=1 corresponds to the target modality.
    
    The trajectory follows the equation:
        z(t) = (1 - t) * f_s + t * f_i + Δ(t)
        
    where Δ(t) is a learned deviation that captures the cross-modal dynamics.
    
    Args:
        feature_dim (int): Dimension of the feature space
        hidden_dim (int): Dimension of the hidden layers
        trajectory_dim (int): Dimension of the trajectory representation
        num_steps (int): Number of discrete steps in the trajectory
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        trajectory_dim: int = 256,
        num_steps: int = 10
    ):
        """
        Initialize the trajectory learner.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            trajectory_dim: Trajectory representation dimension
            num_steps: Number of discrete steps in the trajectory
        """
        super(TrajectoryLearner, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.trajectory_dim = trajectory_dim
        self.num_steps = num_steps
        
        # Time embedding network
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, trajectory_dim)
        )
        
        # Feature projection network
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, trajectory_dim)
        )
        
        # Trajectory generation network
        self.trajectory_generator = nn.Sequential(
            nn.Linear(trajectory_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(
        self,
        sketch_features: torch.Tensor,
        image_features: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate trajectory point at time t.
        
        Args:
            sketch_features: Sketch features of shape (B, feature_dim)
            image_features: Image features of shape (B, feature_dim)
            t: Time parameter of shape (B, 1) with values in [0, 1]
            
        Returns:
            torch.Tensor: Trajectory point at time t of shape (B, feature_dim)
        """
        batch_size = sketch_features.size(0)
        
        # Concatenate sketch and image features
        combined_features = torch.cat([sketch_features, image_features], dim=1)
        
        # Project combined features to trajectory space
        feature_proj = self.feature_projection(combined_features)
        
        # Embed time parameter
        time_emb = self.time_embedding(t)
        
        # Combine feature projection and time embedding
        combined = torch.cat([feature_proj, time_emb], dim=1)
        
        # Generate trajectory deviation
        deviation = self.trajectory_generator(combined)
        
        # Linear interpolation between sketch and image features
        linear_interp = (1 - t) * sketch_features + t * image_features
        
        # Add learned deviation
        trajectory_point = linear_interp + deviation
        
        # Apply layer normalization
        trajectory_point = self.layer_norm(trajectory_point)
        
        # L2 normalize
        trajectory_point = F.normalize(trajectory_point, p=2, dim=1)
        
        return trajectory_point
    
    def generate_trajectory(
        self,
        sketch_features: torch.Tensor,
        image_features: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate the full trajectory from sketch to image.
        
        Args:
            sketch_features: Sketch features of shape (B, feature_dim)
            image_features: Image features of shape (B, feature_dim)
            num_steps: Number of steps in the trajectory (uses self.num_steps if None)
            
        Returns:
            torch.Tensor: Full trajectory of shape (B, num_steps, feature_dim)
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        batch_size = sketch_features.size(0)
        device = sketch_features.device
        
        # Generate time steps
        time_steps = torch.linspace(0, 1, num_steps, device=device)
        time_steps = time_steps.unsqueeze(0).expand(batch_size, -1, 1)
        
        # Generate trajectory points
        trajectory_points = []
        for i in range(num_steps):
            t = time_steps[:, i:i+1, :]
            point = self.forward(sketch_features, image_features, t)
            trajectory_points.append(point)
        
        # Stack trajectory points
        trajectory = torch.stack(trajectory_points, dim=1)
        
        return trajectory


class CrossModalTrajectory(nn.Module):
    """
    Cross-Modal Trajectory module for continuous evolution between modalities.
    
    This module implements the continuous cross-modal trajectory learning
    as described in the NRD framework. It learns to generate smooth trajectories
    that bridge the gap between sketch and image modalities.
    
    The trajectory evolution follows:
        τ = {z(t) | t ∈ [0, 1]}
        
    where each point z(t) represents a state in the cross-modal evolution.
    
    Args:
        feature_dim (int): Dimension of the feature space
        hidden_dim (int): Dimension of the hidden layers
        trajectory_dim (int): Dimension of the trajectory representation
        num_steps (int): Number of discrete steps in the trajectory
        temperature (float): Temperature parameter for trajectory smoothness
        momentum (float): Momentum for trajectory updates
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        trajectory_dim: int = 256,
        num_steps: int = 10,
        temperature: float = 0.07,
        momentum: float = 0.999
    ):
        """
        Initialize the cross-modal trajectory module.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            trajectory_dim: Trajectory representation dimension
            num_steps: Number of discrete steps in the trajectory
            temperature: Temperature for trajectory smoothness
            momentum: Momentum for trajectory updates
        """
        super(CrossModalTrajectory, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.trajectory_dim = trajectory_dim
        self.num_steps = num_steps
        self.temperature = temperature
        self.momentum = momentum
        
        # Sketch to image trajectory learner
        self.sketch_to_image = TrajectoryLearner(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            trajectory_dim=trajectory_dim,
            num_steps=num_steps
        )
        
        # Image to sketch trajectory learner
        self.image_to_sketch = TrajectoryLearner(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            trajectory_dim=trajectory_dim,
            num_steps=num_steps
        )
        
        # Momentum encoder for sketch features
        self.sketch_momentum_encoder = MomentumEncoder(
            feature_dim=feature_dim,
            momentum=momentum
        )
        
        # Momentum encoder for image features
        self.image_momentum_encoder = MomentumEncoder(
            feature_dim=feature_dim,
            momentum=momentum
        )
        
    def forward(
        self,
        sketch_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate bidirectional trajectories.
        
        Args:
            sketch_features: Sketch features of shape (B, feature_dim)
            image_features: Image features of shape (B, feature_dim)
            
        Returns:
            Tuple containing:
                - sketch_to_image_traj: Trajectory from sketch to image (B, num_steps, feature_dim)
                - image_to_sketch_traj: Trajectory from image to sketch (B, num_steps, feature_dim)
        """
        # Update momentum encoders
        sketch_momentum_features = self.sketch_momentum_encoder(sketch_features)
        image_momentum_features = self.image_momentum_encoder(image_features)
        
        # Generate sketch to image trajectory
        sketch_to_image_traj = self.sketch_to_image.generate_trajectory(
            sketch_features,
            image_momentum_features,
            self.num_steps
        )
        
        # Generate image to sketch trajectory
        image_to_sketch_traj = self.image_to_sketch.generate_trajectory(
            image_features,
            sketch_momentum_features,
            self.num_steps
        )
        
        return sketch_to_image_traj, image_to_sketch_traj
    
    def get_trajectory_point(
        self,
        sketch_features: torch.Tensor,
        image_features: torch.Tensor,
        t: float,
        direction: str = 'sketch_to_image'
    ) -> torch.Tensor:
        """
        Get a specific point on the trajectory.
        
        Args:
            sketch_features: Sketch features of shape (B, feature_dim)
            image_features: Image features of shape (B, feature_dim)
            t: Time parameter in [0, 1]
            direction: Direction of trajectory ('sketch_to_image' or 'image_to_sketch')
            
        Returns:
            torch.Tensor: Trajectory point at time t of shape (B, feature_dim)
        """
        batch_size = sketch_features.size(0)
        device = sketch_features.device
        
        t_tensor = torch.tensor([[t]], device=device).expand(batch_size, 1, 1)
        
        if direction == 'sketch_to_image':
            point = self.sketch_to_image(sketch_features, image_features, t_tensor)
        elif direction == 'image_to_sketch':
            point = self.image_to_sketch(image_features, sketch_features, t_tensor)
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        return point.squeeze(1)


class MomentumEncoder(nn.Module):
    """
    Momentum encoder for maintaining consistent feature representations.
    
    This encoder uses exponential moving average (EMA) updates to maintain
    a stable representation of features across training iterations.
    
    Args:
        feature_dim (int): Dimension of the feature space
        momentum (float): Momentum coefficient for EMA updates
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        momentum: float = 0.999
    ):
        """
        Initialize the momentum encoder.
        
        Args:
            feature_dim: Feature dimension
            momentum: Momentum coefficient
        """
        super(MomentumEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # Simple projection for momentum encoder
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Register buffer for momentum features
        self.register_buffer('momentum_features', None)
        
    @torch.no_grad()
    def update_momentum(self, features: torch.Tensor) -> None:
        """
        Update momentum features using EMA.
        
        Args:
            features: Current features of shape (B, feature_dim)
        """
        if self.momentum_features is None:
            self.momentum_features = features.clone()
        else:
            self.momentum_features = (
                self.momentum * self.momentum_features +
                (1 - self.momentum) * features
            )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with momentum update.
        
        Args:
            features: Input features of shape (B, feature_dim)
            
        Returns:
            torch.Tensor: Momentum-encoded features of shape (B, feature_dim)
        """
        # Update momentum
        self.update_momentum(features)
        
        # Apply projection
        momentum_proj = self.projection(self.momentum_features)
        
        # L2 normalize
        momentum_proj = F.normalize(momentum_proj, p=2, dim=1)
        
        return momentum_proj
