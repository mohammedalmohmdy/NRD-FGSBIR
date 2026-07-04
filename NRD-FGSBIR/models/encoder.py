"""
Dual Representation Encoder for Neural Retrieval Dynamics (NRD)
This module implements the dual representation encoder for sketches and images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional


class SketchEncoder(nn.Module):
    """
    Sketch encoder network.
    
    This encoder processes sketch images and extracts high-level features
    for the sketch modality. It uses a CNN backbone (e.g., ResNet) with
    modifications to handle the abstract nature of sketches.
    
    Args:
        backbone (str): Backbone architecture ('resnet50', 'resnet101', etc.)
        pretrained (bool): Whether to use pretrained weights
        feature_dim (int): Dimension of the output feature vector
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 2048,
        dropout: float = 0.5
    ):
        """
        Initialize the sketch encoder.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use ImageNet pretrained weights
            feature_dim: Output feature dimension
            dropout: Dropout probability for regularization
        """
        super(SketchEncoder, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection head to map to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, feature_dim)
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the sketch encoder.
        
        Args:
            x: Input sketch tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded sketch features of shape (B, feature_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)  # Shape: (B, backbone_dim, 1, 1)
        
        # Flatten
        features = features.view(features.size(0), -1)  # Shape: (B, backbone_dim)
        
        # Project to feature dimension
        features = self.projection(features)  # Shape: (B, feature_dim)
        
        # Apply layer normalization
        features = self.layer_norm(features)  # Shape: (B, feature_dim)
        
        # L2 normalize for cosine similarity
        features = F.normalize(features, p=2, dim=1)  # Shape: (B, feature_dim)
        
        return features


class ImageEncoder(nn.Module):
    """
    Image encoder network.
    
    This encoder processes photo images and extracts high-level features
    for the image modality. It uses a CNN backbone (e.g., ResNet) to
    extract rich visual features from natural images.
    
    Args:
        backbone (str): Backbone architecture ('resnet50', 'resnet101', etc.)
        pretrained (bool): Whether to use pretrained weights
        feature_dim (int): Dimension of the output feature vector
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 2048,
        dropout: float = 0.5
    ):
        """
        Initialize the image encoder.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use ImageNet pretrained weights
            feature_dim: Output feature dimension
            dropout: Dropout probability for regularization
        """
        super(ImageEncoder, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection head to map to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, feature_dim)
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the image encoder.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded image features of shape (B, feature_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)  # Shape: (B, backbone_dim, 1, 1)
        
        # Flatten
        features = features.view(features.size(0), -1)  # Shape: (B, backbone_dim)
        
        # Project to feature dimension
        features = self.projection(features)  # Shape: (B, feature_dim)
        
        # Apply layer normalization
        features = self.layer_norm(features)  # Shape: (B, feature_dim)
        
        # L2 normalize for cosine similarity
        features = F.normalize(features, p=2, dim=1)  # Shape: (B, feature_dim)
        
        return features


class DualRepresentationEncoder(nn.Module):
    """
    Dual Representation Encoder for cross-modal retrieval.
    
    This module combines the sketch and image encoders to provide
    dual representation learning for both modalities. It enables
    learning a shared embedding space where sketches and images
    of the same category are close together.
    
    The encoder follows the equation:
        f_s = E_s(x_s)  # Sketch encoding
        f_i = E_i(x_i)  # Image encoding
        
    where E_s and E_i are the sketch and image encoders respectively.
    
    Args:
        backbone (str): Backbone architecture for both encoders
        pretrained (bool): Whether to use pretrained weights
        feature_dim (int): Dimension of the output feature vector
        dropout (float): Dropout probability
        share_weights (bool): Whether to share weights between encoders
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 2048,
        dropout: float = 0.5,
        share_weights: bool = False
    ):
        """
        Initialize the dual representation encoder.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use ImageNet pretrained weights
            feature_dim: Output feature dimension
            dropout: Dropout probability for regularization
            share_weights: Whether to share weights between encoders
        """
        super(DualRepresentationEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        self.share_weights = share_weights
        
        # Create sketch encoder
        self.sketch_encoder = SketchEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        # Create image encoder
        if share_weights:
            # Share the backbone weights but use separate projection heads
            self.image_encoder = ImageEncoder(
                backbone=backbone,
                pretrained=pretrained,
                feature_dim=feature_dim,
                dropout=dropout
            )
            # Share backbone weights
            self.image_encoder.backbone = self.sketch_encoder.backbone
        else:
            # Use separate encoders
            self.image_encoder = ImageEncoder(
                backbone=backbone,
                pretrained=pretrained,
                feature_dim=feature_dim,
                dropout=dropout
            )
    
    def forward(
        self,
        sketch: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the dual representation encoder.
        
        Args:
            sketch: Input sketch tensor of shape (B, C, H, W) or None
            image: Input image tensor of shape (B, C, H, W) or None
            
        Returns:
            Tuple containing:
                - sketch_features: Encoded sketch features of shape (B, feature_dim) or None
                - image_features: Encoded image features of shape (B, feature_dim) or None
        """
        sketch_features = None
        image_features = None
        
        if sketch is not None:
            sketch_features = self.sketch_encoder(sketch)
        
        if image is not None:
            image_features = self.image_encoder(image)
        
        return sketch_features, image_features
    
    def encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Encode sketch images.
        
        Args:
            sketch: Input sketch tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded sketch features of shape (B, feature_dim)
        """
        return self.sketch_encoder(sketch)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode photo images.
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded image features of shape (B, feature_dim)
        """
        return self.image_encoder(image)
    
    def get_feature_dim(self) -> int:
        """
        Get the output feature dimension.
        
        Returns:
            int: Feature dimension
        """
        return self.feature_dim
