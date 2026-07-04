"""
Inference module for Neural Retrieval Dynamics (NRD)
This module provides inference functionality for sketch-based image retrieval.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

from models import NRDModel
from datasets import create_sketchy_dataset, create_shoev2_dataset, create_chairv2_dataset
from utils import load_config


class NRDInference:
    """
    Inference class for Neural Retrieval Dynamics.
    
    This class provides methods for:
    - Loading trained models
    - Encoding sketches and images
    - Performing sketch-based image retrieval
    - Computing similarity matrices
    - Ranking and retrieving results
    
    Args:
        model (NRDModel): Trained NRD model
        config (Dict): Configuration dictionary
        device (torch.device): Device to run inference on
    """
    
    def __init__(
        self,
        model: NRDModel,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the inference module.
        
        Args:
            model: Trained NRD model
            config: Configuration dictionary
            device: Device to run inference on (auto-detected if None)
        """
        self.model = model
        self.config = config
        self.device = device or self._get_device()
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Inference settings
        self.batch_size = config.get('inference', {}).get('batch_size', 64)
        self.num_workers = config.get('inference', {}).get('num_workers', 8)
        
    def _get_device(self) -> torch.device:
        """
        Auto-detect the best available device.
        
        Returns:
            torch.device: Device to use
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config_path: str) -> 'NRDInference':
        """
        Load inference module from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file
            
        Returns:
            NRDInference: Initialized inference module
        """
        # Load configuration
        config = load_config(config_path)
        
        # Load model configuration
        model_config = config.get('model', {})
        
        # Initialize model
        model = NRDModel(
            backbone=model_config.get('sketch_encoder', {}).get('backbone', 'resnet50'),
            pretrained=False,
            feature_dim=model_config.get('sketch_encoder', {}).get('feature_dim', 2048),
            dropout=model_config.get('sketch_encoder', {}).get('dropout', 0.5),
            hidden_dim=model_config.get('trajectory', {}).get('hidden_dim', 512),
            trajectory_dim=model_config.get('trajectory', {}).get('trajectory_dim', 256),
            num_steps=model_config.get('trajectory', {}).get('num_steps', 10),
            num_heads=model_config.get('interaction', {}).get('num_heads', 8),
            num_layers=model_config.get('interaction', {}).get('num_layers', 2),
            progressive_steps=model_config.get('similarity', {}).get('progressive_steps', 5),
            temperature=model_config.get('trajectory', {}).get('temperature', 0.07),
            momentum=model_config.get('trajectory', {}).get('momentum', 0.999),
            similarity_type=model_config.get('similarity', {}).get('similarity_type', 'cosine'),
            alpha=model_config.get('similarity', {}).get('alpha', 0.5),
            beta=model_config.get('similarity', {}).get('beta', 0.3),
            margin=config.get('training', {}).get('triplet', {}).get('margin', 0.3),
            mining_type=config.get('training', {}).get('triplet', {}).get('mining_type', 'semihard'),
            loss_weights=config.get('training', {}).get('loss_weights')
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create inference module
        inference = cls(model, config)
        
        print(f"Loaded model from checkpoint: {checkpoint_path}")
        print(f"Model trained for {checkpoint['epoch']} epochs")
        
        return inference
    
    @torch.no_grad()
    def encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Encode a sketch image.
        
        Args:
            sketch: Sketch tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded sketch features of shape (feature_dim,) or (B, feature_dim)
        """
        if sketch.dim() == 3:
            sketch = sketch.unsqueeze(0)
        
        sketch = sketch.to(self.device)
        features = self.model.encode_sketch(sketch)
        
        return features.cpu()
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode a photo image.
        
        Args:
            image: Image tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded image features of shape (feature_dim,) or (B, feature_dim)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        features = self.model.encode_image(image)
        
        return features.cpu()
    
    @torch.no_grad()
    def encode_sketch_batch(self, sketches: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of sketch images.
        
        Args:
            sketches: Sketch tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded sketch features of shape (B, feature_dim)
        """
        sketches = sketches.to(self.device)
        features = self.model.encode_sketch(sketches)
        
        return features.cpu()
    
    @torch.no_grad()
    def encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of photo images.
        
        Args:
            images: Image tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Encoded image features of shape (B, feature_dim)
        """
        images = images.to(self.device)
        features = self.model.encode_image(images)
        
        return features.cpu()
    
    @torch.no_grad()
    def retrieve(
        self,
        sketch: torch.Tensor,
        gallery_features: torch.Tensor,
        gallery_paths: List[str],
        top_k: int = 10
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Retrieve top-K images for a sketch query.
        
        Args:
            sketch: Sketch tensor of shape (C, H, W)
            gallery_features: Pre-computed gallery features of shape (N, feature_dim)
            gallery_paths: List of gallery image paths
            top_k: Number of top results to return
            
        Returns:
            Tuple containing:
                - top_k_paths: List of top-K image paths
                - top_k_scores: Similarity scores for top-K results
        """
        # Encode sketch
        sketch_features = self.encode_sketch(sketch)
        
        # Compute similarity with gallery
        similarity_scores = self.model.compute_similarity_matrix(
            sketch_features,
            gallery_features
        )
        
        # Get top-K results
        similarity_scores = similarity_scores.squeeze(0)
        top_k_scores, top_k_indices = torch.topk(similarity_scores, k=top_k)
        
        # Get corresponding paths
        top_k_paths = [gallery_paths[idx] for idx in top_k_indices]
        
        return top_k_paths, top_k_scores
    
    @torch.no_grad()
    def retrieve_batch(
        self,
        sketches: torch.Tensor,
        gallery_features: torch.Tensor,
        gallery_paths: List[str],
        top_k: int = 10
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Retrieve top-K images for a batch of sketch queries.
        
        Args:
            sketches: Sketch tensor of shape (B, C, H, W)
            gallery_features: Pre-computed gallery features of shape (N, feature_dim)
            gallery_paths: List of gallery image paths
            top_k: Number of top results to return per query
            
        Returns:
            Tuple containing:
                - top_k_paths_list: List of top-K image paths for each query
                - top_k_scores: Similarity scores of shape (B, K)
        """
        # Encode sketches
        sketch_features = self.encode_sketch_batch(sketches)
        
        # Compute similarity matrix
        similarity_matrix = self.model.compute_similarity_matrix(
            sketch_features,
            gallery_features
        )
        
        # Get top-K results for each query
        top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
        
        # Get corresponding paths
        top_k_paths_list = []
        for i in range(sketches.size(0)):
            top_k_paths = [gallery_paths[idx] for idx in top_k_indices[i]]
            top_k_paths_list.append(top_k_paths)
        
        return top_k_paths_list, top_k_scores
    
    def build_gallery(
        self,
        gallery_loader: DataLoader
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Build feature gallery from image dataset.
        
        Args:
            gallery_loader: DataLoader for gallery images
            
        Returns:
            Tuple containing:
                - gallery_features: Gallery features of shape (N, feature_dim)
                - gallery_paths: List of gallery image paths
        """
        gallery_features_list = []
        gallery_paths = []
        
        for batch in tqdm(gallery_loader, desc='Building gallery'):
            images = batch['image']
            paths = batch['image_path']
            
            # Encode images
            features = self.encode_image_batch(images)
            
            gallery_features_list.append(features)
            gallery_paths.extend(paths)
        
        # Concatenate features
        gallery_features = torch.cat(gallery_features_list, dim=0)
        
        return gallery_features, gallery_paths
    
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
        return self.model.compute_similarity_matrix(query_features, gallery_features)
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor of shape (C, H, W)
        """
        from torchvision import transforms
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get preprocessing transform from config
        augmentation_config = self.config.get('augmentation', {}).get('val', {})
        
        normalize = transforms.Normalize(
            mean=augmentation_config.get('normalize', {}).get('mean', [0.485, 0.456, 0.406]),
            std=augmentation_config.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
        )
        
        transform = transforms.Compose([
            transforms.Resize(augmentation_config.get('resize', 256)),
            transforms.CenterCrop(augmentation_config.get('crop_size', 224)),
            transforms.ToTensor(),
            normalize
        ])
        
        # Apply transform
        image_tensor = transform(image)
        
        return image_tensor


def main():
    """
    Main function for running inference.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='NRD Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--query', type=str, required=True, help='Path to query sketch')
    parser.add_argument('--gallery', type=str, required=True, help='Path to gallery directory')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to return')
    parser.add_argument('--output', type=str, default='results.txt', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load inference module
    inference = NRDInference.from_checkpoint(args.checkpoint, args.config)
    
    # Load query sketch
    query_sketch = inference.load_image(args.query)
    
    # Build gallery (simplified - in practice, use a DataLoader)
    # This is a placeholder - actual implementation would use a proper DataLoader
    print("Note: Gallery building requires proper DataLoader implementation")
    print("This is a simplified inference example")
    
    # For demonstration, just encode the query
    query_features = inference.encode_sketch(query_sketch)
    print(f"Query feature shape: {query_features.shape}")
    
    print("Inference complete")


if __name__ == '__main__':
    main()
