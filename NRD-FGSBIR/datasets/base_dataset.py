"""
Base Dataset Class for Neural Retrieval Dynamics (NRD)
This module provides the base dataset class that all specific dataset implementations inherit from.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BaseDataset(Dataset):
    """
    Base dataset class for Fine-Grained Sketch-Based Image Retrieval (FG-SBIR).
    
    This class provides the common functionality for loading sketch-image pairs,
    applying data augmentation, and preparing data for training and evaluation.
    
    Args:
        data_root (str): Root directory of the dataset
        split (str): Dataset split ('train', 'val', 'test')
        transform (Optional[Callable]): Transform to apply to images
        sketch_transform (Optional[Callable]): Transform to apply to sketches
        image_transform (Optional[Callable]): Transform to apply to images
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        sketch_transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None
    ):
        """
        Initialize the base dataset.
        
        Args:
            data_root: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply to both sketches and images
            sketch_transform: Optional transform to apply only to sketches
            image_transform: Optional transform to apply only to images
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.sketch_transform = sketch_transform
        self.image_transform = image_transform
        
        # Lists to store data paths and labels
        self.sketch_paths: List[str] = []
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        
        # Load the dataset
        self._load_dataset()
        
    def _load_dataset(self) -> None:
        """
        Load the dataset. This method should be overridden by subclasses.
        
        This method should populate:
        - self.sketch_paths: List of sketch image paths
        - self.image_paths: List of corresponding image paths
        - self.labels: List of class labels
        """
        raise NotImplementedError("Subclasses must implement _load_dataset method")
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.sketch_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dict containing:
                - 'sketch': Sketch image tensor (C, H, W)
                - 'image': Photo image tensor (C, H, W)
                - 'label': Class label tensor
                - 'sketch_path': Path to sketch image
                - 'image_path': Path to photo image
        """
        # Load sketch and image
        sketch_path = self.sketch_paths[idx]
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load images as PIL Images
        sketch = self._load_image(sketch_path)
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.sketch_transform is not None:
            sketch = self._apply_transform(sketch, self.sketch_transform)
        elif self.transform is not None:
            sketch = self._apply_transform(sketch, self.transform)
            
        if self.image_transform is not None:
            image = self._apply_transform(image, self.image_transform)
        elif self.transform is not None:
            image = self._apply_transform(image, self.transform)
        
        # Convert to tensor if not already
        if not isinstance(sketch, torch.Tensor):
            sketch = torch.from_numpy(sketch).permute(2, 0, 1).float()
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return {
            'sketch': sketch,
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'sketch_path': sketch_path,
            'image_path': image_path
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load an image from the given path.
        
        Args:
            path: Path to the image file
            
        Returns:
            np.ndarray: Image as numpy array (H, W, C)
        """
        # Load image
        image = Image.open(path).convert('RGB')
        image = np.array(image)
        
        return image
    
    def _apply_transform(self, image: np.ndarray, transform: Callable) -> np.ndarray:
        """
        Apply transform to the image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            transform: Transform function to apply
            
        Returns:
            np.ndarray: Transformed image
        """
        if isinstance(transform, A.Compose):
            transformed = transform(image=image)
            image = transformed['image']
        else:
            image = transform(image)
        
        return image
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset.
        
        Returns:
            int: Number of unique classes
        """
        return len(set(self.labels))
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of samples per class.
        
        Returns:
            Dict[int, int]: Mapping from class label to number of samples
        """
        distribution = {}
        for label in self.labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


def get_default_transform(
    resize: int = 256,
    crop_size: int = 224,
    horizontal_flip: float = 0.5,
    rotation: int = 15,
    color_jitter: Optional[Dict] = None,
    normalize: Optional[Dict] = None,
    is_training: bool = True
) -> A.Compose:
    """
    Get default data augmentation transform.
    
    Args:
        resize: Size to resize images to
        crop_size: Size to crop images to
        horizontal_flip: Probability of horizontal flip
        rotation: Maximum rotation angle in degrees
        color_jitter: Dictionary with color jitter parameters
        normalize: Dictionary with normalization parameters
        is_training: Whether this is for training (enables augmentation)
        
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    transforms = []
    
    # Resize
    transforms.append(A.Resize(resize, resize))
    
    if is_training:
        # Random crop
        transforms.append(A.RandomCrop(crop_size, crop_size))
        
        # Horizontal flip
        if horizontal_flip > 0:
            transforms.append(A.HorizontalFlip(p=horizontal_flip))
        
        # Rotation
        if rotation > 0:
            transforms.append(A.Rotate(limit=rotation, p=0.5))
        
        # Color jitter
        if color_jitter is not None:
            transforms.append(A.ColorJitter(
                brightness=color_jitter.get('brightness', 0),
                contrast=color_jitter.get('contrast', 0),
                saturation=color_jitter.get('saturation', 0),
                hue=color_jitter.get('hue', 0),
                p=0.5
            ))
    else:
        # Center crop for validation/test
        transforms.append(A.CenterCrop(crop_size, crop_size))
    
    # Normalization
    if normalize is not None:
        transforms.append(A.Normalize(
            mean=normalize['mean'],
            std=normalize['std']
        ))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)
