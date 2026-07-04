"""
ChairV2 Dataset Loader for Neural Retrieval Dynamics (NRD)
This module implements the dataset loader for the ChairV2 dataset.
"""

import os
from typing import List, Dict, Optional
from .base_dataset import BaseDataset, get_default_transform


class ChairV2Dataset(BaseDataset):
    """
    Dataset loader for ChairV2 dataset.
    
    The ChairV2 dataset contains sketch-photo pairs of chairs.
    This class handles loading and preprocessing of the data for FG-SBIR tasks.
    
    Dataset structure:
        data_root/
            ├── sketches/
            │   ├── chair1_sketch.png
            │   └── ...
            ├── photos/
            │   ├── chair1_photo.jpg
            │   └── ...
            └── splits/
                ├── train.txt
                ├── val.txt
                └── test.txt
    
    Args:
        data_root (str): Root directory of the ChairV2 dataset
        split (str): Dataset split ('train', 'val', 'test')
        transform (Optional): Transform to apply to images
        sketch_transform (Optional): Transform to apply to sketches
        image_transform (Optional): Transform to apply to images
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None,
        sketch_transform=None,
        image_transform=None
    ):
        """
        Initialize the ChairV2 dataset.
        
        Args:
            data_root: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply to both sketches and images
            sketch_transform: Optional transform to apply only to sketches
            image_transform: Optional transform to apply only to images
        """
        super().__init__(
            data_root=data_root,
            split=split,
            transform=transform,
            sketch_transform=sketch_transform,
            image_transform=image_transform
        )
    
    def _load_dataset(self) -> None:
        """
        Load the ChairV2 dataset.
        
        This method reads the split files and populates the data lists.
        Expected directory structure:
            - sketches/: Contains sketch images
            - photos/: Contains photo images
            - splits/: Contains train.txt, val.txt, test.txt with image paths
        """
        # Define paths
        sketch_dir = os.path.join(self.data_root, 'sketches')
        photo_dir = os.path.join(self.data_root, 'photos')
        split_file = os.path.join(self.data_root, 'splits', f'{self.split}.txt')
        
        # Check if split file exists
        if os.path.exists(split_file):
            # Load from split file
            self._load_from_split_file(split_file, sketch_dir, photo_dir)
        else:
            # Load all data if split file doesn't exist
            self._load_all_data(sketch_dir, photo_dir)
    
    def _load_from_split_file(
        self,
        split_file: str,
        sketch_dir: str,
        photo_dir: str
    ) -> None:
        """
        Load dataset from split file.
        
        Args:
            split_file: Path to the split file
            sketch_dir: Directory containing sketches
            photo_dir: Directory containing photos
        """
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse line (expected format: sketch_name.jpg photo_name.jpg label)
            parts = line.split()
            if len(parts) >= 3:
                sketch_name = parts[0]
                photo_name = parts[1]
                label = int(parts[2])
                
                # Create full paths
                sketch_path = os.path.join(sketch_dir, sketch_name)
                photo_path = os.path.join(photo_dir, photo_name)
                
                # Check if files exist
                if os.path.exists(sketch_path) and os.path.exists(photo_path):
                    self.sketch_paths.append(sketch_path)
                    self.image_paths.append(photo_path)
                    self.labels.append(label)
    
    def _load_all_data(self, sketch_dir: str, photo_dir: str) -> None:
        """
        Load all data from directories.
        
        Args:
            sketch_dir: Directory containing sketches
            photo_dir: Directory containing photos
        """
        # Get all sketch files
        sketch_files = sorted([f for f in os.listdir(sketch_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for sketch_file in sketch_files:
            # Extract base name (without extension)
            base_name = os.path.splitext(sketch_file)[0]
            
            # Look for corresponding photo
            photo_extensions = ['.jpg', '.jpeg', '.png']
            photo_path = None
            
            for ext in photo_extensions:
                potential_photo = os.path.join(photo_dir, base_name + ext)
                if os.path.exists(potential_photo):
                    photo_path = potential_photo
                    break
            
            if photo_path is not None:
                sketch_path = os.path.join(sketch_dir, sketch_file)
                
                # Use filename as label (or extract from filename if needed)
                label = hash(base_name) % 1000  # Simple hash-based labeling
                
                self.sketch_paths.append(sketch_path)
                self.image_paths.append(photo_path)
                self.labels.append(label)


def create_chairv2_dataset(
    data_root: str,
    split: str = 'train',
    resize: int = 256,
    crop_size: int = 224,
    is_training: bool = True,
    augmentation_config: Optional[Dict] = None
) -> ChairV2Dataset:
    """
    Create a ChairV2 dataset with default transforms.
    
    Args:
        data_root: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
        resize: Size to resize images to
        crop_size: Size to crop images to
        is_training: Whether this is for training
        augmentation_config: Dictionary with augmentation parameters
        
    Returns:
        ChairV2Dataset: Configured dataset
    """
    # Default augmentation config
    if augmentation_config is None:
        augmentation_config = {
            'horizontal_flip': 0.5,
            'rotation': 10,
            'color_jitter': {
                'brightness': 0.4,
                'contrast': 0.4,
                'saturation': 0.4,
                'hue': 0.1
            },
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    
    # Create transforms
    transform = get_default_transform(
        resize=resize,
        crop_size=crop_size,
        horizontal_flip=augmentation_config.get('horizontal_flip', 0.5),
        rotation=augmentation_config.get('rotation', 10),
        color_jitter=augmentation_config.get('color_jitter'),
        normalize=augmentation_config.get('normalize'),
        is_training=is_training
    )
    
    # Create dataset
    dataset = ChairV2Dataset(
        data_root=data_root,
        split=split,
        transform=transform
    )
    
    return dataset
