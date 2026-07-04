"""
Sketchy Extended Dataset Loader for Neural Retrieval Dynamics (NRD)
This module implements the dataset loader for the Sketchy Extended dataset.
"""

import os
from typing import List, Dict, Optional
from .base_dataset import BaseDataset, get_default_transform


class SketchyExtendedDataset(BaseDataset):
    """
    Dataset loader for Sketchy Extended dataset.
    
    The Sketchy Extended dataset contains sketch-photo pairs across multiple categories.
    This class handles loading and preprocessing of the data for FG-SBIR tasks.
    
    Dataset structure:
        data_root/
            ├── sketches/
            │   ├── category1/
            │   │   ├── sketch1.png
            │   │   └── ...
            │   └── ...
            ├── photos/
            │   ├── category1/
            │   │   ├── photo1.jpg
            │   │   └── ...
            │   └── ...
            └── splits/
                ├── train.txt
                ├── val.txt
                └── test.txt
    
    Args:
        data_root (str): Root directory of the Sketchy Extended dataset
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
        Initialize the Sketchy Extended dataset.
        
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
        Load the Sketchy Extended dataset.
        
        This method reads the split files and populates the data lists.
        Expected directory structure:
            - sketches/: Contains sketch images organized by category
            - photos/: Contains photo images organized by category
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
            
            # Parse line (expected format: category/sketch_name.jpg category/photo_name.jpg)
            parts = line.split()
            if len(parts) >= 2:
                sketch_rel_path = parts[0]
                photo_rel_path = parts[1]
                
                # Extract category from path
                category = sketch_rel_path.split('/')[0]
                
                # Create full paths
                sketch_path = os.path.join(sketch_dir, sketch_rel_path)
                photo_path = os.path.join(photo_dir, photo_rel_path)
                
                # Check if files exist
                if os.path.exists(sketch_path) and os.path.exists(photo_path):
                    self.sketch_paths.append(sketch_path)
                    self.image_paths.append(photo_path)
                    self.labels.append(self._get_category_id(category))
    
    def _load_all_data(self, sketch_dir: str, photo_dir: str) -> None:
        """
        Load all data from directories.
        
        Args:
            sketch_dir: Directory containing sketches
            photo_dir: Directory containing photos
        """
        # Get all categories
        categories = sorted([d for d in os.listdir(sketch_dir) 
                            if os.path.isdir(os.path.join(sketch_dir, d))])
        
        category_to_id = {cat: idx for idx, cat in enumerate(categories)}
        
        for category in categories:
            category_id = category_to_id[category]
            
            # Get sketch paths
            sketch_cat_dir = os.path.join(sketch_dir, category)
            photo_cat_dir = os.path.join(photo_dir, category)
            
            if not os.path.exists(photo_cat_dir):
                continue
            
            sketch_files = sorted([f for f in os.listdir(sketch_cat_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
            photo_files = sorted([f for f in os.listdir(photo_cat_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # Create pairs (assuming one-to-one correspondence)
            for sketch_file, photo_file in zip(sketch_files, photo_files):
                sketch_path = os.path.join(sketch_cat_dir, sketch_file)
                photo_path = os.path.join(photo_cat_dir, photo_file)
                
                self.sketch_paths.append(sketch_path)
                self.image_paths.append(photo_path)
                self.labels.append(category_id)
    
    def _get_category_id(self, category: str) -> int:
        """
        Get the numeric ID for a category.
        
        Args:
            category: Category name
            
        Returns:
            int: Numeric category ID
        """
        # Collect all unique categories
        all_categories = set()
        for path in self.sketch_paths:
            cat = os.path.basename(os.path.dirname(path))
            all_categories.add(cat)
        
        sorted_categories = sorted(all_categories)
        category_to_id = {cat: idx for idx, cat in enumerate(sorted_categories)}
        
        return category_to_id.get(category, 0)


def create_sketchy_dataset(
    data_root: str,
    split: str = 'train',
    resize: int = 256,
    crop_size: int = 224,
    is_training: bool = True,
    augmentation_config: Optional[Dict] = None
) -> SketchyExtendedDataset:
    """
    Create a Sketchy Extended dataset with default transforms.
    
    Args:
        data_root: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
        resize: Size to resize images to
        crop_size: Size to crop images to
        is_training: Whether this is for training
        augmentation_config: Dictionary with augmentation parameters
        
    Returns:
        SketchyExtendedDataset: Configured dataset
    """
    # Default augmentation config
    if augmentation_config is None:
        augmentation_config = {
            'horizontal_flip': 0.5,
            'rotation': 15,
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
        rotation=augmentation_config.get('rotation', 15),
        color_jitter=augmentation_config.get('color_jitter'),
        normalize=augmentation_config.get('normalize'),
        is_training=is_training
    )
    
    # Create dataset
    dataset = SketchyExtendedDataset(
        data_root=data_root,
        split=split,
        transform=transform
    )
    
    return dataset
