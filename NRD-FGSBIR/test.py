"""
Testing and evaluation script for Neural Retrieval Dynamics (NRD)
This script provides the entry point for testing and evaluating the NRD model.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import warnings
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import NRDModel
from datasets import create_sketchy_dataset, create_shoev2_dataset, create_chairv2_dataset
from trainer import Validator
from utils import load_config, set_reproducibility, setup_logger

warnings.filterwarnings('ignore')


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Test Neural Retrieval Dynamics (NRD) Model')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['sketchy_extended', 'shoev2', 'chairv2'],
                        help='Dataset name (overrides config)')
    parser.add_argument('--data_root', type=str, help='Data root directory (overrides config)')
    
    # Testing settings
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers (overrides config)')
    
    # Hardware
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use (overrides config)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--save_features', action='store_true', help='Save extracted features')
    parser.add_argument('--per_class', action='store_true', help='Compute per-class metrics')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, help='Random seed (overrides config)')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """
    Update configuration with command-line arguments.
    
    Args:
        config: Configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        dict: Updated configuration
    """
    if args.dataset:
        config['training']['dataset'] = args.dataset
    
    if args.data_root:
        config['training']['data_root'] = args.data_root
    
    if args.batch_size:
        config['validation']['batch_size'] = args.batch_size
    
    if args.num_workers:
        config['validation']['num_workers'] = args.num_workers
    
    if args.device:
        config['hardware']['device'] = args.device
    
    if args.seed:
        config['reproducibility']['seed'] = args.seed
    
    return config


def create_dataset(config, split):
    """
    Create dataset based on configuration.
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        Dataset: Configured dataset
    """
    dataset_name = config['training']['dataset']
    data_root = config['training']['data_root']
    
    # Get augmentation config
    augmentation_config = config.get('augmentation', {})
    val_aug = augmentation_config.get('val', {})
    
    if dataset_name == 'sketchy_extended':
        from datasets import create_sketchy_dataset
        dataset = create_sketchy_dataset(
            data_root=data_root,
            split=split,
            resize=val_aug.get('resize', 256),
            crop_size=val_aug.get('crop_size', 224),
            is_training=False,
            augmentation_config=val_aug
        )
    elif dataset_name == 'shoev2':
        from datasets import create_shoev2_dataset
        dataset = create_shoev2_dataset(
            data_root=data_root,
            split=split,
            resize=val_aug.get('resize', 256),
            crop_size=val_aug.get('crop_size', 224),
            is_training=False,
            augmentation_config=val_aug
        )
    elif dataset_name == 'chairv2':
        from datasets import create_chairv2_dataset
        dataset = create_chairv2_dataset(
            data_root=data_root,
            split=split,
            resize=val_aug.get('resize', 256),
            crop_size=val_aug.get('crop_size', 224),
            is_training=False,
            augmentation_config=val_aug
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_model(config):
    """
    Create NRD model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        NRDModel: Configured model
    """
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    model = NRDModel(
        backbone=model_config.get('sketch_encoder', {}).get('backbone', 'resnet50'),
        pretrained=False,  # Don't use pretrained weights when loading checkpoint
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
        margin=training_config.get('triplet', {}).get('margin', 0.3),
        mining_type=training_config.get('triplet', {}).get('mining_type', 'semihard'),
        loss_weights=training_config.get('loss_weights')
    )
    
    return model


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model from checkpoint.
    
    Args:
        model: NRD model
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")


def save_results(metrics, output_dir, filename='results.json'):
    """
    Save evaluation results to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save results
        filename: Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Convert tensors to floats for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_serializable[key] = value.item()
        else:
            metrics_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    
    print(f"Results saved to: {output_path}")


def main():
    """
    Main testing function.
    """
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command-line arguments
    config = update_config_from_args(config, args)
    
    # Setup reproducibility
    set_reproducibility(config)
    
    # Setup logger
    logger = setup_logger(
        name='NRD_Testing',
        log_dir=config.get('logging', {}).get('log_dir', './logs')
    )
    
    logger.info("Starting NRD testing")
    logger.info(f"Configuration: {config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Print configuration
    print("="*50)
    print("Configuration:")
    print("="*50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*50)
    
    # Create dataset
    print(f"\nCreating {args.split} dataset...")
    dataset = create_dataset(config, args.split)
    
    print(f"{args.split.capitalize()} samples: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    
    # Create data loader
    batch_size = config.get('validation', {}).get('batch_size', config['training']['batch_size'])
    num_workers = config.get('validation', {}).get('num_workers', config['training']['num_workers'])
    
    from utils.seed import get_worker_init_fn
    worker_init_fn = get_worker_init_fn(config['reproducibility']['seed'])
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Load checkpoint
    device = torch.device(config.get('hardware', {}).get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    
    # Create validator
    print("\nInitializing validator...")
    validator = Validator(
        model=model,
        val_loader=data_loader,
        device=device,
        config=config
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    logger.info("Running evaluation")
    
    metrics = validator.validate()
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*50)
    
    # Log results
    logger.info(f"Evaluation results: {metrics}")
    
    # Save results
    save_results(metrics, args.output_dir, f'{args.split}_results.json')
    
    # Compute per-class metrics if requested
    if args.per_class:
        print("\nComputing per-class metrics...")
        logger.info("Computing per-class metrics")
        
        # Extract features
        sketch_features, image_features, labels = validator._extract_features()
        
        # Compute similarity matrix
        similarity_matrix = model.compute_similarity_matrix(sketch_features, image_features)
        
        # Compute per-class metrics
        per_class_metrics = validator.compute_per_class_metrics(similarity_matrix, labels)
        
        # Save per-class results
        per_class_path = os.path.join(args.output_dir, f'{args.split}_per_class_results.json')
        
        # Convert to serializable format
        per_class_serializable = {}
        for label, class_metrics in per_class_metrics.items():
            per_class_serializable[str(label)] = class_metrics
        
        with open(per_class_path, 'w') as f:
            json.dump(per_class_serializable, f, indent=4)
        
        print(f"Per-class results saved to: {per_class_path}")
        
        # Print per-class summary
        print("\nPer-class mAP@200:")
        for label, class_metrics in sorted(per_class_metrics.items()):
            print(f"  Class {label}: {class_metrics['map@200']:.4f}")
    
    print("\nTesting completed successfully!")
    logger.info("Testing completed successfully")


if __name__ == '__main__':
    main()
