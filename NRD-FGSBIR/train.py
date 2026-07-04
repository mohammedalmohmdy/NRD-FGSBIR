"""
Main training script for Neural Retrieval Dynamics (NRD)
This script provides the entry point for training the NRD model.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import NRDModel
from datasets import create_sketchy_dataset, create_shoev2_dataset, create_chairv2_dataset
from trainer import Engine
from utils import load_config, set_reproducibility, setup_logger

warnings.filterwarnings('ignore')


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train Neural Retrieval Dynamics (NRD) Model')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['sketchy_extended', 'shoev2', 'chairv2'],
                        help='Dataset name (overrides config)')
    parser.add_argument('--data_root', type=str, help='Data root directory (overrides config)')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (overrides config)')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers (overrides config)')
    
    # Hardware
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use (overrides config)')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs to use (overrides config)')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision (overrides config)')
    
    # Checkpoint
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory (overrides config)')
    
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
        config['training']['batch_size'] = args.batch_size
    
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.num_workers:
        config['training']['num_workers'] = args.num_workers
    
    if args.device:
        config['hardware']['device'] = args.device
    
    if args.num_gpus is not None:
        config['hardware']['num_gpus'] = args.num_gpus
    
    if args.mixed_precision:
        config['hardware']['mixed_precision'] = True
    
    if args.resume:
        config['checkpoint']['resume'] = args.resume
    
    if args.checkpoint_dir:
        config['checkpoint']['save_dir'] = args.checkpoint_dir
    
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
    train_aug = augmentation_config.get('train', {})
    val_aug = augmentation_config.get('val', {})
    
    is_training = (split == 'train')
    aug_config = train_aug if is_training else val_aug
    
    if dataset_name == 'sketchy_extended':
        from datasets import create_sketchy_dataset
        dataset = create_sketchy_dataset(
            data_root=data_root,
            split=split,
            resize=aug_config.get('resize', 256),
            crop_size=aug_config.get('crop_size', 224),
            is_training=is_training,
            augmentation_config=aug_config
        )
    elif dataset_name == 'shoev2':
        from datasets import create_shoev2_dataset
        dataset = create_shoev2_dataset(
            data_root=data_root,
            split=split,
            resize=aug_config.get('resize', 256),
            crop_size=aug_config.get('crop_size', 224),
            is_training=is_training,
            augmentation_config=aug_config
        )
    elif dataset_name == 'chairv2':
        from datasets import create_chairv2_dataset
        dataset = create_chairv2_dataset(
            data_root=data_root,
            split=split,
            resize=aug_config.get('resize', 256),
            crop_size=aug_config.get('crop_size', 224),
            is_training=is_training,
            augmentation_config=aug_config
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
        pretrained=model_config.get('sketch_encoder', {}).get('pretrained', True),
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


def main():
    """
    Main training function.
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
        name='NRD_Training',
        log_dir=config.get('logging', {}).get('log_dir', './logs')
    )
    
    logger.info("Starting NRD training")
    logger.info(f"Configuration: {config}")
    
    # Print configuration
    print("="*50)
    print("Configuration:")
    print("="*50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*50)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_dataset(config, 'train')
    val_dataset = create_dataset(config, 'val')
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    from utils.seed import get_worker_init_fn
    worker_init_fn = get_worker_init_fn(config['reproducibility']['seed'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create training engine
    print("\nInitializing training engine...")
    engine = Engine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Start training
    print("\nStarting training...")
    logger.info("Starting training")
    
    try:
        final_metrics = engine.train()
        
        logger.info(f"Training completed successfully")
        logger.info(f"Best metric: {final_metrics['best_metric']:.4f} at epoch {final_metrics['best_epoch']}")
        logger.info(f"Total training time: {final_metrics['total_time'] / 3600:.2f} hours")
        
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
        print(f"Best {config['early_stopping']['metric']}: {final_metrics['best_metric']:.4f}")
        print(f"Best epoch: {final_metrics['best_epoch']}")
        print(f"Total time: {final_metrics['total_time'] / 3600:.2f} hours")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        logger.info("Training interrupted by user")
        
        # Save checkpoint
        print("Saving checkpoint...")
        engine.trainer.save_checkpoint(engine.trainer.current_epoch, {}, is_best=False)
        logger.info("Checkpoint saved")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        logger.error(f"Error during training: {e}")
        raise


if __name__ == '__main__':
    main()
