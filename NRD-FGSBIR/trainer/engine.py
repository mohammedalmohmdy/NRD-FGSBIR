"""
Training Engine for Neural Retrieval Dynamics (NRD)
This module implements the main training engine coordinating training and validation.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Optional
import os
import time
from datetime import datetime

from ..models import NRDModel
from .train import Trainer
from .validate import Validator


class Engine:
    """
    Training Engine for Neural Retrieval Dynamics.
    
    This class coordinates the training and validation process, handling:
    - Model initialization
    - Optimizer and scheduler setup
    - Training loop
    - Validation loop
    - Checkpoint management
    - Early stopping
    - Logging
    
    Args:
        model (NRDModel): NRD model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (Dict): Configuration dictionary
    """
    
    def __init__(
        self,
        model: NRDModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict
    ):
        """
        Initialize the training engine.
        
        Args:
            model: NRD model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device setup
        self.device = self._setup_device()
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        
        # Scheduler setup
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config.get('hardware', {}).get('mixed_precision', True) else None
        
        # Initialize trainer and validator
        self.trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            config=config,
            scaler=self.scaler
        )
        
        self.validator = Validator(
            model=model,
            val_loader=val_loader,
            device=self.device,
            config=config
        )
        
        # Training settings
        self.num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        self.eval_interval = self.config.get('validation', {}).get('eval_interval', 1)
        
        # Early stopping
        self.early_stopping_metric = self.config.get('early_stopping', {}).get('metric', 'map@200')
        self.early_stopping_mode = self.config.get('early_stopping', {}).get('mode', 'max')
        
        # Logging
        self.log_dir = self.config.get('logging', {}).get('log_dir', './logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Resume from checkpoint if specified
        resume_path = self.config.get('checkpoint', {}).get('resume')
        if resume_path and os.path.exists(resume_path):
            self.trainer.load_checkpoint(resume_path)
    
    def _setup_device(self) -> torch.device:
        """
        Setup the device for training.
        
        Returns:
            torch.device: Device to use
        """
        device_name = self.config.get('hardware', {}).get('device', 'cuda')
        if device_name == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """
        Setup the optimizer.
        
        Returns:
            optim.Optimizer: Configured optimizer
        """
        optimizer_name = self.config.get('training', {}).get('optimizer', 'adam')
        learning_rate = self.config.get('training', {}).get('learning_rate', 0.0001)
        weight_decay = self.config.get('training', {}).get('weight_decay', 0.0001)
        momentum = self.config.get('training', {}).get('momentum', 0.9)
        
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[object]:
        """
        Setup the learning rate scheduler.
        
        Returns:
            Optional: Configured scheduler or None
        """
        scheduler_name = self.config.get('training', {}).get('scheduler', 'cosine')
        
        if scheduler_name == 'cosine':
            num_epochs = self.config.get('training', {}).get('num_epochs', 100)
            warmup_epochs = self.config.get('training', {}).get('warmup_epochs', 5)
            min_lr = self.config.get('training', {}).get('min_lr', 0.000001)
            
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=min_lr
            )
        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train(self) -> Dict[str, float]:
        """
        Main training loop.
        
        This implements the complete training process:
        1. For each epoch:
           - Train for one epoch
           - Validate (at specified intervals)
           - Update scheduler
           - Save checkpoint
           - Check early stopping
        2. Return final metrics
        
        Returns:
            Dict containing final training metrics
        """
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        best_metric = 0.0
        best_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(self.trainer.current_epoch, self.num_epochs):
            self.trainer.current_epoch = epoch
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*50}")
            
            # Train for one epoch
            train_metrics = self.trainer.train_epoch()
            
            print(f"Training Loss: {train_metrics['loss']:.4f}")
            print(f"Triplet Loss: {train_metrics['triplet_loss']:.4f}")
            print(f"Alignment Loss: {train_metrics['alignment_loss']:.4f}")
            print(f"Smoothness Loss: {train_metrics['smoothness_loss']:.4f}")
            
            # Validate at specified intervals
            if (epoch + 1) % self.eval_interval == 0:
                print("\nRunning validation...")
                val_metrics = self.validator.validate()
                
                print(f"Validation mAP@200: {val_metrics['map@200']:.4f}")
                print(f"Validation Top-1: {val_metrics['top1']:.4f}")
                print(f"Validation Top-5: {val_metrics['top5']:.4f}")
                print(f"Validation Top-10: {val_metrics['top10']:.4f}")
                
                # Update scheduler
                self.trainer.step_scheduler(val_metrics)
                
                # Check for best model
                current_metric = val_metrics[self.early_stopping_metric]
                is_best = False
                
                if self.early_stopping_mode == 'max':
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_epoch = epoch + 1
                        is_best = True
                        print(f"New best {self.early_stopping_metric}: {best_metric:.4f}")
                else:
                    if current_metric < best_metric or best_metric == 0:
                        best_metric = current_metric
                        best_epoch = epoch + 1
                        is_best = True
                        print(f"New best {self.early_stopping_metric}: {best_metric:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % self.trainer.save_interval == 0 or is_best:
                    self.trainer.save_checkpoint(epoch + 1, val_metrics, is_best)
                
                # Check early stopping
                if self.trainer.check_early_stopping(current_metric):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                # Still update scheduler if it doesn't require metrics
                if self.scheduler and not hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
                
                # Save checkpoint at intervals
                if (epoch + 1) % self.trainer.save_interval == 0:
                    self.trainer.save_checkpoint(epoch + 1, train_metrics, False)
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Training completed in {total_time / 3600:.2f} hours")
        print(f"Best {self.early_stopping_metric}: {best_metric:.4f} at epoch {best_epoch}")
        print(f"{'='*50}")
        
        # Final metrics
        final_metrics = {
            'best_metric': best_metric,
            'best_epoch': best_epoch,
            'total_time': total_time
        }
        
        return final_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Dict containing evaluation metrics
        """
        print("Running evaluation...")
        metrics = self.validator.validate()
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        return metrics
    
    def resume_training(self, checkpoint_path: str) -> None:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.trainer.load_checkpoint(checkpoint_path)
        print(f"Resumed training from epoch {self.trainer.current_epoch}")
