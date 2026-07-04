"""
Training module for Neural Retrieval Dynamics (NRD)
This module implements the training loop with all required features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import os
import time

from ..models import NRDModel


class Trainer:
    """
    Trainer class for Neural Retrieval Dynamics.
    
    This class implements the complete training loop with:
    - Automatic mixed precision (AMP)
    - Multi-GPU support
    - Cosine learning rate scheduler
    - Checkpoint saving
    - TensorBoard logging
    - Early stopping
    
    Args:
        model (NRDModel): NRD model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (optim.Optimizer): Optimizer
        scheduler (Optional): Learning rate scheduler
        device (torch.device): Device to train on
        config (Dict): Configuration dictionary
        scaler (Optional): GradScaler for mixed precision
    """
    
    def __init__(
        self,
        model: NRDModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
        device: torch.device = torch.device('cuda'),
        config: Optional[Dict] = None,
        scaler: Optional[GradScaler] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: NRD model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Configuration dictionary
            scaler: GradScaler for mixed precision
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.scaler = scaler or GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.early_stopping_counter = 0
        self.early_stopping_enabled = self.config.get('early_stopping', {}).get('enabled', True)
        self.early_stopping_patience = self.config.get('early_stopping', {}).get('patience', 10)
        
        # Checkpoint settings
        self.checkpoint_dir = self.config.get('checkpoint', {}).get('save_dir', './checkpoints')
        self.save_interval = self.config.get('checkpoint', {}).get('save_interval', 5)
        self.max_keep = self.config.get('checkpoint', {}).get('max_keep', 5)
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Logging
        self.log_dir = self.config.get('logging', {}).get('log_dir', './logs')
        self.log_interval = self.config.get('logging', {}).get('log_interval', 100)
        self.print_interval = self.config.get('logging', {}).get('print_interval', 10)
        
        # Mixed precision
        self.use_amp = self.config.get('hardware', {}).get('mixed_precision', True)
        
        # Multi-GPU
        self.num_gpus = self.config.get('hardware', {}).get('num_gpus', 1)
        if self.num_gpus > 1 and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        
        # Move model to device
        self.model.to(self.device)
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        This implements the complete training loop:
        1. Forward pass
        2. Trajectory evolution
        3. Similarity accumulation
        4. Loss computation
        5. Backward pass
        6. Optimizer step
        
        Returns:
            Dict containing training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_alignment_loss = 0.0
        total_smoothness_loss = 0.0
        total_l2_loss = 0.0
        
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            sketch = batch['sketch'].to(self.device)
            positive_image = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Sample negative images (hard negative mining)
            negative_image = self._sample_negative(batch_idx, labels)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with automatic mixed precision
            if self.use_amp:
                with autocast():
                    loss, loss_dict = self.model.compute_loss(
                        sketch=sketch,
                        positive_image=positive_image,
                        negative_image=negative_image,
                        labels=labels
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, loss_dict = self.model.compute_loss(
                    sketch=sketch,
                    positive_image=positive_image,
                    negative_image=negative_image,
                    labels=labels
                )
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss']
            total_triplet_loss += loss_dict['triplet_loss']
            total_alignment_loss += loss_dict['alignment_loss']
            total_smoothness_loss += loss_dict['smoothness_loss']
            total_l2_loss += loss_dict['l2_loss']
            
            # Update progress bar
            if batch_idx % self.print_interval == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'triplet': f"{loss_dict['triplet_loss']:.4f}",
                    'align': f"{loss_dict['alignment_loss']:.4f}"
                })
            
            # Log to tensorboard
            if batch_idx % self.log_interval == 0:
                self._log_metrics(loss_dict, batch_idx)
            
            self.global_step += 1
        
        # Compute average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'triplet_loss': total_triplet_loss / num_batches,
            'alignment_loss': total_alignment_loss / num_batches,
            'smoothness_loss': total_smoothness_loss / num_batches,
            'l2_loss': total_l2_loss / num_batches
        }
        
        return avg_metrics
    
    def _sample_negative(self, batch_idx: int, labels: torch.Tensor) -> torch.Tensor:
        """
        Sample negative images for triplet loss.
        
        Args:
            batch_idx: Current batch index
            labels: Labels for current batch
            
        Returns:
            torch.Tensor: Negative images of shape (B, C, H, W)
        """
        batch_size = labels.size(0)
        
        # Get all images from the dataset
        all_images = []
        all_labels = []
        
        for batch in self.train_loader:
            all_images.append(batch['image'])
            all_labels.append(batch['label'])
        
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Sample negatives (different class)
        negative_indices = []
        for i in range(batch_size):
            label = labels[i].item()
            # Find indices with different labels
            negative_mask = all_labels != label
            negative_idx = torch.where(negative_mask)[0]
            
            # Randomly sample one negative
            if len(negative_idx) > 0:
                sampled_idx = negative_idx[torch.randint(0, len(negative_idx), (1,))].item()
            else:
                # Fallback to random sample
                sampled_idx = torch.randint(0, len(all_images), (1,)).item()
            
            negative_indices.append(sampled_idx)
        
        negative_images = all_images[negative_indices].to(self.device)
        
        return negative_images
    
    def _log_metrics(self, loss_dict: Dict, step: int) -> None:
        """
        Log metrics to TensorBoard.
        
        Args:
            loss_dict: Dictionary of loss values
            step: Global step
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            if not hasattr(self, 'writer'):
                self.writer = SummaryWriter(self.log_dir)
            
            for key, value in loss_dict.items():
                self.writer.add_scalar(f'train/{key}', value, step)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, step)
            
        except ImportError:
            pass
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
        
        # Remove old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the most recent ones."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        while len(checkpoints) > self.max_keep:
            old_checkpoint = checkpoints.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['metrics'].get('best_metric', 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def step_scheduler(self, metrics: Dict) -> None:
        """
        Step the learning rate scheduler.
        
        Args:
            metrics: Dictionary of validation metrics
        """
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                # Check if scheduler requires metric
                if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                    metric_value = metrics.get('map@200', 0.0)
                    self.scheduler.step(metric_value)
                else:
                    self.scheduler.step()
    
    def check_early_stopping(self, metric: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            metric: Current metric value
            
        Returns:
            bool: Whether to stop training
        """
        if not self.early_stopping_enabled:
            return False
        
        if metric > self.best_metric:
            self.best_metric = metric
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs")
                return True
        
        return False
