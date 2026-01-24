"""Training loop and utilities for segmentation models.

Provides a config-driven trainer class with:
- Training/validation loops
- Learning rate scheduling
- Early stopping
- Checkpointing
- Metric logging
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from franksign.training.losses import create_loss


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_dice: float = 0.0
    val_dice: float = 0.0
    train_iou: float = 0.0
    val_iou: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0


@dataclass
class TrainerConfig:
    """Configuration for SegmentationTrainer."""
    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    
    # Loss
    loss: str = "dice_ce"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"dice": 0.5, "ce": 0.5})
    class_weights: Optional[List[float]] = None
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_monitor: str = "val_dice"
    early_stopping_mode: str = "max"
    
    # Checkpointing
    checkpoint_dir: str = "experiments/checkpoints"
    save_best: bool = True
    save_last: bool = True
    
    # Hardware
    device: str = "auto"
    
    # Misc
    seed: int = 42
    log_every_n_steps: int = 10


class SegmentationTrainer:
    """Trainer for segmentation models.
    
    Handles the complete training loop including:
    - Forward/backward passes
    - Optimizer and scheduler steps
    - Metric computation
    - Early stopping
    - Checkpointing
    
    Example:
        >>> config = TrainerConfig(epochs=50, learning_rate=1e-3)
        >>> trainer = SegmentationTrainer(model, config)
        >>> trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
        """
        self.config = config
        self.device = self._get_device()
        self.model = model.to(self.device)
        
        # Setup training components
        self.criterion = create_loss({
            "loss": config.loss,
            "loss_weights": config.loss_weights,
            "class_weights": config.class_weights,
        })
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # State
        self.current_epoch = 0
        self.best_metric = float('-inf') if config.early_stopping_mode == 'max' else float('inf')
        self.patience_counter = 0
        self.history: List[TrainingMetrics] = []
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_device(self) -> torch.device:
        """Determine training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
            )
        elif self.config.scheduler == "step":
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif self.config.scheduler == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        return None
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[TrainingMetrics]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            List of metrics for each epoch
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training epoch
            train_loss, train_dice = self._train_epoch(train_loader)
            
            # Validation epoch
            val_loss, val_dice = self._validate_epoch(val_loader)
            
            epoch_time = time.time() - start_time
            
            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_dice=train_dice,
                val_dice=val_dice,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                epoch_time=epoch_time,
            )
            self.history.append(metrics)
            
            # Logging
            print(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()
            
            # Checkpointing
            current_metric = val_dice if self.config.early_stopping_monitor == "val_dice" else -val_loss
            is_best = self._is_better(current_metric)
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                if self.config.save_best:
                    self.save_checkpoint("best.pt")
            else:
                self.patience_counter += 1
            
            if self.config.save_last:
                self.save_checkpoint("last.pt")
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_dice += self._compute_dice(outputs, masks)
            num_batches += 1
        
        return total_loss / num_batches, total_dice / num_batches
    
    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Run one validation epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            total_dice += self._compute_dice(outputs, masks)
            num_batches += 1
        
        return total_loss / num_batches, total_dice / num_batches
    
    def _compute_dice(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor,
        smooth: float = 1e-6,
    ) -> float:
        """Compute Dice coefficient."""
        preds = torch.argmax(outputs, dim=1)
        
        # Compute per-class dice and average
        num_classes = outputs.shape[1]
        dice_sum = 0.0
        
        for c in range(1, num_classes):  # Skip background
            pred_c = (preds == c).float()
            target_c = (targets == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice_sum += (2 * intersection + smooth) / (union + smooth)
        
        return dice_sum.item() / (num_classes - 1)
    
    def _is_better(self, metric: float) -> bool:
        """Check if metric improved."""
        if self.config.early_stopping_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
