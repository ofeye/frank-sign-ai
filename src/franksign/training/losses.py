"""Loss functions for segmentation training.

Provides combined loss functions optimized for medical image segmentation:
- Dice Loss: Overlap-based, handles class imbalance well
- Focal Loss: Focuses on hard examples
- Combined losses: DiceCE, DiceFocal
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    Args:
        smooth: Smoothing factor to avoid division by zero
        class_weights: Optional per-class weights
        ignore_index: Class index to ignore (e.g., 255 for boundary)
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            logits: Model output (B, C, H, W)
            targets: Ground truth (B, H, W) with class indices
        
        Returns:
            Scalar loss value
        """
        num_classes = logits.shape[1]
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets.clamp(0, num_classes - 1).long(), 
            num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Compute Dice per class
        dims = (0, 2, 3)  # Batch, Height, Width
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)
        
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights.to(dice_per_class.device)
            dice_per_class = dice_per_class * weights
            dice = dice_per_class.sum() / weights.sum()
        else:
            dice = dice_per_class.mean()
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.
    
    FL(p) = -α(1-p)^γ log(p)
    
    Reference: Focal Loss for Dense Object Detection (Lin et al., 2017)
    
    Args:
        alpha: Balancing factor for each class
        gamma: Focusing parameter (0 = CE, higher = more focus on hard examples)
        ignore_index: Class index to ignore
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Focal loss.
        
        Args:
            logits: Model output (B, C, H, W)
            targets: Ground truth (B, H, W)
        
        Returns:
            Scalar loss value
        """
        ce_loss = F.cross_entropy(
            logits, targets.long(), 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        probs = F.softmax(logits, dim=1)
        
        # Gather probabilities for target classes
        targets_clamped = targets.clamp(0, logits.shape[1] - 1).long()
        pt = probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss.
    
    Combines the benefits of both:
    - Dice: Good for imbalanced segmentation
    - CE: Stable gradients, per-pixel supervision
    
    Args:
        dice_weight: Weight for Dice loss component
        ce_weight: Weight for CE loss component
        class_weights: Optional per-class weights for CE
        smooth: Smoothing factor for Dice
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets.long())
        
        return self.dice_weight * dice + self.ce_weight * ce


class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal loss.
    
    Best for highly imbalanced datasets with hard examples.
    
    Args:
        dice_weight: Weight for Dice component
        focal_weight: Weight for Focal component
        gamma: Focal loss focusing parameter
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        gamma: float = 2.0,
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=gamma)
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        return self.dice_weight * dice + self.focal_weight * focal


def create_loss(config: dict) -> nn.Module:
    """Create loss function from configuration.
    
    Args:
        config: Training config with 'loss' key
    
    Returns:
        Configured loss function
    """
    loss_name = config.get("loss", "dice_ce")
    loss_weights = config.get("loss_weights", {"dice": 0.5, "ce": 0.5})
    class_weights = config.get("class_weights")
    
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    if loss_name == "dice":
        return DiceLoss()
    elif loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "focal":
        return FocalLoss()
    elif loss_name == "dice_ce":
        return DiceCELoss(
            dice_weight=loss_weights.get("dice", 0.5),
            ce_weight=loss_weights.get("ce", 0.5),
            class_weights=class_weights,
        )
    elif loss_name == "dice_focal":
        return DiceFocalLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
