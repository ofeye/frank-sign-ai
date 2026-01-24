"""Reusable building blocks for segmentation models.

This module provides PyTorch modules for building U-Net variants:
- ConvBlock: Double convolution with batch normalization
- AttentionGate: Attention mechanism for skip connections
- UpConvBlock: Upsampling with convolution
- SegmentationHead: Final output layer
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization.
    
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dropout: Dropout probability (0 to disable)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class AttentionGate(nn.Module):
    """Attention gate for skip connections.
    
    Learns to focus on relevant spatial regions by computing
    attention coefficients between encoder and decoder features.
    
    Reference: Attention U-Net (Oktay et al., 2018)
    
    Args:
        gate_channels: Channels from gating signal (decoder)
        skip_channels: Channels from skip connection (encoder)
        inter_channels: Intermediate channels for attention computation
    """
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        
        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        
        # Attention coefficient computation
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self, 
        gate: torch.Tensor, 
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention to skip connection.
        
        Args:
            gate: Gating signal from decoder (lower resolution)
            skip: Skip connection from encoder (higher resolution)
        
        Returns:
            Attention-weighted skip connection
        """
        # Upsample gate to match skip resolution
        gate_up = F.interpolate(
            gate, size=skip.shape[2:], mode='bilinear', align_corners=True
        )
        
        # Compute attention weights
        g = self.W_g(gate_up)
        x = self.W_x(skip)
        attention = self.psi(self.relu(g + x))
        
        # Apply attention
        return skip * attention


class UpConvBlock(nn.Module):
    """Upsampling block with convolution.
    
    Options:
    - Transposed convolution (learnable upsampling)
    - Bilinear interpolation + convolution
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        mode: 'transpose' or 'bilinear'
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "bilinear",
    ):
        super().__init__()
        
        if mode == "transpose":
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class DecoderBlock(nn.Module):
    """Single decoder block with upsampling, skip connection, and convolution.
    
    Args:
        in_channels: Input channels from lower level
        skip_channels: Channels from skip connection
        out_channels: Output channels
        use_attention: Whether to use attention gate
        upsample_mode: 'transpose' or 'bilinear'
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = True,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        
        self.up = UpConvBlock(in_channels, in_channels // 2, mode=upsample_mode)
        
        self.attention = (
            AttentionGate(in_channels // 2, skip_channels)
            if use_attention
            else None
        )
        
        # After concatenation: skip_channels + in_channels // 2
        self.conv = ConvBlock(skip_channels + in_channels // 2, out_channels)
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        
        # Apply attention to skip connection
        if self.attention is not None:
            skip = self.attention(x, skip)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate and convolve
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class SegmentationHead(nn.Module):
    """Final segmentation output layer.
    
    Args:
        in_channels: Input channels
        num_classes: Number of output classes
        dropout: Dropout before final conv
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Single encoder block with convolution and pooling.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        pool: Whether to apply max pooling
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
    ):
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (pooled_output, skip_connection)
        """
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features
