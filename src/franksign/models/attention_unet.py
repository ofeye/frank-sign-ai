"""Attention U-Net for Frank Sign segmentation.

Main segmentation model with attention gates on skip connections.
Supports pretrained encoders via torchvision or custom encoder.

Reference: Attention U-Net: Learning Where to Look for the Pancreas
           (Oktay et al., 2018)

Example:
    >>> from franksign.models import AttentionUNet
    >>> model = AttentionUNet(num_classes=3, encoder="resnet34", pretrained=True)
    >>> output = model(torch.randn(1, 3, 256, 256))
    >>> print(output.shape)  # torch.Size([1, 3, 256, 256])
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from franksign.models.components import (
    ConvBlock,
    DecoderBlock,
    EncoderBlock,
    SegmentationHead,
)


class AttentionUNet(nn.Module):
    """Attention U-Net for semantic segmentation.
    
    Architecture:
        - Encoder: Pretrained backbone (ResNet) or custom conv blocks
        - Bottleneck: Central feature processing
        - Decoder: Attention-gated upsampling with skip connections
        - Head: Final classification layer
    
    Args:
        num_classes: Number of output segmentation classes
        in_channels: Number of input image channels (default: 3 for RGB)
        encoder: Encoder type ('resnet18', 'resnet34', 'resnet50', or 'custom')
        pretrained: Use ImageNet pretrained weights for encoder
        encoder_channels: Channel sizes for custom encoder [64, 128, 256, 512]
        decoder_channels: Channel sizes for decoder [256, 128, 64, 32]
        use_attention: Whether to use attention gates (set False for vanilla U-Net)
    
    Example:
        >>> model = AttentionUNet(num_classes=3, encoder="resnet34", pretrained=True)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = model(x)
        >>> print(out.shape)  # [2, 3, 256, 256]
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        encoder: str = "resnet34",
        pretrained: bool = True,
        encoder_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 32],
        use_attention: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_name = encoder
        
        if encoder.startswith("resnet"):
            self.encoder, self.encoder_channels = self._build_resnet_encoder(
                encoder, pretrained, in_channels
            )
            self.custom_encoder = False
        else:
            self.encoder = self._build_custom_encoder(in_channels, encoder_channels)
            self.encoder_channels = encoder_channels
            self.custom_encoder = True
        
        # Bottleneck
        bottleneck_in = self.encoder_channels[-1]
        bottleneck_out = bottleneck_in * 2
        self.bottleneck = ConvBlock(bottleneck_in, bottleneck_out)
        
        # Decoder
        self.decoder = self._build_decoder(
            bottleneck_out, self.encoder_channels, decoder_channels, use_attention
        )
        
        # Segmentation head
        self.head = SegmentationHead(decoder_channels[-1], num_classes)
    
    def _build_resnet_encoder(
        self, 
        name: str, 
        pretrained: bool,
        in_channels: int,
    ) -> tuple[nn.Module, List[int]]:
        """Build ResNet encoder from torchvision."""
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision is required for pretrained encoders")
        
        # Get pretrained ResNet
        weights = "IMAGENET1K_V1" if pretrained else None
        
        if name == "resnet18":
            resnet = models.resnet18(weights=weights)
            channels = [64, 64, 128, 256, 512]
        elif name == "resnet34":
            resnet = models.resnet34(weights=weights)
            channels = [64, 64, 128, 256, 512]
        elif name == "resnet50":
            resnet = models.resnet50(weights=weights)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported encoder: {name}")
        
        # Modify first conv if needed
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Create encoder stages
        encoder = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  # /2
            nn.Sequential(resnet.maxpool, resnet.layer1),          # /4
            resnet.layer2,                                          # /8
            resnet.layer3,                                          # /16
            resnet.layer4,                                          # /32
        ])
        
        return encoder, channels
    
    def _build_custom_encoder(
        self,
        in_channels: int,
        channels: List[int],
    ) -> nn.ModuleList:
        """Build custom convolutional encoder."""
        encoder = nn.ModuleList()
        
        prev_ch = in_channels
        for ch in channels:
            encoder.append(EncoderBlock(prev_ch, ch, pool=True))
            prev_ch = ch
        
        return encoder
    
    def _build_decoder(
        self,
        bottleneck_channels: int,
        encoder_channels: List[int],
        decoder_channels: List[int],
        use_attention: bool,
    ) -> nn.ModuleList:
        """Build decoder with attention gates."""
        decoder = nn.ModuleList()
        
        # Reverse encoder channels for skip connections
        skip_channels = list(reversed(encoder_channels))
        
        in_ch = bottleneck_channels
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = skip_channels[i] if i < len(skip_channels) else out_ch
            decoder.append(
                DecoderBlock(in_ch, skip_ch, out_ch, use_attention=use_attention)
            )
            in_ch = out_ch
        
        return decoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits tensor of shape (B, num_classes, H, W)
        """
        input_size = x.shape[2:]
        
        # Encoder forward
        skip_connections = []
        
        if self.custom_encoder:
            for block in self.encoder:
                x, skip = block(x)
                skip_connections.append(skip)
        else:
            for stage in self.encoder:
                x = stage(x)
                skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder forward
        skip_connections = list(reversed(skip_connections))
        for i, block in enumerate(self.decoder):
            skip = skip_connections[i] if i < len(skip_connections) else None
            if skip is not None:
                x = block(x, skip)
            else:
                # No skip connection, use dummy
                x = block.up(x)
                x = block.conv(x)
        
        # Head
        x = self.head(x)
        
        # Restore original resolution
        if x.shape[2:] != input_size:
            x = nn.functional.interpolate(
                x, size=input_size, mode='bilinear', align_corners=True
            )
        
        return x
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(
    config: dict,
    num_classes: int = 3,
) -> nn.Module:
    """Create model from configuration.
    
    Args:
        config: Model configuration dict with keys:
            - architecture: 'attention_unet', 'unet', etc.
            - encoder: 'resnet34', 'resnet18', etc.
            - pretrained: bool
        num_classes: Number of segmentation classes
    
    Returns:
        Initialized model
    """
    arch = config.get("architecture", "attention_unet")
    
    if arch in ("attention_unet", "unet"):
        return AttentionUNet(
            num_classes=num_classes,
            encoder=config.get("encoder", "resnet34"),
            pretrained=config.get("pretrained", True),
            use_attention=(arch == "attention_unet"),
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


if __name__ == "__main__":
    # Quick test
    model = AttentionUNet(num_classes=3, encoder="resnet34", pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {model.get_num_parameters():,}")
