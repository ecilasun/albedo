"""
AlbedoNet model architecture for albedo/illumination decomposition.

Based on "AlbedoNet: Predicting Human-Perceived Material Reflectance"
by S. G. Narasimhan et al.

The model separates an input image into:
- Albedo: True surface color (shadow-free reflectance)
- Illumination: Lighting/shadow component
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """Multi-scale encoder for feature extraction."""

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Scale 1: 1/2 resolution
        self.block1 = nn.Sequential(
            ConvBlock(in_channels, base_channels, 7, 2, 3),
            ConvBlock(base_channels, base_channels, 3, 1, 1),
        )

        # Scale 2: 1/4 resolution
        self.block2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, 3, 2, 1),
            ConvBlock(base_channels * 2, base_channels * 2, 3, 1, 1),
        )

        # Scale 3: 1/8 resolution
        self.block3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, 3, 2, 1),
            ConvBlock(base_channels * 4, base_channels * 4, 3, 1, 1),
        )

        # Scale 4: 1/16 resolution
        self.block4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, 3, 2, 1),
            ConvBlock(base_channels * 8, base_channels * 8, 3, 1, 1),
        )

        # Scale 5: 1/32 resolution (bottleneck)
        self.block5 = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 16, 3, 2, 1),
            ConvBlock(base_channels * 16, base_channels * 16, 3, 1, 1),
        )

    def forward(self, x):
        """Return features from all scales for skip connections."""
        s1 = self.block1(x)  # 1/2
        s2 = self.block2(s1)  # 1/4
        s3 = self.block3(s2)  # 1/8
        s4 = self.block4(s3)  # 1/16
        s5 = self.block5(s4)  # 1/32
        return s1, s2, s3, s4, s5


class Decoder(nn.Module):
    """Multi-scale decoder with skip connections."""

    def __init__(self, base_channels=64):
        super().__init__()

        # Upsample from bottleneck
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2),
            ConvBlock(base_channels * 8 * 2, base_channels * 8, 3, 1, 1),
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2),
            ConvBlock(base_channels * 4 * 2, base_channels * 4, 3, 1, 1),
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2),
            ConvBlock(base_channels * 2 * 2, base_channels * 2, 3, 1, 1),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2),
            ConvBlock(base_channels * 2, base_channels, 3, 1, 1),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 2, 2),
            ConvBlock(base_channels, base_channels // 2, 3, 1, 1),
        )

    def forward(self, s1, s2, s3, s4, s5):
        """Decode with skip connections from encoder."""
        x = self.up5(s5)
        x = torch.cat([x, s4], dim=1)
        x = self.up4(x)

        x = torch.cat([x, s3], dim=1)
        x = self.up3(x)

        x = torch.cat([x, s2], dim=1)
        x = self.up2(x)

        x = torch.cat([x, s1], dim=1)
        x = self.up1(x)

        return x


class GuidanceModule(nn.Module):
    """Multi-scale guidance module for illumination estimation.

    This module produces a coarse illumination map at multiple scales,
    which is then refined by the decoder.
    """

    def __init__(self, base_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(base_channels * 16, base_channels * 8, 3, 1, 1),
            ConvBlock(base_channels * 8, base_channels * 4, 3, 1, 1),
            nn.Conv2d(base_channels * 4, 1, 1),  # Single channel illumination
        )

    def forward(self, s5):
        return self.conv(s5)


class AlbedoNet(nn.Module):
    """AlbedoNet for albedo-illumination decomposition.

    Input: RGB image (3 channels)
    Output:
        - albedo: Reflectance map (3 channels, [0, 1])
        - illumination: Lighting map (3 channels or 1 grayscale, [0, 1])
    """

    def __init__(self, out_channels=3, use_grayscale_illumination=False):
        super().__init__()

        self.use_grayscale_illumination = use_grayscale_illumination
        illum_channels = 1 if use_grayscale_illumination else out_channels

        # Encoder
        self.encoder = Encoder(in_channels=3, base_channels=64)

        # Guidance module for illumination
        self.guidance = GuidanceModule(base_channels=64)

        # Decoder
        self.decoder = Decoder(base_channels=64)

        # Output layers
        self.albedo_out = nn.Conv2d(64, out_channels, 1)
        self.illum_out = nn.Conv2d(64, illum_channels, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input RGB image, shape (B, 3, H, W), values in [0, 1]

        Returns:
            albedo: Reflectance map, shape (B, 3, H, W), values in [0, 1]
            illumination: Lighting map, shape (B, C, H, W), values in [0, 1]
        """
        # Encoder features
        s1, s2, s3, s4, s5 = self.encoder(x)

        # Coarse illumination from guidance module
        coarse_illum = torch.sigmoid(self.guidance(s5))

        # Decoder with skip connections
        decoded = self.decoder(s1, s2, s3, s4, s5)

        # Output albedo and illumination
        albedo = torch.sigmoid(self.albedo_out(decoded))
        illumination = torch.sigmoid(self.illum_out(decoded))

        # Blend coarse illumination with decoded illumination at low resolution
        # and upsample to full resolution
        if coarse_illum.size(2) != illumination.size(2):
            coarse_illum = F.interpolate(
                coarse_illum, size=illumination.size()[2:], mode='bilinear', align_corners=False
            )
        illumination = 0.5 * illumination + 0.5 * coarse_illum

        return albedo, illumination

    def load_pretrained(self, checkpoint_path):
        """Load pretrained weights from a checkpoint file.

        Args:
            checkpoint_path: Path to .pth or .pt checkpoint file
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
