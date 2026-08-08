"""Model architecture for the ssy1245/Intrinsic_Decomposition checkpoint."""

import torch
import torch.nn as nn
from torchvision import models


class DecoderBlock(nn.Module):
    """Upsample, merge one encoder skip, and refine the features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        features = self.upsample(features)
        return self.conv(torch.cat([features, skip], dim=1))


class IntrinsicDecompositionNet(nn.Module):
    """ResNet-18 encoder with albedo and shading decoder heads."""

    def __init__(self, color_shading: bool = False):
        super().__init__()
        self.color_shading = color_shading
        encoder = models.resnet18(weights=None)

        self.enc0 = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )
        self.pool = encoder.maxpool
        self.enc1 = encoder.layer1
        self.enc2 = encoder.layer2
        self.enc3 = encoder.layer3
        self.enc4 = encoder.layer4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )
        self.e4_proj = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.GroupNorm(8, 256),
        )

        self.albedo_dec4 = DecoderBlock(256, 256, 128)
        self.albedo_dec3 = DecoderBlock(128, 128, 64)
        self.albedo_dec2 = DecoderBlock(64, 64, 64)
        self.albedo_dec1 = DecoderBlock(64, 64, 32)
        self.albedo_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        self.shading_dec4 = DecoderBlock(256, 256, 128)
        self.shading_dec3 = DecoderBlock(128, 128, 64)
        self.shading_dec2 = DecoderBlock(64, 64, 64)
        self.shading_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )
        shading_channels = 3 if color_shading else 1
        self.shading_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, shading_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_0 = self.enc0(image)
        encoder_1 = self.enc1(self.pool(encoder_0))
        encoder_2 = self.enc2(encoder_1)
        encoder_3 = self.enc3(encoder_2)
        encoder_4 = self.enc4(encoder_3)
        bottleneck = self.bottleneck(encoder_4) + self.e4_proj(encoder_4)
        return bottleneck, encoder_3, encoder_2, encoder_1, encoder_0

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bottleneck, encoder_3, encoder_2, encoder_1, encoder_0 = self.encode(image)

        albedo = self.albedo_dec4(bottleneck, encoder_3)
        albedo = self.albedo_dec3(albedo, encoder_2)
        albedo = self.albedo_dec2(albedo, encoder_1)
        albedo = self.albedo_dec1(albedo, encoder_0)
        albedo = self.albedo_head(albedo)

        shading = self.shading_dec4(bottleneck, encoder_3)
        shading = self.shading_dec3(shading, encoder_2)
        shading = self.shading_dec2(shading, encoder_1)
        shading = self.shading_up1(shading)
        shading = self.shading_head(shading)

        return albedo, shading