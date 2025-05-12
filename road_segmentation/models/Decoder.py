import torch.nn as nn
import torch.nn.functional as F
import torch
from .SKAttention import SKFusion

class Decoder(nn.Module):
    def __init__(self, low_channels, out_channels, num_classes):
        super().__init__()

        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # SKFusion after concatenation of low and high features
        self.skfusion = SKFusion(in_channels=48 + out_channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(48 + out_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, low_feat, high_feat):
        low_feat = self.reduce_low(low_feat)
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([low_feat, high_feat], dim=1)
        x = self.skfusion(x)  # Apply SKFusion attention
        x = self.fuse(x)
        return self.classifier(x)
