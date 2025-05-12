import torch
import torch.nn as nn
import torch.nn.functional as F
from .SKAttention import SKFusion
import torchvision.models as models

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()
        
        self.atrousBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrousBlock2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrousBlock3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()        
        )

        self.atrousBlock4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.imagePooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # SKFusion after concatenation of ASPP features
        self.skfusion = SKFusion(in_channels=out_channels * 5)

        self.output = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):        
        x1 = self.atrousBlock1(x)
        x2 = self.atrousBlock2(x)
        x3 = self.atrousBlock3(x)
        x4 = self.atrousBlock4(x)

        img_pool = F.interpolate(self.imagePooling(x), size=x.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, img_pool], dim=1)
        x = self.skfusion(x)  # Apply SKFusion to ASPP features
        return self.output(x)