import torch.nn as nn
import torch.nn.functional as F
import torch
from .ASPP import ASPPModule as ASPPModule
from .Decoder import Decoder
from .MobileNetV2Backbone import MobileNetV2Backbone

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = MobileNetV2Backbone()
        self.aspp = ASPPModule(in_channels=1280, out_channels=256)
        self.decoder = Decoder(low_channels=24, out_channels=256, num_classes=num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        low, high = self.backbone(x)
        high = self.aspp(high)
        x = self.decoder(low, high)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x