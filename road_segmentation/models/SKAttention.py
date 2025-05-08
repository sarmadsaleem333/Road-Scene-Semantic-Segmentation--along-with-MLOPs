import torch.nn as nn
import torch.nn.functional as F

class SKFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):

        super(SKFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
       
        self.conv3=nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
    
        )
        self.conv5=nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
    
        )

        self.fc1=nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1)
        self.fc2=nn.Conv2d(in_channels//reduction, 2* out_channels, kernel_size=1)

        self.global_pool=nn.AdaptiveAvgPool2d(1)
