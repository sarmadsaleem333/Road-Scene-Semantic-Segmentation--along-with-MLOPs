import torch.nn as nn
import torch.nn.functional as F

class SKFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SKFusion, self).__init__()
        self.in_channels = in_channels


        self.conv_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels * 2)  

    def forward(self, x):
        u1 = self.conv_3x3(x) 
        u2 = self.conv_5x5(x) 
        u = u1 + u2

        s = self.global_pool(u).view(-1, self.in_channels)

        z = F.relu(self.fc1(s))
        a_b = self.fc2(z)  

        a_b = a_b.view(-1, 2, self.in_channels) 
        a_b = F.softmax(a_b, dim=1)  

        a, b = a_b[:, 0, :], a_b[:, 1, :]  
        a = a.view(-1, self.in_channels, 1, 1)  
        b = b.view(-1, self.in_channels, 1, 1)

        v = a * u1 + b * u2  
        return v
