import torch.nn as nn
import torch


class MobileNetV2Backbone (nn.Module):
    def __init__(self,low_level_idx=2,high_level_idx=4):
        super().__init__()
        base_model=models.mobilenet_v2(pretrained=True)
        self.low_level_features=base_model.features[:4]
        self.high_level_features=base_model.features[4:]


    def forward (self,x):
        low_feat=self.low_level_features(x)
        high_feat=self.high_level_features(low_feat)

        return low_feat,high_feat
    


# # Print output shapes
# print("Low-level feature shape:", low_feat.shape)
# print("High-level feature shape:", high_feat.shape)
