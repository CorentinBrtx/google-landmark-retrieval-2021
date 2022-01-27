import torch.nn as nn


class EfficientNetBackbone(nn.Module):
    def __init__(self, feature_size: int, efficientNet: nn.Module):
        super(EfficientNetBackbone, self).__init__()

        self.efficientNet = efficientNet

    def forward(self, x):
        return self.efficientNet(x)
