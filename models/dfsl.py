"""DFSL feature extractor from gokling1219/DFSL (official TensorFlow code)."""
import torch
from torch import nn
from torch.nn import functional as F


def conv3d(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, 3, padding=1)


class ResidualStage(nn.Module):
    """Four convolutions with the first activation used as the residual."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels)
        self.conv2 = conv3d(out_channels, out_channels)
        self.conv3 = conv3d(out_channels, out_channels)
        self.conv4 = conv3d(out_channels, out_channels)

    def forward(self, x):
        residual = F.relu(self.conv1(x), inplace=False)
        x = F.relu(self.conv2(residual), inplace=False)
        x = F.relu(self.conv3(x), inplace=False)
        return F.relu(self.conv4(x), inplace=False) + residual


class DFSLFeatureEncoder(nn.Module):
    """Official 9x9x100 residual 3-D CNN producing a 160-D embedding."""
    feature_dim = 160

    def __init__(self):
        super().__init__()
        self.stage1 = ResidualStage(1, 8)
        self.pool1 = nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2), padding=(0, 1, 1))
        self.stage2 = ResidualStage(8, 16)
        self.pool2 = nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.final = nn.Conv3d(16, 32, 3)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (100, 9, 9):
            raise ValueError(f"DFSL expects [N,100,9,9], got {tuple(x.shape)}")
        x = self.pool1(self.stage1(x.unsqueeze(1)))
        x = self.pool2(self.stage2(x))
        x = self.final(x).flatten(1)
        if x.shape[1] != self.feature_dim:
            raise RuntimeError(f"Unexpected DFSL embedding shape {tuple(x.shape)}")
        return x


def initialize_official(module):
    """TensorFlow release: truncated normal std=0.1 and bias=0.1."""
    if isinstance(module, nn.Conv3d):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.1,
                             a=-0.2, b=0.2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)


def prototype_logits(query, support, n_way, n_shot):
    prototypes = support.reshape(n_way, n_shot, -1).mean(1)
    # The release uses mean squared Euclidean distance, not a summed distance.
    return -((query[:, None, :] - prototypes[None, :, :]) ** 2).mean(2)
