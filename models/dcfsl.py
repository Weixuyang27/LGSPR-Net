"""DCFSL modules adapted from Li-ZK/DCFSL-2021 (official PyTorch code)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Mapping(nn.Module):
    def __init__(self, in_channels, out_channels=100):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


def conv3x3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
    )


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv3x3x3(in_channels, out_channels)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.conv3 = conv3x3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        return F.relu(x1 + self.conv3(x2), inplace=True)


class DRes3DCNN(nn.Module):
    """The official two-block 3-D residual embedding network."""
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock3D(1, 8)
        self.pool1 = nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2), padding=(0, 1, 1))
        self.block2 = ResidualBlock3D(8, 16)
        self.pool2 = nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(16, 32, 3, bias=False)
        self.feature_dim = 160

    def forward(self, x):
        x = self.pool1(self.block1(x.unsqueeze(1)))
        x = self.pool2(self.block2(x))
        return self.conv(x).flatten(1)


class DCFSLNetwork(nn.Module):
    def __init__(self, source_bands, target_bands, condition_classes, mapping_dim=100):
        super().__init__()
        if mapping_dim != 100:
            raise ValueError("The official DCFSL 3-D encoder requires mapping_dim=100")
        self.source_mapping = Mapping(source_bands, mapping_dim)
        self.target_mapping = Mapping(target_bands, mapping_dim)
        self.encoder = DRes3DCNN()
        self.classifier = nn.Linear(self.encoder.feature_dim, condition_classes)

    @property
    def feature_dim(self):
        return self.encoder.feature_dim

    def forward(self, x, domain):
        if domain == "source":
            x = self.source_mapping(x)
        elif domain == "target":
            x = self.target_mapping(x)
        else:
            raise ValueError("domain must be 'source' or 'target'")
        feature = self.encoder(x)
        return feature, self.classifier(feature)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


def grl_coefficient(step, max_iter=10000, alpha=10.0):
    value = 2.0 / (1.0 + math.exp(-alpha * float(step) / float(max_iter))) - 1.0
    return float(value)


class DomainClassifier(nn.Module):
    def __init__(self, in_features=1024):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.extend((nn.Linear(in_features, in_features), nn.ReLU(), nn.Dropout(0.5)))
        self.layers = nn.Sequential(*layers)
        self.domain = nn.Linear(in_features, 1)

    def forward(self, x, coefficient):
        x = _GradientReverse.apply(x, coefficient)
        return self.domain(self.layers(x))


class RandomizedMultilinearMap(nn.Module):
    """Official randomized feature x class-probability conditioning map."""
    def __init__(self, feature_dim, class_dim, output_dim=1024):
        super().__init__()
        self.output_dim = int(output_dim)
        self.register_buffer("feature_matrix", torch.randn(feature_dim, output_dim))
        self.register_buffer("class_matrix", torch.randn(class_dim, output_dim))

    def forward(self, feature, probability):
        scale = math.pow(float(self.output_dim), 0.5)
        return (feature @ self.feature_matrix) * (probability @ self.class_matrix) / scale


def initialize_official(module):
    """Initialization used by the released DCFSL implementation."""
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.ones_(module.bias)


def euclidean_logits(query, prototypes):
    return -torch.sum((query[:, None, :] - prototypes[None, :, :]) ** 2, dim=-1)
