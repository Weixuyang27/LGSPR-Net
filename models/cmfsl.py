"""CMFSL modules adapted from B-Xi/TIP_2022_CMFSL."""
import torch
from torch import nn
from torch.nn import functional as F


def conv3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels))


def conv2d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels))


class SpectralPatchRelationModule(nn.Module):
    """SPRM: reweight each band from its mean, std, and center response."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(5, 3), stride=(1, 3),
                              padding=(2, 1), bias=False)

    def forward(self, x, eps=1e-5):
        n, bands, _, _ = x.shape
        flattened = x.reshape(n, bands, -1)
        center = flattened[:, :, (flattened.shape[2] - 1) // 2].unsqueeze(2)
        mean = flattened.mean(2, keepdim=True)
        std = (flattened.var(2, keepdim=True) + eps).sqrt()
        descriptor = torch.cat((mean, std, center), dim=2)
        weight = torch.sigmoid(self.conv(descriptor.unsqueeze(1)).transpose(1, 2))
        return x * (1.0 + weight.expand_as(x))


class LXConvNet(nn.Module):
    """Released low/high spatial-frequency interaction encoder."""
    feature_dim = 128

    def __init__(self):
        super().__init__()
        self.conv1 = conv3d(1, 8)
        self.conv2 = conv3d(8, 8)
        self.conv3 = conv3d(8, 16)
        # With 100 bands, spectral pooling produces 33 slices: 16*33=528.
        self.conv4 = conv2d(528, self.feature_dim)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (100, 9, 9):
            raise ValueError(f"CMFSL expects [N,100,9,9], got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        high = F.relu(self.conv1(x), inplace=True)
        low_input = F.avg_pool3d(x, (1, 2, 2), stride=(1, 2, 2))
        low = F.relu(self.conv1(low_input), inplace=True)
        high = F.avg_pool3d(high, (3, 1, 1), stride=(3, 1, 1))
        low = F.avg_pool3d(low, (3, 1, 1), stride=(3, 1, 1))

        high_high = F.relu(self.conv2(high), inplace=True)
        high_low = F.relu(self.conv2(F.avg_pool3d(
            high, (1, 2, 2), stride=(1, 2, 2))), inplace=True)
        low_high_input = F.interpolate(
            low, size=(low.shape[2], 2 * low.shape[3] + 1, 2 * low.shape[4] + 1),
            mode="trilinear", align_corners=False)
        low_high = F.relu(self.conv2(low_high_input), inplace=True)
        low_low = F.relu(self.conv2(low), inplace=True)
        high = high_high + low_high
        low = high_low + low_low

        high_to_low = F.relu(self.conv3(F.avg_pool3d(
            high, (1, 2, 2), stride=(1, 2, 2))), inplace=True)
        low_to_low = F.relu(self.conv3(low), inplace=True)
        merged = high_to_low + low_to_low
        merged = merged.reshape(merged.shape[0], 528, merged.shape[3], merged.shape[4])
        merged = F.relu(self.conv4(merged), inplace=True)
        return F.adaptive_avg_pool2d(merged, (1, 1)).flatten(1)


class CMFSLNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_sprm = SpectralPatchRelationModule()
        self.target_sprm = SpectralPatchRelationModule()
        self.encoder = LXConvNet()

    def forward(self, x, domain):
        if domain == "source":
            x = self.source_sprm(x)
        elif domain == "target":
            x = self.target_sprm(x)
        else:
            raise ValueError("domain must be 'source' or 'target'")
        return self.encoder(x)


def initialize_official(module):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def estimate_cov(features):
    matrix = features if features.shape[0] == 1 else features.t()
    centered = matrix - matrix.mean(1, keepdim=True)
    return centered @ centered.t() / max(matrix.shape[1] - 1, 1)


def build_ccm_statistics(support, labels):
    task_cov = estimate_cov(support)
    identity = torch.eye(support.shape[1], device=support.device,
                         dtype=support.dtype)
    means, matrices = [], []
    for class_id in torch.unique(labels, sorted=True):
        class_features = support[labels == class_id]
        means.append(class_features.mean(0))
        mixing = class_features.shape[0] / (class_features.shape[0] + 1.0)
        matrices.append(mixing * estimate_cov(class_features) +
                        (1.0 - mixing) * task_cov + identity)
    return torch.stack(means), torch.linalg.inv(torch.stack(matrices))


def ccm_logits(query, means, precision):
    difference = query[:, None, :] - means[None, :, :]
    return -torch.einsum("qcd,cde,qce->qc", difference, precision, difference)
