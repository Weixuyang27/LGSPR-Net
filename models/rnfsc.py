"""RN-FSC modules adapted from the authors' released PyTorch code.

Upstream: gokling1219/RN-FSC, commit
e971e999a003be67f82d5165bc84f5845d3f399e.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RNFeatureEncoder(nn.Module):
    """The four-layer 3-D CNN used by the released target fine-tuning/test code."""
    def __init__(self):
        super().__init__()
        channels = (1, 16, 32, 64, 128)
        layers = []
        for index in range(3):
            layers.append(nn.Sequential(
                nn.Conv3d(channels[index], channels[index + 1], 3, padding=1),
                nn.BatchNorm3d(channels[index + 1]),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(4, 2, 2), padding=1),
            ))
        layers.append(nn.Sequential(
            nn.Conv3d(channels[3], channels[4], 3, padding=1),
            nn.BatchNorm3d(channels[4]),
            nn.ReLU(),
        ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (100, 28, 28):
            raise ValueError(f"RN-FSC expects [N,100,28,28], got {tuple(x.shape)}")
        return self.layers(x.unsqueeze(1))


class RelationNetwork(nn.Module):
    """Score every query/prototype pair with the released relation head."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 256, 3), nn.BatchNorm2d(256), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, 3), nn.BatchNorm2d(128), nn.ReLU())
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, pairs):
        value = self.layer2(self.layer1(pairs)).flatten(1)
        return torch.sigmoid(self.fc2(self.dropout(F.relu(self.fc1(value)))))


def spatial_relation_features(features):
    """Fold 3-D feature channel/depth axes as done in the released scripts."""
    if features.shape[1:] != (128, 2, 5, 5):
        raise ValueError(f"Unexpected RN-FSC feature shape: {tuple(features.shape)}")
    return features.reshape(features.size(0), 256, 5, 5)


def relation_scores(relation_network, support_features, query_features,
                    n_way, n_shot):
    prototypes = support_features.reshape(
        n_way, n_shot, *support_features.shape[1:]).mean(1)
    prototypes = spatial_relation_features(prototypes)
    queries = spatial_relation_features(query_features)
    query_count = queries.size(0)
    prototype_grid = prototypes.unsqueeze(0).expand(query_count, -1, -1, -1, -1)
    query_grid = queries.unsqueeze(1).expand(-1, n_way, -1, -1, -1)
    pairs = torch.cat((prototype_grid, query_grid), dim=2).reshape(
        query_count * n_way, 512, 5, 5)
    return relation_network(pairs).reshape(query_count, n_way)


def relation_mse(scores, labels, n_way):
    targets = torch.zeros(scores.size(0), n_way, device=scores.device)
    targets.scatter_(1, labels.reshape(-1, 1), 1.0)
    return F.mse_loss(scores, targets)


def initialize_official(module):
    """Initialization in the released RN-FSC scripts."""
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        kernel = module.kernel_size
        n_value = kernel[0] * kernel[1] * module.out_channels
        nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / n_value))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.01)
        nn.init.ones_(module.bias)
