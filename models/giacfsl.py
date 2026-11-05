"""Gia-CFSL modules adapted from the authors' released PyTorch implementation.

Upstream: YuxiangZhang-BIT/IEEE_TNNLS_Gia-CFSL, commit
c0479e5b68c896479b7bb96bd6631d28ab6e833c.
"""
from copy import deepcopy

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


def _conv3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
    )


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _conv3d(in_channels, out_channels)
        self.conv2 = _conv3d(out_channels, out_channels)
        self.conv3 = _conv3d(out_channels, out_channels)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        return F.relu(x1 + self.conv3(x2), inplace=True)


class GiaFeatureEncoder(nn.Module):
    """Released two-stage 3-D residual encoder with two 128-D outputs."""
    def __init__(self, patch_size=9, embedding_dim=128):
        super().__init__()
        self.block1 = ResidualBlock3D(1, 8)
        self.pool1 = nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2), padding=(0, 1, 1))
        self.block2 = ResidualBlock3D(8, 16)
        self.pool2 = nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(16, 32, 3, bias=False)
        with torch.no_grad():
            x = torch.zeros(1, 1, 100, patch_size, patch_size)
            x = self.pool1(self.block1(x))
            x = self.pool2(self.block2(x))
            middle_dim = x.numel()
            final_dim = self.conv(x).numel()
        self.middle = nn.Sequential(nn.Linear(middle_dim, embedding_dim), nn.BatchNorm1d(embedding_dim))
        self.final = nn.Sequential(nn.Linear(final_dim, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def forward(self, x):
        x = self.pool1(self.block1(x.unsqueeze(1)))
        x = self.pool2(self.block2(x))
        middle = self.middle(x.flatten(1))
        final = self.final(self.conv(x).flatten(1))
        return [middle, final]


class GiaNetwork(nn.Module):
    def __init__(self, source_bands, target_bands, mapping_dim=100, embedding_dim=128):
        super().__init__()
        if mapping_dim != 100:
            raise ValueError("The released Gia-CFSL encoder requires mapping_dim=100")
        self.source_mapping = Mapping(source_bands, mapping_dim)
        self.target_mapping = Mapping(target_bands, mapping_dim)
        self.encoder = GiaFeatureEncoder(9, embedding_dim)

    def forward(self, x, domain):
        if domain == "source":
            x = self.source_mapping(x)
        elif domain == "target":
            x = self.target_mapping(x)
        else:
            raise ValueError("domain must be 'source' or 'target'")
        return self.encoder(x)


class PointSimilarity(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(channels, channels * 2, 1, bias=False),
                  nn.BatchNorm2d(channels * 2), nn.LeakyReLU()]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        layers += [nn.Conv2d(channels * 2, channels, 1, bias=False),
                   nn.BatchNorm2d(channels), nn.LeakyReLU()]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(channels, 1, 1))
        self.transform = nn.Sequential(*layers)

    def forward(self, nodes, old_edges, metric):
        left, right = nodes.unsqueeze(2), nodes.unsqueeze(1)
        difference = (left - right).square() if metric == "l2" else (left - right).abs()
        learned = torch.sigmoid(self.transform(difference.permute(0, 3, 1, 2))).squeeze(1)
        eye = torch.eye(nodes.size(1), device=nodes.device).unsqueeze(0)
        masked = old_edges * (1.0 - eye)
        old_sum = masked.sum(-1, keepdim=True)
        edges = F.normalize(learned * masked, p=1, dim=-1) * old_sum
        edges = edges + eye + 1e-6
        edges = edges / edges.sum(2, keepdim=True)
        return edges, -difference.sum(3)


class PointToDistribution(nn.Module):
    def __init__(self, num_support):
        super().__init__()
        self.num_support = num_support
        self.transform = nn.Sequential(nn.Linear(2 * num_support, num_support), nn.LeakyReLU())

    def forward(self, point_edge, distribution_node):
        value = torch.cat((point_edge[:, :, :self.num_support], distribution_node), dim=2)
        return self.transform(value)


class DistributionSimilarity(nn.Module):
    def __init__(self, num_support, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(num_support, num_support * 2, 1, bias=False),
                  nn.BatchNorm2d(num_support * 2), nn.LeakyReLU()]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        layers += [nn.Conv2d(num_support * 2, num_support, 1, bias=False),
                   nn.BatchNorm2d(num_support), nn.LeakyReLU()]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(num_support, 1, 1))
        self.transform = nn.Sequential(*layers)

    def forward(self, nodes, old_edges, metric):
        left, right = nodes.unsqueeze(2), nodes.unsqueeze(1)
        difference = (left - right).square() if metric == "l2" else (left - right).abs()
        learned = torch.sigmoid(self.transform(difference.permute(0, 3, 1, 2))).squeeze(1)
        eye = torch.eye(nodes.size(1), device=nodes.device).unsqueeze(0)
        masked = old_edges * (1.0 - eye)
        old_sum = masked.sum(-1, keepdim=True)
        edges = F.normalize(learned * masked, p=1, dim=-1) * old_sum
        edges = edges + eye + 1e-6
        return edges / edges.sum(2, keepdim=True)


class DistributionToPoint(nn.Module):
    def __init__(self, embedding_dim, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(embedding_dim * 2, embedding_dim * 2, 1, bias=False),
                  nn.BatchNorm2d(embedding_dim * 2), nn.LeakyReLU(),
                  nn.Conv2d(embedding_dim * 2, embedding_dim, 1, bias=False),
                  nn.BatchNorm2d(embedding_dim), nn.LeakyReLU()]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.transform = nn.Sequential(*layers)

    def forward(self, edges, nodes):
        eye = torch.eye(nodes.size(1), device=nodes.device).unsqueeze(0)
        attention = F.normalize(edges * (1.0 - eye), p=1, dim=-1)
        aggregated = torch.bmm(attention, nodes)
        value = torch.cat((nodes, aggregated), dim=-1).transpose(1, 2).unsqueeze(-1)
        return self.transform(value).transpose(1, 2).squeeze(-1)


class IDEBlock(nn.Module):
    """Intra-domain dual point/distribution graph aggregation block."""
    def __init__(self, generations, dropout, num_support, point_metric="l2",
                 distribution_metric="l2", embedding_dim=128):
        super().__init__()
        self.generations = generations
        self.point_metric = point_metric
        self.distribution_metric = distribution_metric
        self.initial = PointSimilarity(embedding_dim, dropout)
        self.point_similarity = nn.ModuleList()
        self.point_to_distribution = nn.ModuleList()
        self.distribution_similarity = nn.ModuleList()
        self.distribution_to_point = nn.ModuleList()
        for generation in range(generations):
            rate = dropout if generation < generations - 1 else 0.0
            self.point_similarity.append(PointSimilarity(embedding_dim, rate))
            self.point_to_distribution.append(PointToDistribution(num_support))
            self.distribution_similarity.append(DistributionSimilarity(num_support, rate))
            self.distribution_to_point.append(DistributionToPoint(embedding_dim, rate))

    def forward(self, middle, point, distribution_node, distribution_edge, point_edge):
        point_edge, _ = self.initial(middle, point_edge, self.point_metric)
        point_nodes, distribution_nodes = [], []
        for index in range(self.generations):
            point_edge, _ = self.point_similarity[index](point, point_edge, self.point_metric)
            distribution_node = self.point_to_distribution[index](point_edge, distribution_node)
            distribution_edge = self.distribution_similarity[index](
                distribution_node, distribution_edge, self.distribution_metric)
            point = self.distribution_to_point[index](distribution_edge, point)
            point_nodes.append(point)
            distribution_nodes.append(distribution_node)
        return point_nodes, distribution_nodes


def _mlp(channels):
    layers = []
    for index in range(1, len(channels)):
        layers.append(nn.Conv1d(channels[index - 1], channels[index], 1))
        if index < len(channels) - 1:
            layers += [nn.BatchNorm1d(channels[index]), nn.ReLU()]
    return nn.Sequential(*layers)


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, heads=4):
        super().__init__()
        if feature_dim % heads:
            raise ValueError("feature_dim must be divisible by heads")
        self.heads = heads
        self.dim = feature_dim // heads
        merge = nn.Conv1d(feature_dim, feature_dim, 1)
        self.projections = nn.ModuleList([deepcopy(merge) for _ in range(3)])
        self.merge = merge

    def forward(self, query, key, value):
        batch = query.size(0)
        query, key, value = [layer(x).view(batch, self.dim, self.heads, -1)
                             for layer, x in zip(self.projections, (query, key, value))]
        scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / self.dim ** 0.5
        probability = F.softmax(scores, dim=-1)
        output = torch.einsum("bhnm,bdhm->bdhn", probability, value)
        return self.merge(output.contiguous().view(batch, self.dim * self.heads, -1))


def _detached_standardize(nodes):
    """Match sklearn.preprocessing.scale used by the released CSA block."""
    value = nodes.detach()
    mean = value.mean(dim=1, keepdim=True)
    variance = (value - mean).square().mean(dim=1, keepdim=True)
    return (value - mean) / variance.sqrt().clamp_min(1e-12)


class CSABlock(nn.Module):
    """Cross-domain similarity aggregation and released linear-MMD objective."""
    def __init__(self, num_support, embedding_dim=128, generations=1):
        super().__init__()
        self.attention = nn.ModuleList(
            [MultiHeadAttention(embedding_dim, 4) for _ in range(generations)])
        self.mlp = _mlp([embedding_dim * 2, embedding_dim * 2, embedding_dim])
        self.distribution_mlp = _mlp([num_support, embedding_dim, embedding_dim])
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.distribution_mlp[-1].bias)

    def forward(self, point_source, point_target, distribution_source, distribution_target):
        for index in range(len(point_source)):
            point_source[index] = _detached_standardize(point_source[index]).transpose(1, 2)
            point_target[index] = _detached_standardize(point_target[index]).transpose(1, 2)
            distribution_source[index] = self.distribution_mlp(
                _detached_standardize(distribution_source[index]).transpose(1, 2))
            distribution_target[index] = self.distribution_mlp(
                _detached_standardize(distribution_target[index]).transpose(1, 2))
        for index, layer in enumerate(self.attention):
            src_other, tar_other = point_target[index], point_source[index]
            src_dis_other, tar_dis_other = distribution_target[index], distribution_source[index]
            src_delta = layer(point_source[index], src_other, src_other)
            tar_delta = layer(point_target[index], tar_other, tar_other)
            src_point = src_delta * src_other
            tar_point = tar_delta * tar_other
            src_delta = layer(distribution_source[index], src_dis_other, src_point)
            tar_delta = layer(distribution_target[index], tar_dis_other, tar_point)
            point_source[0] = point_source[0] + self.mlp(torch.cat((point_source[index], src_delta), dim=1))
            point_target[0] = point_target[0] + self.mlp(torch.cat((point_target[index], tar_delta), dim=1))
        source = point_source[0].squeeze(0).transpose(0, 1)
        target = point_target[0].squeeze(0).transpose(0, 1)
        delta = source.mean(0) - target.mean(0)
        return delta.dot(delta)


def initialize_graph(labels, num_support, device):
    """Released support-supervised/query-uniform dual-graph initialization."""
    labels = labels.unsqueeze(0)
    same = (labels.unsqueeze(2) == labels.unsqueeze(1)).float()
    total = labels.size(1)
    support_same = same[:, :num_support, :num_support]
    query_distribution = torch.full(
        (1, total - num_support, num_support), 1.0 / num_support, device=device)
    distribution_node = torch.cat((support_same, query_distribution), dim=1)
    point_edge = same.clone()
    point_edge[:, num_support:, :num_support] = 1.0 / num_support
    point_edge[:, :num_support, num_support:] = 1.0 / num_support
    point_edge[:, num_support:, num_support:] = 0.0
    diagonal = torch.arange(num_support, total, device=device)
    point_edge[:, diagonal, diagonal] = 1.0
    return distribution_node, point_edge, point_edge.clone()


def _cosine_cost(x, y):
    x = x / x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    y = y / y.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return (1.0 - torch.bmm(x.transpose(1, 2), y)).transpose(1, 2)


def _thresholded_self_cost(x):
    cost = _cosine_cost(x, x)
    threshold = cost.min() + 0.1 * (cost.max() - cost.min())
    return F.relu(cost - threshold)


def _ipot(cost, iterations=30, beta=0.5):
    batch, rows, columns = cost.shape
    sigma = torch.ones(batch, columns, 1, device=cost.device) / float(columns)
    transport = torch.ones(batch, rows, columns, device=cost.device)
    kernel = torch.exp(-cost / beta)
    q_value = kernel * transport
    for _ in range(iterations):
        delta = 1.0 / (rows * torch.bmm(q_value, sigma)).clamp_min(1e-12)
        sigma = 1.0 / (columns * torch.bmm(q_value.transpose(1, 2), delta)).clamp_min(1e-12)
        transport = delta * q_value * sigma.transpose(1, 2)
    return transport


def _wasserstein(source, target):
    cost = _cosine_cost(source, target).transpose(1, 2)
    threshold = cost.min() + 0.1 * (cost.max() - cost.min())
    cost = F.relu(cost - threshold)
    transport = _ipot(cost, iterations=30)
    value = -(cost.transpose(1, 2) * transport.transpose(1, 2)).sum((1, 2), keepdim=True)
    return value.abs().mean()


def _gromov_wasserstein(source, target, iterations=5, ot_iterations=20, beta=0.1):
    source_cost, target_cost = _thresholded_self_cost(source), _thresholded_self_cost(target)
    batch, source_n, _ = source_cost.shape
    target_n = target_cost.size(1)
    p = torch.full((batch, source_n, 1), 1.0 / source_n, device=source.device)
    q = torch.full((batch, target_n, 1), 1.0 / target_n, device=source.device)
    one_source = torch.ones(batch, source_n, 1, device=source.device)
    one_target = torch.ones(batch, target_n, 1, device=source.device)
    constant = (torch.bmm(torch.bmm(source_cost.square(), p), one_target.transpose(1, 2)) +
                torch.bmm(one_source, torch.bmm(q.transpose(1, 2), target_cost.square().transpose(1, 2))))
    coupling = torch.bmm(p, q.transpose(1, 2))
    current = constant
    for _ in range(iterations):
        current = constant - 2.0 * torch.bmm(torch.bmm(source_cost, coupling), target_cost.transpose(1, 2))
        coupling = _ipot(current, iterations=ot_iterations, beta=beta).detach()
    current = constant - 2.0 * torch.bmm(torch.bmm(source_cost, coupling), target_cost.transpose(1, 2))
    return (current * coupling).sum((1, 2)).abs().mean()


def optimal_transport_alignment(source_nodes, target_nodes):
    """Released Wasserstein + Gromov-Wasserstein alignment over graph generations."""
    values = []
    for source, target in zip(source_nodes, target_nodes):
        values.append(_wasserstein(source, target) + _gromov_wasserstein(source, target))
    return torch.stack(values).mean()


def initialize_official(module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.ones_(module.bias)


def euclidean_logits(query, prototypes):
    return -torch.sum((query[:, None, :] - prototypes[None, :, :]).square(), dim=-1)
