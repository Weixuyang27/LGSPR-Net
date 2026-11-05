"""Author's PyTorch SSRN, adapted for NCHW patches and arbitrary band counts.

Source: zilongzhong/SSTN, NetworksBlocks.py, commit
1b377d9c1157d37c17c03ca17aaa3b655378d547 (Zilong Zhong et al.).
Preserves module names/weights; replaces final squeeze with flatten(1) to
retain the batch dimension for a final batch containing one example.
"""
import torch.nn as nn
import torch.nn.functional as F


class SPCModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, (7, 1, 1),
                            stride=(2, 1, 1), bias=False)

    def forward(self, x):
        return self.s1(x.unsqueeze(1)).squeeze(1)


class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, (k, 3, 3), bias=False)

    def forward(self, x):
        return self.s1(x).squeeze(2)


class ResSPC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spc1 = nn.Sequential(
            nn.Conv3d(channels, channels, (7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True), nn.BatchNorm3d(channels))
        self.spc2 = nn.Sequential(
            nn.Conv3d(channels, channels, (7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True))
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        return F.leaky_relu(self.bn2(self.spc2(self.spc1(x))) + x)


class ResSPA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spa1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(inplace=True), nn.BatchNorm2d(channels))
        self.spa2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.leaky_relu(self.bn2(self.spa2(self.spa1(x))) + x)


class SSRN(nn.Module):
    def __init__(self, n_bands=102, num_classes=9):
        super().__init__()
        if n_bands < 7 or num_classes < 2:
            raise ValueError("SSRN needs at least 7 bands and 2 classes")
        self.n_bands = n_bands
        self.layer1 = SPCModuleIN(1, 28)
        self.layer2 = ResSPC(28)
        self.layer3 = ResSPC(28)
        self.layer4 = SPAModuleIN(28, 28, (n_bands - 7) // 2 + 1)
        self.bn4 = nn.BatchNorm2d(28)
        self.layer5 = ResSPA(28)
        self.layer6 = ResSPA(28)
        self.fc = nn.Linear(28, num_classes)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.n_bands:
            raise ValueError(f"Expected [N, {self.n_bands}, H, W], got {tuple(x.shape)}")
        if x.shape[-1] != x.shape[-2] or x.shape[-1] < 5:
            raise ValueError("Expected square patches of size >= 5")
        x = F.leaky_relu(self.layer1(x))
        x = self.layer3(self.layer2(x))
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer6(self.layer5(x))
        return self.fc(F.avg_pool2d(x, x.size(-1)).flatten(1))
