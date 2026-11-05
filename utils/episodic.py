"""Dataset and sampler helpers for episodic HSI training."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class ArraySplitDataset(Dataset):
    def __init__(self, sample_pack, train=True):
        indices = np.flatnonzero(sample_pack["set"] == (1 if train else 3))
        data = sample_pack["data"][:, :, :, indices]
        self.data = data.transpose((3, 2, 0, 1))
        self.labels = sample_pack["Labels"][indices]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class EpisodicDataset(Dataset):
    def __init__(self, task, split="train"):
        if split not in {"train", "test"}:
            raise ValueError(f"unknown episodic split: {split}")
        if split == "train":
            self.data = task.support_datas
            self.labels = task.support_labels
        else:
            self.data = task.query_datas
            self.labels = task.query_labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, num_classes, available_per_class, shuffle=True):
        self.num_per_class = num_per_class
        self.num_classes = num_classes
        self.available_per_class = available_per_class
        self.shuffle = shuffle

    def __iter__(self):
        batch = []
        for class_index in range(self.num_classes):
            if self.shuffle:
                selected = torch.randperm(self.available_per_class)[: self.num_per_class]
                selected = selected.tolist()
            else:
                selected = list(range(self.num_per_class))
            offset = class_index * self.available_per_class
            batch.extend(offset + index for index in selected)
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        # One sampler iteration yields one complete episode.
        return 1


def zero_pad_cube(data):
    """Place an HSI cube in the center of a threefold zero-padded canvas."""
    zeros = np.zeros_like(data)
    empty_row = np.concatenate((zeros, zeros, zeros), axis=1)
    center_row = np.concatenate((zeros, data, zeros), axis=1)
    return np.concatenate((empty_row, center_row, empty_row), axis=0)


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
    """Apply multiplicative intensity variation and additive Gaussian noise."""
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    return alpha * data + beta * noise
