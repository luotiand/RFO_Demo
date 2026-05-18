from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from rfo_demo.core.fno_utils import MatReader


class BurgersDataset(Dataset):
    def __init__(self, file_path, train_ratio: float = 0.9, mode: str = "train", target_len=None):
        reader = MatReader(file_path, to_cuda=False)
        self.inputs = reader.read_field("a")
        self.outputs = reader.read_field("u")
        self.target_len = target_len
        if target_len is not None:
            self.inputs = self.downsample_by_step(self.inputs, target_len)
            self.outputs = self.downsample_by_step(self.outputs, target_len, dim=-1)

        total_samples = len(self.inputs)
        indices = np.random.permutation(total_samples)
        split_idx = int(train_ratio * total_samples)
        self.indices = indices[:split_idx] if mode == "train" else indices[split_idx:]

    @staticmethod
    def downsample_by_step(x, target_len, dim: int = 1):
        original_len = x.shape[dim]
        step = original_len // target_len
        indices = torch.arange(0, original_len, step, device=x.device)[:target_len]
        return torch.index_select(x, dim=dim, index=indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.inputs[original_idx].float(), self.outputs[original_idx].float()

    def get_full_data(self):
        return self.inputs[self.indices], self.outputs[self.indices]


class AdvectionDataset(Dataset):
    def __init__(self, file_path, train_ratio: float = 0.9, mode: str = "train", target_len: int | None = 1024):
        with h5py.File(file_path, "r") as handle:
            u = np.array(handle["tensor"][:], dtype=np.float32)
        self.inputs = torch.from_numpy(u[0])
        self.outputs = torch.from_numpy(u[-1])
        self.target_len = target_len
        if target_len is not None:
            self.inputs = self.downsample_by_step(self.inputs, target_len)
            self.outputs = self.downsample_by_step(self.outputs, target_len, dim=-1)

        total_samples = len(self.inputs)
        indices = np.random.permutation(total_samples)
        split_idx = int(train_ratio * total_samples)
        self.indices = indices[:split_idx] if mode == "train" else indices[split_idx:]

    @staticmethod
    def downsample_by_step(x, target_len, dim: int = 1):
        original_len = x.shape[dim]
        step = original_len // target_len
        indices = torch.arange(0, original_len, step, device=x.device)[:target_len]
        return torch.index_select(x, dim=dim, index=indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.inputs[original_idx].float(), self.outputs[original_idx].float()

    def get_full_data(self):
        return self.inputs[self.indices], self.outputs[self.indices]


class DarcyDataset(Dataset):
    def __init__(self, file_path, train_ratio: float = 1.0, test_ratio: float = 0.05, mode: str = "train", target_size=None):
        reader = MatReader(file_path, to_cuda=False)
        self.inputs = reader.read_field("coeff")
        self.outputs = reader.read_field("sol")
        self.target_size = target_size
        if target_size is not None:
            self.inputs = self.downsample_2d(self.inputs, target_size)
            self.outputs = self.downsample_2d(self.outputs, target_size)

        total_samples = len(self.inputs)
        indices = np.random.permutation(total_samples)
        split_idx = int(train_ratio * total_samples)
        test_idx = int(test_ratio * total_samples)
        self.indices = indices[:split_idx] if mode == "train" else indices[:test_idx]

    @staticmethod
    def downsample_2d(x, target_size):
        original_size = x.shape[1]
        indices = torch.linspace(0, original_size - 1, target_size, dtype=torch.long, device=x.device)
        x_downsampled = torch.index_select(x, dim=1, index=indices)
        return torch.index_select(x_downsampled, dim=2, index=indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.inputs[original_idx].float(), self.outputs[original_idx].float()

    def get_full_data(self):
        return self.inputs[self.indices], self.outputs[self.indices]


class MyDatasetNs(Dataset):
    def __init__(self, file_path: str, input_steps: int = 10, pred_steps: int = 10, train: bool = True, train_ratio: float = 0.9, seed: int = 42):
        with h5py.File(file_path, "r") as handle:
            u = np.array(handle["u"][()], dtype=np.float32)
        u_processed = u.transpose(3, 0, 1, 2)
        self.x = u_processed[:, :input_steps, :, :]
        self.y = u_processed[:, input_steps:input_steps + pred_steps, :, :]
        np.random.seed(seed)
        indices = np.random.permutation(self.x.shape[0])
        split_idx = int(len(indices) * train_ratio)
        self.indices = indices[:split_idx] if train else indices[split_idx:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return torch.from_numpy(self.x[real_idx]), torch.from_numpy(self.y[real_idx])

    def get_full_data(self):
        return torch.from_numpy(self.x[self.indices]), torch.from_numpy(self.y[self.indices])
