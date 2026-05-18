from __future__ import annotations

import operator
from functools import reduce

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn as nn


class MatReader:
    def __init__(self, file_path, to_torch: bool = True, to_cuda: bool = False, to_float: bool = True):
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self) -> None:
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            self.data = h5py.File(self.file_path, mode="r")
            self.old_mat = False

    def load_file(self, file_path) -> None:
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

    def set_cuda(self, to_cuda: bool) -> None:
        self.to_cuda = to_cuda

    def set_torch(self, to_torch: bool) -> None:
        self.to_torch = to_torch

    def set_float(self, to_float: bool) -> None:
        self.to_float = to_float


class UnitGaussianNormalizer:
    def __init__(self, x, eps: float = 1e-5):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            else:
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        return (x * std) + mean

    def cuda(self) -> None:
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self) -> None:
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class GaussianNormalizer:
    def __init__(self, x, eps: float = 1e-5):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, sample_idx=None):
        return (x * (self.std + self.eps)) + self.mean

    def cuda(self) -> None:
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self) -> None:
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class RangeNormalizer:
    def __init__(self, x, low: float = 0.0, high: float = 1.0):
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)
        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        shape = x.size()
        x = x.view(shape[0], -1)
        x = self.a * x + self.b
        return x.view(shape)

    def decode(self, x):
        shape = x.size()
        x = x.view(shape[0], -1)
        x = (x - self.b) / self.a
        return x.view(shape)


class LpLoss:
    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)
        if self.reduction:
            return torch.mean(all_norms) if self.size_average else torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            return torch.mean(diff_norms / y_norms) if self.size_average else torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize: bool = False):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = nn.ModuleList()
        for idx in range(self.n_layers):
            self.layers.append(nn.Linear(layers[idx], layers[idx + 1]))
            if idx != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[idx + 1]))
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



def count_params(model) -> int:
    total = 0
    for param in list(model.parameters()):
        total += reduce(operator.mul, list(param.size() + (2,) if param.is_complex() else param.size()))
    return total
