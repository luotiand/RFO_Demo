from __future__ import annotations

from pathlib import Path

import torch


class LpLoss:
    def __init__(self, p: int = 2, reduction: str = "mean") -> None:
        self.p = p
        self.reduction = reduction

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size(0)
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)
        diff_norm = torch.norm(x_flat - y_flat, p=self.p, dim=1)
        y_norm = torch.norm(y_flat, p=self.p, dim=1)
        y_norm_safe = torch.where(y_norm < 1e-10, torch.full_like(y_norm, 1e-10), y_norm)
        rel_error = diff_norm / y_norm_safe
        if self.reduction == "mean":
            return torch.mean(rel_error)
        if self.reduction == "sum":
            return torch.sum(rel_error)
        return rel_error

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size(0)
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)
        diff_norm = torch.norm(x_flat - y_flat, p=self.p, dim=1)
        if self.reduction == "mean":
            return torch.mean(diff_norm)
        if self.reduction == "sum":
            return torch.sum(diff_norm)
        return diff_norm

    def __call__(self, x: torch.Tensor, y: torch.Tensor, mode: str = "rel") -> torch.Tensor:
        if mode == "abs":
            return self.abs(x, y)
        return self.rel(x, y)


def setup_torch() -> None:
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = True


def select_device(preferred_cuda: str = "cuda:0") -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if preferred_cuda == "cuda:1" and torch.cuda.device_count() > 1:
        return torch.device("cuda:1")
    return torch.device("cuda:0")


def sample_binary_time_like(reference: torch.Tensor) -> torch.Tensor:
    batch_size = reference.shape[0]
    time_options = torch.tensor([0.0, 1.0], device=reference.device, dtype=reference.dtype)
    view_shape = (batch_size,) + (1,) * (reference.ndim - 1)
    rand_indices = torch.randint(0, 2, view_shape, device=reference.device)
    return time_options[rand_indices].expand_as(reference)


def constant_time_like(reference: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full_like(reference, fill_value=value)


def calculate_l2_relative_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    with torch.no_grad():
        return LpLoss(p=2, reduction="mean")(pred, true, mode="rel").item()


def ensure_runtime_dirs(save_path: str, para_path: str) -> None:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(para_path).mkdir(parents=True, exist_ok=True)
