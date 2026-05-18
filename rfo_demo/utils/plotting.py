from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

from rfo_demo.visualization import hinton, plot_1d_results, plot_2d_results, plot_3d_compare_with_diff, plot_results



def plot_forward_reverse_comparison(forward_gt, forward_pred, reverse_gt, reverse_pred, base_save_path: str) -> None:
    def to_numpy(value):
        return value.cpu().numpy() if isinstance(value, torch.Tensor) else value

    f_gt = to_numpy(forward_gt).astype(np.float32)
    f_pred = to_numpy(forward_pred).astype(np.float32)
    r_gt = to_numpy(reverse_gt).astype(np.float32)
    r_pred = to_numpy(reverse_pred).astype(np.float32)
    batch_size = f_gt.shape[0]
    if batch_size < 5:
        raise ValueError(f"Batch size ({batch_size}) is smaller than 5.")
    rand_indices = np.random.permutation(batch_size)[:5].tolist()
    x_axis = np.linspace(0, 1, f_gt.shape[1])

    for plot_index, sample_index in enumerate(rand_indices, start=1):
        fig_forward, ax_forward = plt.subplots(1, 1, figsize=(10, 6))
        ax_forward.plot(x_axis, f_gt[sample_index], "b-", linewidth=2.5, label="Ground Truth")
        ax_forward.plot(x_axis, f_pred[sample_index], "r--", linewidth=2.5, label="Predicted")
        ax_forward.set_title("Forward", fontsize=36)
        ax_forward.set_xlabel("Position", fontsize=28)
        ax_forward.set_ylabel("Value", fontsize=28)
        ax_forward.legend(fontsize=18, frameon=False, loc="upper right")
        ax_forward.grid(False)
        plt.tight_layout()
        plt.savefig(f"{base_save_path}_forward_rand{plot_index}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_forward)

        fig_reverse, ax_reverse = plt.subplots(1, 1, figsize=(10, 6))
        ax_reverse.plot(x_axis, r_gt[sample_index], "b-", linewidth=2.5, label="Ground Truth")
        ax_reverse.plot(x_axis, r_pred[sample_index], "r--", linewidth=2.5, label="Predicted")
        ax_reverse.set_title("Inverse", fontsize=36)
        ax_reverse.set_xlabel("Position", fontsize=28)
        ax_reverse.set_ylabel("Value", fontsize=28)
        ax_reverse.legend(fontsize=18, frameon=False, loc="upper right")
        ax_reverse.grid(False)
        plt.tight_layout()
        plt.savefig(f"{base_save_path}_reverse_rand{plot_index}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_reverse)



def plot_training_metrics(train_losses, test_l2rel_list, niter: int, check_interval: int, save_path: str, model_name: str, target_len: int) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(range(1, niter + 1), train_losses, "b-", label="Train L2 Relative Loss")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_yscale("log")
    ax1.set_title("Training Loss", fontsize=14)
    ax1.legend()
    ax1.grid(True)

    eval_epochs = list(range(check_interval, niter + 1, check_interval))
    if not eval_epochs or eval_epochs[-1] != niter:
        eval_epochs.append(niter)
    eval_epochs = eval_epochs[: len(test_l2rel_list)]

    ax2.plot(eval_epochs, test_l2rel_list, "g-s", label="Test L2 Relative Error")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Error", fontsize=12)
    ax2.set_title("Test L2 Relative Error", fontsize=14)
    ax2.legend()
    ax2.grid(True)
    plt.savefig(f"{save_path}{model_name.lower()}_{target_len}_training_metrics.png", dpi=300)
    plt.close()



def plot_separate_loss_curves(train_losses, test_losses, test_epochs, save_path: str, model_class: str, target_len: int) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(range(1, len(train_losses) + 1), train_losses, "b-", label="Train L2 Loss")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_yscale("log")
    ax1.set_title("Training L2 Loss Curve", fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(test_epochs, test_losses, "g-s", label="Test L2 Loss")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_yscale("log")
    ax2.set_title("Test L2 Loss Curve", fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_path}{model_class.lower()}_{target_len}_loss_curves.png", dpi=300)
    plt.close()


__all__ = [
    "hinton",
    "plot_1d_results",
    "plot_2d_results",
    "plot_3d_compare_with_diff",
    "plot_forward_reverse_comparison",
    "plot_results",
    "plot_separate_loss_curves",
    "plot_training_metrics",
]
