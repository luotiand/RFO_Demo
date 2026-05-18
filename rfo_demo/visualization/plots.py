from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch



def plot_3d_compare_with_diff(data1, data2, titles, cmap_main="viridis", cmap_diff="seismic", elev: int = 25, azim: int = -135, filename: str = "operator_learning_3d.png", figsize=(24, 8)):
    data1_avg = data1.mean(axis=0)
    data2_avg = data2.mean(axis=0)
    diff = data1_avg - data2_avg
    fig = plt.figure(figsize=figsize)
    axes = [
        fig.add_subplot(131, projection="3d"),
        fig.add_subplot(132, projection="3d"),
        fig.add_subplot(133, projection="3d"),
    ]
    _plot_3d_field(axes[0], data1_avg, title=titles[0], cmap=cmap_main, elev=elev, azim=azim)
    _plot_3d_field(axes[1], data2_avg, title=titles[1], cmap=cmap_main, elev=elev, azim=azim)
    _plot_3d_diff(axes[2], diff, title=titles[2], cmap=cmap_diff, elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



def _plot_3d_field(ax, data, title, cmap, elev, azim):
    t_dim, h_dim, w_dim = data.shape
    voxel_array = np.zeros((w_dim + 1, h_dim + 1, t_dim + 1), dtype=bool)
    voxel_array[:-1, :-1, :-1] = data.transpose(2, 1, 0) > np.median(data)
    colors = np.zeros((w_dim + 1, h_dim + 1, t_dim + 1, 4))
    normalized_data = data.transpose(2, 1, 0) / data.max()
    colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(normalized_data)
    ax.voxels(voxel_array, facecolors=colors, edgecolor="none", alpha=0.3)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Width", labelpad=10)
    ax.set_ylabel("Height", labelpad=10)
    ax.set_zlabel("Time Step", labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(data)
    plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)



def _plot_3d_diff(ax, diff, title, cmap, elev, azim):
    t_dim, h_dim, w_dim = diff.shape
    vmax = np.abs(diff).max()
    norm_diff = diff / vmax
    pos_mask = np.zeros((w_dim + 1, h_dim + 1, t_dim + 1), dtype=bool)
    neg_mask = np.zeros((w_dim + 1, h_dim + 1, t_dim + 1), dtype=bool)
    pos_mask[:-1, :-1, :-1] = diff.transpose(2, 1, 0) > 0
    neg_mask[:-1, :-1, :-1] = diff.transpose(2, 1, 0) < 0
    pos_colors = np.zeros((w_dim + 1, h_dim + 1, t_dim + 1, 4))
    neg_colors = np.zeros((w_dim + 1, h_dim + 1, t_dim + 1, 4))
    pos_colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(0.5 + norm_diff.transpose(2, 1, 0) / 2)
    neg_colors[:-1, :-1, :-1] = plt.cm.get_cmap(cmap)(0.5 + norm_diff.transpose(2, 1, 0) / 2)
    ax.voxels(pos_mask, facecolors=pos_colors, edgecolor="none", alpha=0.5)
    ax.voxels(neg_mask, facecolors=neg_colors, edgecolor="none", alpha=0.5)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Width", labelpad=10)
    ax.set_ylabel("Height", labelpad=10)
    ax.set_zlabel("Time Step", labelpad=10)
    ax.view_init(elev=elev, azim=azim)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)



def hinton(matrix, max_weight=None, ax=None):
    plt.figure(figsize=(6, 6))
    ax = ax if ax is not None else plt.gca()
    if max_weight is None:
        max_weight = 2 ** np.ceil(np.log(np.max(np.abs(matrix))) / np.log(2))
    ax.patch.set_facecolor("gray")
    ax.set_aspect("auto", "box")
    ax.set_xticks([])
    ax.set_yticks([])
    for (x_pos, y_pos), weight in np.ndenumerate(matrix):
        color = "white" if weight > 0 else "black"
        size = np.sqrt(np.abs(weight) / max_weight)
        rect = plt.Rectangle([x_pos - size / 2, y_pos - size / 2], size, size, facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    ax.autoscale_view()
    ax.invert_yaxis()
    plt.title("Hinton Diagram of Model Weights")
    plt.savefig("hint", dpi=300)
    plt.close()



def plot_2d_results(data1, data2, labels, base_filename):
    with torch.no_grad():
        batch_size = data1.shape[0]
        rand_indices = torch.randperm(batch_size)[:5].tolist()
    for i, idx in enumerate(rand_indices, 1):
        samples = [data1[idx].cpu().numpy(), data2[idx].cpu().numpy()]
        suffixes = ["gt", "predict"]
        for sample, label, suffix in zip(samples, labels, suffixes):
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[94, 6], wspace=0.0)
            ax = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
            im = ax.imshow(sample, cmap="viridis", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=36, pad=10)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=24, pad=1)
            plt.subplots_adjust(left=0.05, right=0.85, top=0.85, bottom=0.05)
            plt.savefig(f"{base_filename}_rand{i}_{suffix}.png", dpi=300)
            plt.close(fig)



def plot_1d_results(data1, data2, labels, title, filename):
    with torch.no_grad():
        batch_size = data1.shape[0]
        data1_flat = data1.view(batch_size, -1)
        data2_flat = data2.view(batch_size, -1)
        diff_norm = torch.norm(data1_flat - data2_flat, p=2, dim=1)
        data2_norm = torch.norm(data2_flat, p=2, dim=1)
        data2_norm_safe = torch.where(data2_norm < 1e-10, torch.ones_like(data2_norm) * 1e-10, data2_norm)
        sample_l2_rel = diff_norm / data2_norm_safe
        min_l2_idx = torch.argmin(sample_l2_rel).item()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].plot(data1[min_l2_idx].cpu().detach().numpy())
    axes[0].set_title(labels[0])
    axes[0].grid(True)
    axes[1].plot(data2[min_l2_idx].cpu().detach().numpy())
    axes[1].set_title(labels[1])
    axes[1].grid(True)
    diff = data1[min_l2_idx].cpu().detach().numpy() - data2[min_l2_idx].cpu().detach().numpy()
    axes[2].plot(diff)
    axes[2].set_title(f"Difference ({labels[0]} - {labels[1]})")
    axes[2].grid(True)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filename, dpi=300)
    plt.close()



def plot_results(data1, data2, labels, title, xlabel, ylabel, filename):
    with torch.no_grad():
        data1_flat = data1.view(data1.shape[0], -1)
        data2_flat = data2.view(data2.shape[0], -1)
        diff_norm = torch.norm(data1_flat - data2_flat, p=2, dim=1)
        data2_norm = torch.norm(data2_flat, p=2, dim=1)
        data2_norm_safe = torch.where(data2_norm < 1e-10, torch.ones_like(data2_norm) * 1e-10, data2_norm)
        overall_l2_rel = (diff_norm / data2_norm_safe).mean().item()
    plt.figure()
    plt.plot(data1.mean(dim=1).cpu().detach().numpy(), label=labels[0].format(len(data1)))
    plt.plot(data2.mean(dim=1).cpu().detach().numpy(), label=labels[1].format(len(data2)))
    plt.title(f"{title}\noverral_L2error: {overall_l2_rel:.6f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(filename, dpi=300)
    plt.close()
