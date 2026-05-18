import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from rfo_demo.core import LpLoss, calculate_l2_relative_error, constant_time_like, ensure_runtime_dirs, sample_binary_time_like, select_device, setup_torch
from rfo_demo.data import AdvectionDataset
from rfo_demo.models import get_model_class
from rfo_demo.optim import Adam
from rfo_demo.utils import plot_forward_reverse_comparison, plot_training_metrics, setup_logger


TRAIN_DATA_PATH = "/data5/store1/dlt/PDEBench/pdebench/data_download/data/1D_broken/Advection/Train/1D_Advection_Sols_beta0.1.hdf5"
TEST_DATA_PATH = TRAIN_DATA_PATH



def _build_loaders(batch_size: int, target_len: int):
    train_dataset = AdvectionDataset(TRAIN_DATA_PATH, mode="train", target_len=target_len)
    test_dataset = AdvectionDataset(TEST_DATA_PATH, mode="test", target_len=target_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    return train_dataset, test_dataset, train_loader, test_loader



def _compute_normalization(train_loader):
    a_all, x_all = [], []
    for a_batch, x_batch in train_loader:
        a_all.append(a_batch)
        x_all.append(x_batch)
    a_all = torch.cat(a_all, dim=0)
    x_all = torch.cat(x_all, dim=0)
    return a_all.mean(), a_all.std() + 1e-6, x_all.mean(), x_all.std() + 1e-6



def run(config: dict) -> None:
    setup_torch()
    device = select_device("cuda:0")
    save_path = config["save_path"]
    para_path = config["para_path"]
    ensure_runtime_dirs(save_path, para_path)
    setup_logger(save_path)

    niter = config["niter"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    total_time = config["T"]
    rf_dt = config["rf_dt"]
    train = config["train"]
    scorenet_model_class = config["scorenet_model_class"]
    model_name = config["model_name"]
    rf = config["rf"]
    target_len = config["target_len"]

    logging.info("Initializing training on device: %s", device)
    train_dataset, test_dataset, train_loader, test_loader = _build_loaders(batch_size, target_len)
    logging.info("Training dataset size: %s, Test dataset size: %s", len(train_dataset), len(test_dataset))
    a_mean, a_std, x_mean, x_std = _compute_normalization(train_loader)
    logging.info("Normalization stats - a: (mean=%.6f, std=%.6f); x: (mean=%.6f, std=%.6f)", a_mean, a_std, x_mean, x_std)

    model_class = get_model_class(scorenet_model_class)
    score_net = model_class(16, 64).to(device)
    logging.info("Model initialized: %s on %s", scorenet_model_class, device)

    sample_a, sample_x = test_dataset.get_full_data()
    a = (sample_a.to(device) - a_mean) / a_std
    x = (sample_x.to(device) - x_mean) / x_std
    logging.info("Test data loaded - a shape: %s, x shape: %s", tuple(a.shape), tuple(x.shape))

    if train:
        optimizer = Adam(score_net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = LpLoss(p=2, reduction="mean")
        train_losses = []
        test_l2rel_list = []
        check_interval = max(1, niter // 10)
        start_time = time.time()
        logging.info("Start training - epochs: %s, batch_size: %s", niter, batch_size)

        for epoch in range(niter):
            score_net.train()
            train_loss = 0.0
            epoch_start = time.time()

            for a_batch, x_batch in train_loader:
                a_batch = (a_batch.to(device) - a_mean) / a_std
                x_batch = (x_batch.to(device) - x_mean) / x_std
                t = sample_binary_time_like(a_batch)

                optimizer.zero_grad()
                xt_batch = rf.straight_process(a_batch, x_batch, t)
                exact_score = x_batch - a_batch
                pred_score = score_net(xt_batch, t)
                loss = criterion(pred_score, exact_score)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                del a_batch, x_batch, t, xt_batch, exact_score, pred_score
                torch.cuda.empty_cache()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            scheduler.step()

            if (epoch + 1) % check_interval == 0 or epoch == niter - 1:
                score_net.eval()
                with torch.no_grad():
                    xt = [a]
                    for t_val in np.arange(0.0, total_time, rf_dt):
                        t_tensor = constant_time_like(a, float(t_val))
                        score = score_net(xt[-1], t_tensor)
                        xt.append(rf.forward_process(xt[-1], score, dt=rf_dt))
                    xt_last = xt[-1] * x_std + x_mean
                    x_true = x * x_std + x_mean
                    l2_rel = calculate_l2_relative_error(xt_last, x_true)
                    test_l2rel_list.append(l2_rel)
                    logging.info("Epoch %s/%s - Train Loss: %.6f, Test L2 Rel: %.6f", epoch + 1, niter, avg_train_loss, l2_rel)
                score_net.train()

            if (epoch + 1) % 5 == 0:
                now = time.time()
                epoch_time = now - epoch_start
                remaining_time = (niter - epoch - 1) * (now - start_time) / (epoch + 1)
                logging.info("Epoch %s - Train Loss: %.6f, Time: %.2fs, Estimated Remaining: %.2fmin", epoch + 1, avg_train_loss, epoch_time, remaining_time / 60)

        torch.save(score_net.state_dict(), f"{para_path}{model_name}")
        logging.info("Model saved to: %s%s", para_path, model_name)
        plot_training_metrics(train_losses, test_l2rel_list, niter, check_interval, save_path, scorenet_model_class, target_len)

    score_net.load_state_dict(torch.load(f"{para_path}{model_name}", map_location=device))
    score_net.eval()
    logging.info("Start inference and evaluation")

    with torch.no_grad():
        xt = [a]
        for t_val in np.arange(0.0, total_time, rf_dt):
            t_tensor = constant_time_like(a, float(t_val))
            score = score_net(xt[-1], t_tensor)
            xt.append(rf.forward_process(xt[-1], score, dt=rf_dt))

        yt = [x]
        for t_val in np.arange(0.0, total_time, rf_dt):
            t_tensor = constant_time_like(a, float(t_val))
            score = score_net(yt[-1], total_time - t_tensor)
            yt.append(rf.reverse_process(yt[-1], score, dt=rf_dt))

        xt_last = xt[-1] * x_std + x_mean
        x_true = x * x_std + x_mean
        yt_last = yt[-1] * a_std + a_mean
        y_true = a * a_std + a_mean
        final_l2rel = calculate_l2_relative_error(xt_last, x_true)
        logging.info("Final evaluation - L2 Relative Error: %.6f", final_l2rel)
        plot_forward_reverse_comparison(
            forward_gt=x_true,
            forward_pred=xt_last,
            reverse_gt=y_true,
            reverse_pred=yt_last,
            base_save_path=f"{save_path}{scorenet_model_class.lower()}_{target_len}",
        )
