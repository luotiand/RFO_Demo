import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rfo_demo.core import constant_time_like, ensure_runtime_dirs, sample_binary_time_like, select_device, setup_torch
from rfo_demo.core.fno_utils import LpLoss as FnoLpLoss
from rfo_demo.core.fno_utils import MatReader, UnitGaussianNormalizer, count_params
from rfo_demo.models import get_model_class
from rfo_demo.optim import Adam
from rfo_demo.utils import plot_separate_loss_curves, setup_logger
from rfo_demo.visualization import plot_2d_results


TRAIN_DATA_PATH = "/data5/store1/dlt/RFO_Demo/data/ns_V1e-4_N1200_T30_R64.mat"
TEST_DATA_PATH = TRAIN_DATA_PATH
NTRAIN = 1000
NTEST = 200
PHYS_TIME_STEPS = 10
MODES = 8
WIDTH = 20
SCHEDULER_STEP = 100
SCHEDULER_GAMMA = 0.5
SUB = 1
GRID_SIZE = 64 // SUB
T_OUT = 20
ROLLOUT_STEPS = 10
CHECK_INTERVAL = 50


def _load_ns_data(device: torch.device):
    reader = MatReader(TRAIN_DATA_PATH)
    train_a = reader.read_field("u")[:NTRAIN, ::SUB, ::SUB, :PHYS_TIME_STEPS].to(device)
    train_u = reader.read_field("u")[:NTRAIN, ::SUB, ::SUB, T_OUT:T_OUT + ROLLOUT_STEPS].to(device)
    reader = MatReader(TEST_DATA_PATH)
    test_a = reader.read_field("u")[-NTEST:, ::SUB, ::SUB, :PHYS_TIME_STEPS].to(device)
    test_u = reader.read_field("u")[-NTEST:, ::SUB, ::SUB, T_OUT:T_OUT + ROLLOUT_STEPS].to(device)
    return train_a, train_u, test_a, test_u


def run(config: dict) -> None:
    setup_torch()
    device = select_device("cuda:0")
    save_path = config["save_path"]
    para_path = config["para_path"]
    ensure_runtime_dirs(save_path, para_path)
    forward_vis_path = os.path.join(save_path, "forward_phys_time")
    reverse_vis_path = os.path.join(save_path, "reverse_phys_time")
    os.makedirs(forward_vis_path, exist_ok=True)
    os.makedirs(reverse_vis_path, exist_ok=True)
    setup_logger(save_path)

    niter = config["niter"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    rf_dt = config["rf_dt"]
    train = config["train"]
    scorenet_model_class = config["scorenet_model_class"]
    model_name = config["model_name"]
    rf = config["rf"]

    train_a, train_u, test_a, test_u = _load_ns_data(device)
    logging.info("Data shape: %s", tuple(test_u.shape))

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a_norm = a_normalizer.encode(train_a)
    test_a_norm = a_normalizer.encode(test_a)
    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u_norm = y_normalizer.encode(train_u)
    test_u_norm = y_normalizer.encode(test_u)

    train_a_norm = train_a_norm.reshape(NTRAIN, GRID_SIZE, GRID_SIZE, PHYS_TIME_STEPS, 1).to(device)
    test_a_norm = test_a_norm.reshape(NTEST, GRID_SIZE, GRID_SIZE, PHYS_TIME_STEPS, 1).to(device)
    train_loader = DataLoader(TensorDataset(train_a_norm, train_u_norm), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_a_norm, test_u_norm), batch_size=1, shuffle=False)

    model_class = get_model_class(scorenet_model_class)
    model = model_class(MODES, MODES, 4, WIDTH).to(device)
    logging.info("Model parameters: %s", count_params(model))

    if train:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
    myloss = FnoLpLoss(size_average=False)

    if train:
        train_losses = []
        test_losses = []
        test_epochs = []
        for epoch in range(niter):
            model.train()
            train_l2 = 0.0
            for x, y in train_loader:
                current_bs = x.shape[0]
                t = sample_binary_time_like(x)
                optimizer.zero_grad()
                xt = rf.straight_process(x, y.unsqueeze(-1), t)
                out = model(xt, t).view(current_bs, GRID_SIZE, GRID_SIZE, ROLLOUT_STEPS)
                l2 = myloss(out.view(current_bs, -1), (y - x.squeeze(-1)).view(current_bs, -1))
                l2.backward()
                optimizer.step()
                train_l2 += l2.item()
            avg_train_loss = train_l2 / NTRAIN
            train_losses.append(avg_train_loss)
            scheduler.step()
            logging.info("Epoch %s/%s - Train Loss: %.6f", epoch + 1, niter, avg_train_loss)

            if (epoch + 1) % CHECK_INTERVAL == 0:
                model.eval()
                test_l2 = 0.0
                with torch.no_grad():
                    for x, y in test_loader:
                        current_bs = x.shape[0]
                        xt = [x]
                        for t_val in np.arange(0.0, 1, rf_dt):
                            t = constant_time_like(x, float(t_val))
                            score = model(xt[-1], t)
                            xt.append(rf.forward_process(xt[-1], score, dt=rf_dt))
                        y_decoded = y_normalizer.decode(y.squeeze(-2))
                        xt_last = xt[-1].view(current_bs, GRID_SIZE, GRID_SIZE, ROLLOUT_STEPS)
                        out_decoded = y_normalizer.decode(xt_last)
                        test_l2 += myloss(out_decoded.view(current_bs, -1), y_decoded.view(current_bs, -1)).item()
                avg_test_loss = test_l2 / NTEST
                test_losses.append(avg_test_loss)
                test_epochs.append(epoch + 1)
                logging.info("Evaluation Epoch %s - Test Loss: %.6f", epoch + 1, avg_test_loss)

        torch.save(model.state_dict(), f"{para_path}{model_name}")
        logging.info("Model saved to: %s%s", para_path, model_name)
        plot_separate_loss_curves(train_losses, test_losses, test_epochs, save_path, scorenet_model_class, PHYS_TIME_STEPS)

    model.load_state_dict(torch.load(f"{para_path}{model_name}", map_location=device))
    model.eval()
    logging.info("Start inference and save visualizations")

    with torch.no_grad():
        xt_forward = [test_a_norm]
        for t_val in np.arange(0.0, 1, rf_dt):
            t = constant_time_like(test_a_norm, float(t_val))
            score = model(xt_forward[-1], t)
            xt_forward.append(rf.forward_process(xt_forward[-1], score, dt=rf_dt))

        xt_reverse = [test_u_norm.unsqueeze(-1)]
        for t_val in np.arange(0.0, 1, rf_dt):
            t = constant_time_like(test_u_norm.unsqueeze(-1), float(t_val))
            score = model(xt_reverse[-1], 1 - t)
            xt_reverse.append(rf.reverse_process(xt_reverse[-1], score, dt=rf_dt))

        pred_u_forward = y_normalizer.decode(xt_forward[-1].view(NTEST, GRID_SIZE, GRID_SIZE, PHYS_TIME_STEPS))
        true_u = y_normalizer.decode(test_u_norm.view(NTEST, GRID_SIZE, GRID_SIZE, PHYS_TIME_STEPS))
        pred_a_reverse = a_normalizer.decode(xt_reverse[-1].view(NTEST, GRID_SIZE, GRID_SIZE, PHYS_TIME_STEPS))
        true_a = a_normalizer.decode(test_a_norm.view(NTEST, GRID_SIZE, GRID_SIZE, PHYS_TIME_STEPS))

        for t_step in range(PHYS_TIME_STEPS):
            plot_2d_results(data1=true_u[..., t_step], data2=pred_u_forward[..., t_step], labels=["Ground Truth", "Prediction"], base_filename=os.path.join(forward_vis_path, f"t{t_step}"))
        for t_step in range(PHYS_TIME_STEPS):
            plot_2d_results(data1=true_a[..., t_step], data2=pred_a_reverse[..., t_step], labels=["Ground Truth", "Prediction"], base_filename=os.path.join(reverse_vis_path, f"t{t_step}"))
        logging.info("Visualizations saved to: %s and %s", forward_vis_path, reverse_vis_path)
