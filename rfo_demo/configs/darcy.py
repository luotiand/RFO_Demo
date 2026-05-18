import torch

from rfo_demo.operators import RectFlow

CONFIG = {
    "para_path": "/data5/store1/dlt/rectified_flow/modelpara/",
    "save_path": "/data5/store1/dlt/rectified_flow/result/darcyplot/",
    "eq_T": 1.0,
    "N": 10000,
    "niter": 500,
    "target_len": 85,
    "lr": 1e-3,
    "batch_size": 16,
    "T": 1.0,
    "rf_dt": 1.0,
    "h_dim": 64,
    "train": 0,
    "rf": RectFlow(),
    "scorenet_model_class": "FNO2d",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_name": "fno2d.pth",
}

globals().update(CONFIG)
