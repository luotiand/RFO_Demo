import torch

from rfo_demo.operators import RectFlow

CONFIG = {
    "para_path": "/data5/store1/dlt/rectified_flow/modelpara/",
    "save_path": "/data5/store1/dlt/rectified_flow/result/burgerLNO32/",
    "eq_T": 1.0,
    "N": 10000,
    "niter": 500,
    "target_len": 256,
    "lr": 1e-3,
    "batch_size": 20,
    "T": 1.0,
    "rf_dt": 1.0,
    "h_dim": 4096,
    "train": 1,
    "rf": RectFlow(),
    "scorenet_model_class": "FNO1d",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_name": "fno1d.pth",
}

globals().update(CONFIG)
