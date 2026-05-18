import torch

from rfo_demo.operators import RectFlow

CONFIG = {
    "para_path": "/data5/store1/dlt/rectified_flow/modelpara/",
    "save_path": "/data5/store1/dlt/rectified_flow/result/ns_plot/",
    "eq_T": 1.0,
    "N": 10000,
    "niter": 500,
    "lr": 1e-3,
    "batch_size": 8,
    "T": 1.0,
    "rf_dt": 1,
    "h_dim": 128,
    "train": 1,
    "rf": RectFlow(),
    "scorenet_model_class": "FNO3d",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_name": "fno3d.pth",
}

globals().update(CONFIG)
