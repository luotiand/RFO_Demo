import torch

from rfo_demo.models import FNO1d, FNO2d, FNO3d
from rfo_demo.operators import RectFlow


if __name__ == "__main__":
    rf = RectFlow()
    model_1d = FNO1d(4, 8)
    x1 = torch.randn(2, 16)
    t1 = torch.zeros_like(x1)
    print("FNO1d:", model_1d(x1, t1).shape)

    model_2d = FNO2d(4, 4, 8)
    x2 = torch.randn(2, 8, 8)
    t2 = torch.zeros_like(x2)
    print("FNO2d:", model_2d(x2, t2).shape)

    model_3d = FNO3d(4, 4, 3, 8)
    x3 = torch.randn(2, 8, 8, 4, 1)
    t3 = torch.zeros_like(x3)
    print("FNO3d:", model_3d(x3, t3).shape)
    print("RectFlow:", rf.forward_process(x1, torch.ones_like(x1), 0.1).shape)
