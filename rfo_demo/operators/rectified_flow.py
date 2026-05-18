import torch


class RectFlow:
    def __init__(self) -> None:
        self.name = "rectified_flow"

    def straight_process(self, x0, x_t, t):
        return (1 - t) * x0 + t * x_t

    def forward_process(self, xt, score, dt: float):
        return xt + dt * score

    def reverse_process(self, xt, score, dt: float):
        return xt - dt * score


__all__ = ["RectFlow"]
