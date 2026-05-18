from .config import load_python_config
from .logging import setup_logger


def plot_forward_reverse_comparison(*args, **kwargs):
    from .plotting import plot_forward_reverse_comparison as _impl

    return _impl(*args, **kwargs)


def plot_separate_loss_curves(*args, **kwargs):
    from .plotting import plot_separate_loss_curves as _impl

    return _impl(*args, **kwargs)


def plot_training_metrics(*args, **kwargs):
    from .plotting import plot_training_metrics as _impl

    return _impl(*args, **kwargs)


__all__ = ["load_python_config", "plot_forward_reverse_comparison", "plot_separate_loss_curves", "plot_training_metrics", "setup_logger"]
