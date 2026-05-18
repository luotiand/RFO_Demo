from .fno1d import FNO1d
from .fno2d import FNO2d
from .fno3d import FNO3d

MODEL_REGISTRY = {
    "FNO1d": FNO1d,
    "FNO2d": FNO2d,
    "FNO3d": FNO3d,
}



def get_model_class(name: str):
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}") from exc


__all__ = ["FNO1d", "FNO2d", "FNO3d", "MODEL_REGISTRY", "get_model_class"]
