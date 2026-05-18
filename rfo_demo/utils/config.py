from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any


IGNORED_KEYS = {"__builtins__"}


def _is_exported_value(key: str, value: Any) -> bool:
    if key in IGNORED_KEYS or key.startswith("__"):
        return False
    if isinstance(value, ModuleType):
        return False
    return True



def load_python_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    namespace: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        exec(handle.read(), namespace)
    if "CONFIG" in namespace:
        return dict(namespace["CONFIG"])
    return {key: value for key, value in namespace.items() if _is_exported_value(key, value)}
