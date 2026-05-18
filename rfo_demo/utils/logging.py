from __future__ import annotations

import logging
from pathlib import Path



def setup_logger(save_path: str) -> logging.Logger:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    log_file = Path(save_path) / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)
