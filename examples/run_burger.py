from pathlib import Path

from rfo_demo.cli import run_case


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "rfo_demo" / "configs" / "burger.py"
    run_case("burger", str(config_path))
