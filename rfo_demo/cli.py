import argparse

from rfo_demo.experiments import run_experiment
from rfo_demo.utils.config import load_python_config


def run_case(case: str, config_path: str) -> None:
    config = load_python_config(config_path)
    run_experiment(case, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RFO Demo unified CLI")
    parser.add_argument("--case", required=True, choices=["advection", "burger", "darcy", "ns"], help="Experiment name")
    parser.add_argument("--config", required=True, help="Path to a Python config file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_case(args.case, args.config)


if __name__ == "__main__":
    main()
