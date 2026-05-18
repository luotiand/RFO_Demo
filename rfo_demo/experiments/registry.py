from rfo_demo.experiments.advection import run as run_advection
from rfo_demo.experiments.burger import run as run_burger
from rfo_demo.experiments.darcy import run as run_darcy
from rfo_demo.experiments.ns import run as run_ns

EXPERIMENTS = {
    "advection": run_advection,
    "burger": run_burger,
    "darcy": run_darcy,
    "ns": run_ns,
}



def run_experiment(case: str, config: dict) -> None:
    try:
        runner = EXPERIMENTS[case]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown experiment '{case}'. Available: {available}") from exc
    runner(config)
