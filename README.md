# RFO Demo

A package-oriented Rectified Flow Operator repository with internalized data, model, operator, visualization, and experiment modules.

## Highlights

- Single package namespace: `rfo_demo`
- Unified CLI entrypoint: `python -m rfo_demo.cli`
- No legacy `main/`, `script/`, `scorenet/`, or `rectified/` directories
- Built-in `tests/` and `examples/` for quick validation and onboarding

## Repository Layout

```text
RFO_Demo/
├── examples/
├── tests/
├── rfo_demo/
│   ├── cli.py
│   ├── configs/
│   ├── core/
│   ├── data/
│   ├── experiments/
│   ├── models/
│   ├── operators/
│   ├── optim/
│   ├── utils/
│   └── visualization/
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment:

```bash
python -m rfo_demo.cli --case burger --config rfo_demo/configs/burger.py
```

Run examples:

```bash
python examples/check_shapes.py
python examples/run_burger.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Notes

- Existing dataset paths and output paths are intentionally preserved.
- Config files remain Python files to avoid changing current experiment behavior.
- The project now treats `rfo_demo` as the only source package boundary.
