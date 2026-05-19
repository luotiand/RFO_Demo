# RFO_Demo

`RFO_Demo` is a Rectified Flow Operator repository for PDE and operator-learning experiments.

Supported cases:

- `advection`
- `burger`
- `darcy`
- `ns`

## Entry Point

Run experiments through the unified CLI:

```bash
python -m rfo_demo.cli --case <case> --config <config_path>
```

Examples:

```bash
python -m rfo_demo.cli --case burger --config rfo_demo/configs/burger.py
python -m rfo_demo.cli --case darcy --config rfo_demo/configs/darcy.py
python -m rfo_demo.cli --case ns --config rfo_demo/configs/ns.py
```

Main entry files:

- `rfo_demo/cli.py`
- `rfo_demo/experiments/registry.py`

## Installation

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Configuration

Configuration files are located in `rfo_demo/configs/`.

Each config is a Python `CONFIG` dictionary that defines training and runtime settings such as:

- output paths
- checkpoint paths
- training epochs
- learning rate
- batch size
- target resolution
- model class
- flow operator

Available configs:

- `rfo_demo/configs/advection.py`
- `rfo_demo/configs/burger.py`
- `rfo_demo/configs/darcy.py`
- `rfo_demo/configs/ns.py`

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

## Source Guide

If you want to read the code, start here:

1. `rfo_demo/cli.py`
2. `rfo_demo/experiments/registry.py`
3. `rfo_demo/experiments/*.py`
4. `rfo_demo/configs/*.py`
5. `rfo_demo/models/*.py`
6. `rfo_demo/operators/rectified_flow.py`

## Examples And Tests

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

- Some configs use absolute dataset paths. Check them before running.
- Full experiments require dependencies such as `torch`, `h5py`, `scipy`, and `matplotlib`.
