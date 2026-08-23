# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/`. Training starts at `src/train.py`, evaluation at `src/eval.py`, and responsibilities are separated into `data/`, `models/`, `pipeline/`, `evaluation/`, and `utils/`. Experiment definitions and launchers live in `experiments/`; one-off analysis, migration, and conversion utilities belong in `scripts/`. Project notes are under `docs/`. Runtime artifacts such as checkpoints, metrics, and evaluation exports are written to `runs/` and are not source files. External point-cloud datasets should contain identity-named directories of `.ply` files.

## Build, Test, and Development Commands

Create a Python 3.10+ virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training with `python -m src.train --data_dir ./dataset_ply --n_points 512 --width 8 --epochs 30`. Evaluate with `python -m src.eval --data_dir ./dataset_ply --run latest --export_csv --seed 42`. Run configured experiments with `python experiments/run_traditional.py`; inspect `experiments/experiments.yaml` before long GPU jobs. Use `python -m compileall src scripts experiments` as a fast syntax check.

## Coding Style & Naming Conventions

Use four-space indentation and conventional PEP 8 naming: `snake_case` for functions, modules, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Add type hints to public or non-obvious interfaces and keep CLI entry points behind `if __name__ == "__main__":`. Preserve the existing package imports (`from src...`) and favor small modules grouped by responsibility. No formatter or linter is configured, so keep imports ordered and changes stylistically consistent with nearby code.

## Testing Guidelines

There is no committed automated test suite or coverage threshold. For data or sampling changes, run one epoch on a reduced dataset and verify determinism with `--seed 42`. For evaluation changes, exercise the relevant `--split` and confirm outputs beneath `runs/<run>/ep<N>/`. New unit tests should use `pytest`, live in `tests/`, and follow `test_<module>.py` and `test_<behavior>()` naming.

## Commit & Pull Request Guidelines

Recent history favors short, imperative subjects such as `Save epoch snapshots in folders`; scoped prefixes like `fix:` and `scripts:` are also accepted. Keep commits focused and avoid committing datasets, checkpoints, generated run outputs, or temporary logs. Pull requests should explain the behavioral change, list commands used to validate it, link related issues or experiment notes, and include representative metrics or output paths when training/evaluation behavior changes. Call out configuration, memory, or GPU-runtime impacts explicitly.
