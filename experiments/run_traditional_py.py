#!/usr/bin/env python3
"""
Genera scripts Python individuales a partir de experiments.yaml para ejecutar
cada run manualmente (sin MLflow).

Uso:
    python experiments/run_traditional_py.py
    python experiments/run_traditional_py.py --only w8_np512_m0.5_lr3e-4_bs16_seed42_fpsbaya

Output:
    experiments/scripts/run_<nombre>.py
"""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

OUTPUT_DIR = _REPO_ROOT / "experiments" / "scripts"


def load_yaml_config(yaml_path: str) -> dict:
    """Carga el archivo YAML de experimentos."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML es necesario para leer experiments.yaml. Instalalo con: pip install pyyaml"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(defaults: dict, run_cfg: dict) -> dict:
    """Mezcla defaults con overrides del run. Los valores None se ignoran."""
    merged = dict(defaults)
    for key, value in run_cfg.items():
        if value is not None:
            merged[key] = value
    return merged


def build_run_name(cfg: dict) -> str:
    """Genera un nombre de run deterministico concatenando todos los hiperparametros."""
    if cfg.get("run_name"):
        return str(cfg["run_name"])

    lr_str = f"{cfg['lr']:.0e}".replace("e-0", "e-").replace("e+0", "e+").replace(".0e", "e")
    mode = "lazy" if cfg.get("lazy") else "eager"
    sampling = cfg.get("sampling", "random")
    name = (
        f"np{cfg['n_points']}"
        f"_w{cfg['width']}"
        f"_ep{cfg['epochs']}"
        f"_bs{cfg['batch_size']}"
        f"_lr{lr_str}"
        f"_m{cfg['margin']}"
        f"_{sampling}"
        f"_{mode}"
    )
    if cfg.get("save_sampled"):
        name += "_ss"
    return name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera scripts Python para ejecutar cada run manualmente.",
    )
    p.add_argument(
        "--yaml",
        type=str,
        default=str(_REPO_ROOT / "experiments" / "experiments.yaml"),
        help="Path al YAML de experimentos.",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="Nombres de runs a generar, separados por coma.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directorio de salida para los scripts.",
    )
    return p.parse_args()


def build_command(merged: dict, run_name: str) -> list[str]:
    runs_dir = merged.get("runs_dir", "runs")
    cmd = [
        "-u",
        "-m",
        "src.train",
        "--data_dir",
        str(merged["data_dir"]),
        "--runs_dir",
        str(runs_dir),
        "--run_name",
        str(run_name),
        "--n_points",
        str(merged["n_points"]),
        "--width",
        str(merged["width"]),
        "--batch_size",
        str(merged["batch_size"]),
        "--lr",
        str(merged["lr"]),
        "--margin",
        str(merged["margin"]),
        "--epochs",
        str(merged["epochs"]),
        "--clip_norm",
        str(merged["clip_norm"]),
        "--seed",
        str(merged["seed"]),
        "--val_size",
        str(merged["val_size"]),
        "--test_size",
        str(merged["test_size"]),
        "--sampling",
        str(merged.get("sampling", "random")),
    ]

    if merged.get("resume", False):
        cmd.append("--resume")

    if merged.get("lazy", False):
        cmd.append("--lazy")

    if merged.get("save_sampled", False):
        cmd.append("--save_sampled")

    if merged.get("early_stopping_patience") is not None:
        cmd.extend(["--early_stopping_patience", str(merged["early_stopping_patience"])])

    return cmd


def build_script_content(run_name: str, train_args: list[str]) -> str:
    train_args_json = json.dumps(train_args, indent=4)
    run_name_json = json.dumps(run_name)

    return f'''#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


RUN_NAME = {run_name_json}
TRAIN_ARGS = {train_args_json}


def log_line(log_file, start_time: float, message: str = "") -> None:
    elapsed = time.time() - start_time
    if message:
        line = f"[{{datetime.now().strftime('%H:%M:%S')}} +{{elapsed:.1f}}s] {{message}}"
    else:
        line = ""
    print(line, flush=True)
    log_file.write(line + "\\n")
    log_file.flush()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    log_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = repo_root / "experiments" / "logs" / log_date
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{{RUN_NAME}}.log"

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, *TRAIN_ARGS]

    with log_path.open("w", encoding="utf-8") as log_file:
        log_line(log_file, start_time, f"Iniciando run: {{RUN_NAME}}")
        log_line(log_file, start_time, "Comando: " + " ".join(cmd))
        log_line(log_file, start_time)

        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        exit_code = proc.wait()
        log_line(log_file, start_time)
        log_line(log_file, start_time, f"Run finalizado. Exit code: {{exit_code}}")
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> None:
    args = parse_args()

    config = load_yaml_config(args.yaml)
    defaults: dict = config.get("defaults", {})
    runs_cfg: list[dict] = config.get("runs", [])

    if not runs_cfg:
        runs_cfg = [{}]

    only_names = set(args.only.split(",")) if args.only else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(runs_cfg)
    generated = []

    for idx, run_cfg in enumerate(runs_cfg):
        merged = merge_config(defaults, run_cfg)
        run_name = build_run_name(merged)

        if only_names and run_name not in only_names:
            print(f"[{idx + 1}/{total}] SKIP {run_name} (no en --only)")
            continue

        script_name = f"run_{run_name}.py"
        script_path = out_dir / script_name
        content = build_script_content(run_name, build_command(merged, run_name))

        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        print(f"[{idx + 1}/{total}] {script_name}")
        generated.append(script_name)

    print(f"\n{len(generated)} scripts generados en {out_dir}/")
    print("\nPara ejecutar uno:")
    if generated:
        print(f"  python {out_dir / generated[0]}")
    print("\nPara ejecutar todos en secuencia:")
    print(f"  python {Path(__file__).name} --out {out_dir}")
    print("  python -c \"import pathlib, subprocess, sys; [sys.exit(r.returncode) for p in sorted(pathlib.Path('experiments/scripts').glob('run_*.py')) if (r := subprocess.run([sys.executable, str(p)])).returncode]\"")


if __name__ == "__main__":
    main()
