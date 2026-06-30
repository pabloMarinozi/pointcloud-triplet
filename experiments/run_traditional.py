#!/usr/bin/env python3
"""
Genera scripts bash individuales a partir de experiments.yaml para ejecutar
cada run manualmente (sin MLflow).

Uso:
    python experiments/run_traditional.py                    # genera todos
    python experiments/run_traditional.py --only w8_lr3e-4  # filtra por nombre

Output:
    experiments/scripts/run_<nombre>.sh
"""

from __future__ import annotations

import argparse
import os
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
    """Genera un nombre de run determinístico concatenando todos los hiperparámetros."""
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
        description="Genera scripts bash para ejecutar cada run manualmente.",
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


def sh_quote(value) -> str:
    """Escapa un valor para shell script."""
    return str(value)


def build_cmd_lines(merged: dict, run_name: str) -> str:
    """Construye el contenido de un script bash a partir de la config mergeada."""
    runs_dir = merged.get("runs_dir", "runs")

    parts = [
        "#!/usr/bin/env bash",
        "#",
        f"# Run: {run_name}",
        "# Generado por experiments/run_traditional.py",
        "#",
        "set -euo pipefail",
        "",
        "REPO_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/../..\" && pwd)\"",
        "cd \"$REPO_ROOT\"",
        "",
        "LOG_DATE=$(date +%Y-%m-%d)",
        "mkdir -p \"$REPO_ROOT/experiments/logs/$LOG_DATE\"",
        f"LOGFILE=\"$REPO_ROOT/experiments/logs/$LOG_DATE/run_{run_name}.log\"",
        "SCRIPT_START=$(date +%s)",
        f"echo \"[$(date '+%H:%M:%S') +0.0s] Iniciando run: {run_name}\" | tee \"$LOGFILE\"",
        "echo \"\" | tee -a \"$LOGFILE\"",
        "",
        "export PYTHONUNBUFFERED=1",
        "set +o pipefail",
        "python -u -m src.train \\",
        f"  --data_dir {sh_quote(merged['data_dir'])} \\",
        f"  --runs_dir {sh_quote(runs_dir)} \\",
        f"  --run_name {sh_quote(run_name)} \\",
        f"  --n_points {sh_quote(merged['n_points'])} \\",
        f"  --width {sh_quote(merged['width'])} \\",
        f"  --batch_size {sh_quote(merged['batch_size'])} \\",
        f"  --lr {sh_quote(merged['lr'])} \\",
        f"  --margin {sh_quote(merged['margin'])} \\",
        f"  --epochs {sh_quote(merged['epochs'])} \\",
        f"  --clip_norm {sh_quote(merged['clip_norm'])} \\",
        f"  --seed {sh_quote(merged['seed'])} \\",
        f"  --val_size {sh_quote(merged['val_size'])} \\",
        f"  --test_size {sh_quote(merged['test_size'])} \\",
        f"  --sampling {sh_quote(merged.get('sampling', 'random'))} \\",
    ]

    if merged.get("resume", False):
        parts.append("  --resume \\")

    if merged.get("lazy", False):
        parts.append("  --lazy \\")

    if merged.get("save_sampled", False):
        parts.append("  --save_sampled \\")

    if merged.get("early_stopping_patience") is not None:
        parts.append(f"  --early_stopping_patience {sh_quote(merged['early_stopping_patience'])} \\")

    # Pegar el pipe a tee en la misma línea del último flag
    if parts[-1].endswith(" \\"):
        parts[-1] = parts[-1][:-2] + ' 2>&1 | tee -a "$LOGFILE"'
    else:
        parts[-1] = parts[-1] + ' 2>&1 | tee -a "$LOGFILE"'

    parts.extend([
        "EXIT_CODE=${PIPESTATUS[0]}",
        "set -o pipefail",
        "",
        "SCRIPT_END=$(date +%s)",
        "ELAPSED=$((SCRIPT_END - SCRIPT_START))",
        "echo \"\" | tee -a \"$LOGFILE\"",
        "echo \"[$(date '+%H:%M:%S') +${ELAPSED}.0s] Run finalizado. Exit code: $EXIT_CODE\" | tee -a \"$LOGFILE\"",
        "exit $EXIT_CODE",
        "",
    ])

    return "\n".join(parts)


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

        script_name = f"run_{run_name}.sh"
        script_path = out_dir / script_name
        content = build_cmd_lines(merged, run_name)

        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        print(f"[{idx + 1}/{total}] {script_name}")
        generated.append(script_name)

    print(f"\n{len(generated)} scripts generados en {out_dir}/")
    print("\nPara ejecutar uno:")
    if generated:
        print(f"  bash {out_dir / generated[0]}")
    print("\nPara ejecutar todos en secuencia:")
    print(f"  for f in {out_dir}/run_*.sh; do echo \"=== \\$f ===\"; bash \"\\$f\" || break; done")


if __name__ == "__main__":
    main()
