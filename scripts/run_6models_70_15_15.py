"""
Entrena 6 modelos con split 70/15/15 (train/val/test), 200 épocas fijas (sin early stopping).

Configuraciones:
  - width: 8, 16, 32
  - n_points: 512
  - lr: 1e-3, 3e-4
  - margin=0.5, batch_size=16, seed=42

Run names: w8_np512_m0.5_lr1e-3_bs16_seed42, w8_np512_m0.5_lr3e-4_bs16_seed42, ...

Uso:
  python -m scripts.run_6models_70_15_15 --data_dir D:\path\to\ply
  python -m scripts.run_6models_70_15_15 --data_dir D:\path\to\ply --runs_dir runs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EPOCHS = 200
N_POINTS = 512
MARGIN = 0.5
BATCH_SIZE = 16
SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15

EXPERIMENTS = [
    {"width": 8, "lr": 1e-3},
    {"width": 8, "lr": 3e-4},
    {"width": 16, "lr": 1e-3},
    {"width": 16, "lr": 3e-4},
    {"width": 32, "lr": 1e-3},
    {"width": 32, "lr": 3e-4},
]


def run_name(width: int, lr: float) -> str:
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+").replace(".0e", "e")
    return f"w{width}_np{N_POINTS}_m{MARGIN}_lr{lr_str}_bs{BATCH_SIZE}_seed{SEED}"


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar 6 modelos (w8/16/32, np512, lr 1e-3 y 3e-4) con split 70/15/15."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directorio raíz con .ply (recursivo).")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Donde guardar experimentos.")
    args = parser.parse_args()

    runs_dir = str(Path(args.runs_dir).resolve())
    data_dir = str(Path(args.data_dir).resolve())

    print("Split: 70% train, 15% val, 15% test")
    print(f"Épocas: {EPOCHS} (sin early stopping)")
    print("=" * 80)

    for i, exp in enumerate(EXPERIMENTS):
        name = run_name(exp["width"], exp["lr"])
        print(f"\n[{i + 1}/{len(EXPERIMENTS)}] {name}")
        print("-" * 80)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.train",
            "--data_dir",
            data_dir,
            "--runs_dir",
            runs_dir,
            "--run_name",
            name,
            "--resume",
            "--n_points",
            str(N_POINTS),
            "--width",
            str(exp["width"]),
            "--lr",
            str(exp["lr"]),
            "--margin",
            str(MARGIN),
            "--batch_size",
            str(BATCH_SIZE),
            "--epochs",
            str(EPOCHS),
            "--val_size",
            str(VAL_SIZE),
            "--test_size",
            str(TEST_SIZE),
            "--seed",
            str(SEED),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[FAIL] {name} terminó con código {result.returncode}.", file=sys.stderr)
            sys.exit(result.returncode)

    print("\n" + "=" * 80)
    print("[OK] Los 6 modelos terminaron.")
    print("Evaluación en val: python -m src.eval --data_dir ... --runs_dir ... --run all --export_csv")
    print("Para métrica final en test hay que evaluar contra splits/test_paths.json (no implementado en eval aún).")
    print("=" * 80)


if __name__ == "__main__":
    main()
