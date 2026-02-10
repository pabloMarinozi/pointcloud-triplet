"""
Migra los runs existentes a la estructura con carpetas ep<N>/.

Para cada run en runs_dir:
  - Lee la última época de metrics.csv.
  - Crea runs/<run>/ep<N>/.
  - Copia model_best.pt -> ep<N>/model.pt.
  - Mueve reference_embeddings_*.npz a ep<N>/.
  - Mueve evaluation_report.json, evaluation/, evaluation_test/ a ep<N>/ si existen.
  - Crea last_epoch.json y model_version.json en la raíz del run.

Así los runs quedan como si hubieran sido generados con la nueva estrategia.

Uso:
  python -m scripts.migrate_runs_to_ep_folders
  python -m scripts.migrate_runs_to_ep_folders --runs_dir runs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def get_last_epoch_from_metrics(exp_dir: str) -> tuple[int | None, float]:
    """Devuelve (última_época, val_loss_última_fila) o (None, 0.0)."""
    csv_path = os.path.join(exp_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return None, 0.0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None, 0.0
        last_row = lines[-1].strip().split(",")
        if len(last_row) >= 3:
            epoch = int(float(last_row[0]))
            val_loss = float(last_row[2])
            return epoch, val_loss
        if len(last_row) >= 1:
            return int(float(last_row[0])), 0.0
    except (ValueError, IndexError):
        pass
    return None, 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Migrar runs a estructura con ep<N>/ (modelo, refs y evaluación por época)."
    )
    parser.add_argument("--runs_dir", type=str, default="runs", help="Carpeta de experimentos.")
    parser.add_argument("--dry_run", action="store_true", help="Solo mostrar qué se haría, sin escribir.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: No existe {runs_dir}")
        return 1

    run_names = [d.name for d in runs_dir.iterdir() if d.is_dir()]
    run_names.sort()

    if not run_names:
        print(f"No hay runs en {runs_dir}")
        return 0

    for run_name in run_names:
        exp_dir = runs_dir / run_name
        last_epoch, last_val_loss = get_last_epoch_from_metrics(str(exp_dir))

        if last_epoch is None:
            print(f"[SKIP] {run_name}: no se pudo leer última época de metrics.csv")
            continue

        ep_dir = exp_dir / f"ep{last_epoch}"
        if ep_dir.exists() and any(ep_dir.iterdir()):
            print(f"[SKIP] {run_name}: ya existe {ep_dir.name}/ con contenido")
            continue

        print(f"\n{run_name} -> ep{last_epoch}/")

        if args.dry_run:
            if not ep_dir.exists():
                print(f"  Crear {ep_dir.name}/")
            if (exp_dir / "model_best.pt").exists():
                print(f"  Copiar model_best.pt -> ep{last_epoch}/model.pt")
            for f in exp_dir.iterdir():
                if f.is_file() and f.name.startswith("reference_embeddings_") and f.suffix == ".npz":
                    print(f"  Mover {f.name} -> ep{last_epoch}/")
            for name in ("evaluation_report.json", "evaluation", "evaluation_test"):
                p = exp_dir / name
                if p.exists():
                    print(f"  Mover {name} -> ep{last_epoch}/")
            print(f"  Escribir last_epoch.json y model_version.json en raíz")
            continue

        ep_dir.mkdir(parents=True, exist_ok=True)

        # Copia del modelo
        model_best = exp_dir / "model_best.pt"
        if model_best.exists():
            shutil.copy2(model_best, ep_dir / "model.pt")
            print(f"  [OK] model_best.pt -> ep{last_epoch}/model.pt")

        # Mover reference_embeddings_*.npz
        for f in exp_dir.iterdir():
            if f.is_file() and f.name.startswith("reference_embeddings_") and f.suffix == ".npz":
                dest = ep_dir / f.name
                shutil.move(str(f), str(dest))
                print(f"  [OK] {f.name} -> ep{last_epoch}/")

        # Mover evaluación
        for name in ("evaluation_report.json", "evaluation", "evaluation_test"):
            p = exp_dir / name
            if p.exists():
                dest = ep_dir / name
                if p.is_dir():
                    shutil.move(str(p), str(dest))
                else:
                    shutil.move(str(p), str(dest))
                print(f"  [OK] {name} -> ep{last_epoch}/")

        # last_epoch.json y model_version.json en raíz
        last_epoch_path = exp_dir / "last_epoch.json"
        with open(last_epoch_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": last_epoch}, f, indent=2)
        print(f"  [OK] last_epoch.json (epoch={last_epoch})")

        model_version_path = exp_dir / "model_version.json"
        with open(model_version_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": last_epoch, "val_loss": last_val_loss}, f, indent=2)
        print(f"  [OK] model_version.json (epoch={last_epoch}, val_loss={last_val_loss:.6f})")

    print("\nListo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
