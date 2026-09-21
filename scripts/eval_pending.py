"""Evalúa secuencialmente los runs que todavía no tienen evaluation_report.json.

Uso (desde la raíz del repo, con el venv activado):

    python scripts/eval_pending.py

Por defecto evalúa la lista `RUNS_TO_EVAL` sobre `DATA_DIR` con `--split both`.
Edita esas constantes si necesitás cambiar el dataset, el split o la lista de runs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DATA_DIR = "D:/thresh105_qr120_umbral008_acomodado"
RUNS_DIR = "runs"
SPLIT = "both"  # val | test | both | select_and_test

RUNS_TO_EVAL = [
    # "w16_np512_m0.5_lr3e-4_bs16_seed42_fps",
    # "w32_np512_m0.5_lr3e-4_bs16_seed42_fps",
    # "w8_np512_m0.5_lr1e-3_bs16_seed42_fps",
    "w8_np512_m0.5_lr1e-4_bs16_seed42_fps",
    "w8_np512_m0.5_lr5e-4_bs16_seed42_fps",
]


def write_line(log_file, message: str) -> None:
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent

    now = datetime.now()
    log_dir = project_root / "experiments" / "logs" / now.strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"eval_pending_{now.strftime('%H-%M-%S')}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    failures: list[tuple[str, int]] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        write_line(log_file, f"Inicio: {now.isoformat(timespec='seconds')}")
        write_line(log_file, f"Runs a evaluar: {len(RUNS_TO_EVAL)}")
        write_line(log_file, f"Data dir: {DATA_DIR}")
        write_line(log_file, f"Split: {SPLIT}")

        for index, run_name in enumerate(RUNS_TO_EVAL, start=1):
            write_line(
                log_file,
                f"\n[{index}/{len(RUNS_TO_EVAL)}] Evaluando {run_name}...",
            )

            cmd = [
                sys.executable,
                "-u",
                "-m",
                "src.eval",
                "--data_dir",
                DATA_DIR,
                "--runs_dir",
                RUNS_DIR,
                "--run",
                run_name,
                "--split",
                SPLIT,
            ]
            write_line(log_file, "Comando: " + " ".join(cmd))

            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()
            if return_code != 0:
                failures.append((run_name, return_code))
                write_line(
                    log_file,
                    f"[ERROR] {run_name} terminó con código {return_code}.",
                )
                write_line(log_file, "Continuando con la siguiente evaluación...")
                continue

            write_line(log_file, f"[OK] {run_name}")

        if failures:
            write_line(log_file, "\nEvaluaciones con errores:")
            for run_name, return_code in failures:
                write_line(log_file, f"- {run_name}: código {return_code}")
        else:
            write_line(log_file, "\nTodas las evaluaciones terminaron correctamente.")
        write_line(log_file, f"Log general: {log_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
