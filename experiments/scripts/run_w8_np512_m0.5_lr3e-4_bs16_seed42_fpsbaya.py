#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


RUN_NAME = "w8_np512_m0.5_lr3e-4_bs16_seed42_fpsbaya"
TRAIN_ARGS = [
    "-u",
    "-m",
    "src.train",
    "--data_dir",
    "D:/thresh105_qr120_umbral008_acomodado",
    "--runs_dir",
    "runs",
    "--run_name",
    "w8_np512_m0.5_lr3e-4_bs16_seed42_fpsbaya",
    "--n_points",
    "512",
    "--width",
    "8",
    "--batch_size",
    "16",
    "--lr",
    "0.0003",
    "--margin",
    "0.5",
    "--epochs",
    "200",
    "--clip_norm",
    "1.0",
    "--seed",
    "42",
    "--val_size",
    "0.15",
    "--test_size",
    "0.15",
    "--sampling",
    "fps_baya"
]


def log_line(log_file, start_time: float, message: str = "") -> None:
    elapsed = time.time() - start_time
    if message:
        line = f"[{datetime.now().strftime('%H:%M:%S')} +{elapsed:.1f}s] {message}"
    else:
        line = ""
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)

    log_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = repo_root / "experiments" / "logs" / log_date
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{RUN_NAME}.log"

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, *TRAIN_ARGS]

    with log_path.open("w", encoding="utf-8") as log_file:
        log_line(log_file, start_time, f"Iniciando run: {RUN_NAME}")
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
        log_line(log_file, start_time, f"Run finalizado. Exit code: {exit_code}")
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
