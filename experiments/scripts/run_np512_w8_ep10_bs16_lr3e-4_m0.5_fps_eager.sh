#!/usr/bin/env bash
#
# Run: np512_w8_ep10_bs16_lr3e-4_m0.5_fps_eager
# Generado por experiments/run_traditional.py
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LOG_DATE=$(date +%Y-%m-%d)
mkdir -p "$REPO_ROOT/experiments/logs/$LOG_DATE"
LOGFILE="$REPO_ROOT/experiments/logs/$LOG_DATE/run_np512_w8_ep10_bs16_lr3e-4_m0.5_fps_eager.log"
SCRIPT_START=$(date +%s)
echo "[$(date '+%H:%M:%S') +0.0s] Iniciando run: np512_w8_ep10_bs16_lr3e-4_m0.5_fps_eager" | tee "$LOGFILE"
echo "" | tee -a "$LOGFILE"

export PYTHONUNBUFFERED=1
set +o pipefail
python -u -m src.train \
  --data_dir ./dataset \
  --runs_dir runs \
  --run_name np512_w8_ep10_bs16_lr3e-4_m0.5_fps_eager \
  --n_points 512 \
  --width 8 \
  --batch_size 16 \
  --lr 0.0003 \
  --margin 0.5 \
  --epochs 10 \
  --clip_norm 1.0 \
  --seed 42 \
  --val_size 0.15 \
  --test_size 0.15 \
  --sampling fps 2>&1 | tee -a "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]}
set -o pipefail

SCRIPT_END=$(date +%s)
ELAPSED=$((SCRIPT_END - SCRIPT_START))
echo "" | tee -a "$LOGFILE"
echo "[$(date '+%H:%M:%S') +${ELAPSED}.0s] Run finalizado. Exit code: $EXIT_CODE" | tee -a "$LOGFILE"
exit $EXIT_CODE
