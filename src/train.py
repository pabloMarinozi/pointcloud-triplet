from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch

from src.data.io import find_ply_files, sample_point_cloud
from src.models.triplet import TripletNet
from src.pipeline.trainer import TripletTrainingPipeline
from src.utils.seed import set_seed

# Cadencia de progreso al cargar nubes (cada cuántos archivos se imprime).
PROGRESS_EVERY_N_FILES = 5000


def build_all_point_clouds(ply_dir: str, n_points: int):
    """
    Replica el bloque del Colab:
    - encuentra PLYs recursivamente
    - define clase = nombre de la carpeta contenedora
    - samplea n_points por nube
    Retorna lista: (folder, file_path, cloud_np)
    """
    t0 = time.perf_counter()
    print("[PROGRESO] Buscando archivos .ply (recursivo)...", flush=True)
    files = find_ply_files(ply_dir)
    elapsed = time.perf_counter() - t0
    print(f"[PROGRESO] Encontrados {len(files)} archivos .ply en {elapsed:.1f}s", flush=True)

    print(f"[PROGRESO] Cargando y sampleando nubes (n_points={n_points})...", flush=True)
    t0 = time.perf_counter()
    all_point_clouds = []
    for i, file_path in enumerate(files):
        folder = os.path.basename(os.path.dirname(file_path))
        cloud = sample_point_cloud(file_path, n_points)
        all_point_clouds.append((folder, file_path, cloud))
        if (i + 1) % PROGRESS_EVERY_N_FILES == 0:
            elapsed = time.perf_counter() - t0
            print(f"  ... {i + 1}/{len(files)} nubes cargadas ({elapsed:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[PROGRESO] Carga lista: {len(all_point_clouds)} nubes en {elapsed:.1f}s", flush=True)
    return all_point_clouds


def parse_args():
    p = argparse.ArgumentParser(description="Train TripletNet on 3D point clouds (.ply).")
    p.add_argument("--data_dir", type=str, required=True, help="Directorio raíz con .ply (recursivo).")
    p.add_argument("--runs_dir", type=str, default="runs", help="Donde guardar experimentos.")
    p.add_argument("--run_name", type=str, default=None, help="Nombre determinístico del run (si no se da, usa timestamp).")
    p.add_argument("--resume", action="store_true", help="Reanudar entrenamiento desde checkpoint_last.pt si existe.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n_points", type=int, default=1024)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--val_size", type=float, default=0.15, help="Val ratio (train/val/test = 70/15/15 por defecto).")
    p.add_argument("--test_size", type=float, default=0.15, help="Test ratio.")
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        metavar="N",
        help="Stop if val_loss does not improve for N epochs (default: disabled).",
    )

    return p.parse_args()


def main():
    t0 = time.perf_counter()
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PROGRESO] Device: {device}", flush=True)

    all_point_clouds = build_all_point_clouds(args.data_dir, args.n_points)

    print("[PROGRESO] Creando pipeline (directorio, splits, datasets, dataloaders, modelo)...", flush=True)
    t_pipe = time.perf_counter()
    pipeline = TripletTrainingPipeline(
        all_point_clouds=all_point_clouds,
        model_class=TripletNet,
        n_points=args.n_points,
        width=args.width,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        epochs=args.epochs,
        clip_norm=args.clip_norm,
        seed=args.seed,
        device=device,
        runs_dir=args.runs_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        run_name=args.run_name,
        early_stopping_patience=args.early_stopping_patience,
    )
    print(f"[PROGRESO] Pipeline listo en {time.perf_counter() - t_pipe:.1f}s", flush=True)

    print("[PROGRESO] Iniciando entrenamiento...", flush=True)
    pipeline.train(resume=args.resume)
    print(f"[PROGRESO] Total pre+train: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
