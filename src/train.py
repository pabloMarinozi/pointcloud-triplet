from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch

from src.data.io import find_ply_files, sample_point_cloud
from src.models.triplet import TripletNet
from src.pipeline.trainer import LazyTripletTrainingPipeline, TripletTrainingPipeline
from src.utils.seed import set_seed

# Cadencia de progreso al cargar nubes (cada cuántos archivos se imprime).
PROGRESS_EVERY_N_FILES = 5000


def build_all_point_clouds(ply_dir: str, n_points: int, sampling: str = "random"):
    t0 = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Buscando archivos .ply (recursivo)...", flush=True)
    files = find_ply_files(ply_dir)
    elapsed = time.perf_counter() - t0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Encontrados {len(files)} archivos .ply en {elapsed:.1f}s", flush=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sampleando nubes (n_points={n_points}, sampling={sampling})...", flush=True)
    t0 = time.perf_counter()
    all_point_clouds = []
    for i, file_path in enumerate(files):
        folder = os.path.basename(os.path.dirname(file_path))
        cloud = sample_point_cloud(file_path, n_points, sampling)
        all_point_clouds.append((folder, file_path, cloud))
        if (i + 1) % PROGRESS_EVERY_N_FILES == 0:
            elapsed = time.perf_counter() - t0
            print(f"[{datetime.now().strftime('%H:%M:%S')}]   ... {i + 1}/{len(files)} nubes cargadas ({elapsed:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Carga completa: {len(all_point_clouds)} nubes en {elapsed:.1f}s", flush=True)
    return all_point_clouds


def discover_point_clouds(ply_dir: str):
    t0 = time.perf_counter()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Descubriendo archivos .ply (solo paths)...", flush=True)
    files = find_ply_files(ply_dir)
    paths = [(os.path.basename(os.path.dirname(f)), f) for f in files]
    elapsed = time.perf_counter() - t0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Encontrados {len(paths)} paths en {elapsed:.1f}s", flush=True)
    return paths


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
        "--open_set_classes",
        type=int,
        default=0,
        help="Identidades completas reservadas como desconocidas (0 desactiva open-set).",
    )
    p.add_argument(
        "--open_set_val_size",
        type=float,
        default=0.5,
        help="Fracción de identidades desconocidas usada para calibrar el umbral.",
    )
    p.add_argument("--sampling", type=str, choices=["random", "fps", "fps_baya"], default="random", help="Estrategia de muestreo de puntos.")
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        metavar="N",
        help="Stop if val_loss does not improve for N epochs (default: disabled).",
    )
    p.add_argument("--lazy", action="store_true", help="Usar lazy loading: las nubes se leen del disco en cada __getitem__. Por defecto (--eager) se precargan todas en RAM.")
    p.add_argument("--save_sampled", action="store_true", help="Guardar los puntos sampleados como .ply en experiments/sampling/<run>+<archivo>.")

    return p.parse_args()


def main():
    t0 = time.perf_counter()

    def ts_print(msg: str) -> None:
        elapsed = time.perf_counter() - t0
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now} +{elapsed:.1f}s] {msg}", flush=True)

    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_print(f"Device: {device}")

    ts_print(f"Seed: {args.seed}")
    ts_print(f"Modo: {'lazy' if args.lazy else 'eager'}")

    if args.lazy:
        ts_print("Descubriendo archivos .ply (solo paths, sin cargar a RAM)...")
        all_point_clouds = discover_point_clouds(args.data_dir)
        PipelineClass = LazyTripletTrainingPipeline
    else:
        ts_print(f"Cargando y sampleando nubes (n_points={args.n_points}, sampling={args.sampling})...")
        all_point_clouds = build_all_point_clouds(args.data_dir, args.n_points, args.sampling)
        PipelineClass = TripletTrainingPipeline

    ts_print("Creando pipeline (splits, datasets, dataloaders, modelo)...")
    t_pipe = time.perf_counter()
    pipeline = PipelineClass(
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
        sampling=args.sampling,
        save_sampled=args.save_sampled,
        open_set_classes=args.open_set_classes,
        open_set_val_size=args.open_set_val_size,
    )
    pipe_init_s = time.perf_counter() - t_pipe
    ts_print(f"Pipeline listo en {pipe_init_s:.1f}s")
    pipeline._log(f"Pipeline listo en {pipe_init_s:.1f}s")

    ts_print("Iniciando entrenamiento...")
    pipeline._log("Iniciando entrenamiento...")
    t_train = time.perf_counter()
    pipeline.train(resume=args.resume)
    elapsed_train = time.perf_counter() - t_train
    total_elapsed = time.perf_counter() - t0
    ts_print(f"Entrenamiento finalizado: {elapsed_train:.1f}s")
    pipeline._log(f"Entrenamiento finalizado: {elapsed_train:.1f}s")
    ts_print(f"Total script (pre+train): {total_elapsed:.1f}s")
    pipeline._log(f"Total script (pre+train): {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
