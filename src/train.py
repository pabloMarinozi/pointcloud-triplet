from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch

from src.data.io import find_ply_files, sample_point_cloud
from src.models.triplet import TripletNet
from src.pipeline.trainer import TripletTrainingPipeline
from src.utils.seed import set_seed


def build_all_point_clouds(ply_dir: str, n_points: int):
    """
    Replica el bloque del Colab:
    - encuentra PLYs recursivamente
    - define clase = nombre de la carpeta contenedora
    - samplea n_points por nube
    Retorna lista: (folder, file_path, cloud_np)
    """
    files = find_ply_files(ply_dir)
    print(f"Total .ply files found: {len(files)}")

    all_point_clouds = []
    for file_path in files:
        folder = os.path.basename(os.path.dirname(file_path))
        cloud = sample_point_cloud(file_path, n_points)
        all_point_clouds.append((folder, file_path, cloud))

    print(f"Total point clouds loaded: {len(all_point_clouds)}")
    return all_point_clouds


def parse_args():
    p = argparse.ArgumentParser(description="Train TripletNet on 3D point clouds (.ply).")
    p.add_argument("--data_dir", type=str, required=True, help="Directorio raíz con .ply (recursivo).")
    p.add_argument("--runs_dir", type=str, default="runs", help="Donde guardar experimentos.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n_points", type=int, default=1024)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--val_size", type=float, default=0.2)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_point_clouds = build_all_point_clouds(args.data_dir, args.n_points)

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
    )

    pipeline.train()


if __name__ == "__main__":
    main()
