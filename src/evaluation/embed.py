from __future__ import annotations

import numpy as np
import open3d as o3d
import torch

from src.data.dataset import normalize_unit_sphere, sample_n, augment


def read_points_from_ply(ply_path: str) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(cloud.points, dtype=np.float32)
    return pts


def get_embedding_from_model(model, x: torch.Tensor) -> torch.Tensor:
    """
    Compatibilidad: TripletNet tiene .embed() en nuestro refactor.
    """
    if hasattr(model, "embed"):
        return model.embed(x)
    if hasattr(model, "forward_once"):
        return model.forward_once(x)
    za, _, _ = model(x, x, x)
    return za


def embed_point_cloud_path(
    model,
    ply_path: str,
    n_points: int,
    device: torch.device,
    use_augmentation: bool = False,
) -> np.ndarray:
    """
    Retorna embedding 1D (emb_dim,).
    """
    pts = read_points_from_ply(ply_path)
    pts = normalize_unit_sphere(pts).astype(np.float32)

    pts_proc = augment(pts, n_points) if use_augmentation else sample_n(pts, n_points)
    x = torch.from_numpy(pts_proc.T).unsqueeze(0).float().to(device)

    with torch.no_grad():
        emb = get_embedding_from_model(model, x).squeeze(0).detach().cpu().numpy()

    return emb
