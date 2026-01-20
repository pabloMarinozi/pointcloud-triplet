from __future__ import annotations

import os
from typing import List

import numpy as np
import open3d as o3d


def find_ply_files(ply_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, filenames in os.walk(ply_dir):
        for f in filenames:
            if f.lower().endswith(".ply"):
                files.append(os.path.join(root, f))
    return files


def sample_point_cloud(file_path: str, n_points: int) -> np.ndarray:
    """
    Lee un .ply con Open3D y devuelve exactamente n_points (con/sin reemplazo).
    Retorna (n_points, 3) float32.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    pts = np.asarray(pcd.points)

    if pts.shape[0] >= n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
    else:
        idx = np.random.choice(len(pts), n_points, replace=True)

    return pts[idx].astype(np.float32)
