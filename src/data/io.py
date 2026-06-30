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


BAYA_SIZE = 500


def _fps_from_bayas(pts: np.ndarray, n_points: int, baya_size: int = BAYA_SIZE) -> np.ndarray:
    n = len(pts)
    if n < n_points:
        idx = np.random.choice(n, n_points, replace=True)
        return pts[idx].astype(np.float32)

    n_bayas = n // baya_size
    if n_bayas == 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return np.asarray(pcd.farthest_point_down_sample(n_points).points, dtype=np.float32)

    base = n_points // n_bayas
    remainder = n_points % n_bayas
    takes = [
        min(base + 1, baya_size) if i < remainder else min(base, baya_size)
        for i in range(n_bayas)
    ]

    collected = []
    for i in range(n_bayas):
        take = takes[i]
        if take <= 0:
            continue
        start = i * baya_size
        end = min((i + 1) * baya_size, n)
        baya = pts[start:end]
        if len(baya) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(baya)
        if take < len(baya):
            pcd = pcd.farthest_point_down_sample(take)
        collected.append(np.asarray(pcd.points, dtype=np.float32))

    merged = np.concatenate(collected, axis=0)
    if len(merged) < n_points:
        idx = np.random.choice(len(merged), n_points - len(merged), replace=True)
        merged = np.concatenate([merged, merged[idx]], axis=0)
    return merged[:n_points].astype(np.float32)


def sample_point_cloud(file_path: str, n_points: int, sampling: str = "random") -> np.ndarray:
    """
    Lee un .ply con Open3D y devuelve exactamente n_points (con/sin reemplazo).
    sampling: "random" | "fps" | "fps_baya"
    Retorna (n_points, 3) float32.
    """
    pcd = o3d.io.read_point_cloud(file_path)

    if sampling == "fps":
        pts = np.asarray(pcd.points)
        if len(pts) >= n_points:
            idx = np.random.permutation(len(pts))
            pcd.points = o3d.utility.Vector3dVector(pts[idx])
            pcd = pcd.farthest_point_down_sample(n_points)
            return np.asarray(pcd.points, dtype=np.float32)
        else:
            idx = np.random.choice(len(pts), n_points, replace=True)
            return pts[idx].astype(np.float32)

    pts = np.asarray(pcd.points)

    if sampling == "fps_baya":
        idx = np.random.permutation(len(pts))
        return _fps_from_bayas(pts[idx], n_points)

    if pts.shape[0] >= n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
    else:
        idx = np.random.choice(len(pts), n_points, replace=True)

    return pts[idx].astype(np.float32)
