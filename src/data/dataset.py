from __future__ import annotations

import random
from math import cos, sin, pi
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import open3d as o3d


def to_numpy(cloud: Any) -> np.ndarray:
    if isinstance(cloud, o3d.geometry.PointCloud):
        return np.asarray(cloud.points, dtype=np.float32)
    return np.asarray(cloud, dtype=np.float32)


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    pts = points - points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    return pts / (scale + 1e-8)


def sample_n(points: np.ndarray, n_points: int) -> np.ndarray:
    n = len(points)
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
    else:
        idx = np.random.choice(n, n_points, replace=True)
    return points[idx]


def augment(points: np.ndarray, n_points: int) -> np.ndarray:
    theta = np.random.uniform(-pi, pi)
    R = np.array(
        [[cos(theta), -sin(theta), 0],
         [sin(theta),  cos(theta), 0],
         [0,           0,          1]],
        dtype=np.float32
    )
    pts = points @ R.T

    # escalado
    s = np.float32(np.random.uniform(0.8, 1.25))
    pts *= s

    # jitter
    noise = np.clip(np.random.normal(0, 0.01, pts.shape), -0.05, 0.05).astype(np.float32)
    pts += noise

    # dropout de puntos
    keep = max(1, int(len(pts) * np.random.uniform(0.9, 1.0)))
    idx = np.random.choice(len(pts), keep, replace=False)
    pts = pts[idx]

    return sample_n(pts, n_points)


PointCloudItem = Tuple[str, str, np.ndarray]  # (folder/class, file_path, cloud_np)


class TripletPointCloudDataset(Dataset):
    """
    Dataset que devuelve (anchor, positive, negative) con forma (3, N) cada uno.
    Basado 1:1 en el Colab.
    """

    def __init__(self, all_point_clouds: List[PointCloudItem], n_points: int, train: bool = True):
        self.n_points = n_points
        self.train = train
        self.items: List[Tuple[str, np.ndarray]] = []  # (cls, pts_norm)
        self.class_to_indices: Dict[str, List[int]] = {}

        for idx, (folder, _, cloud) in enumerate(all_point_clouds):
            pts = normalize_unit_sphere(to_numpy(cloud)).astype(np.float32)
            self.items.append((folder, pts))
            self.class_to_indices.setdefault(folder, []).append(idx)

        valid_classes = {cls: idxs for cls, idxs in self.class_to_indices.items() if len(idxs) >= 2}
        if len(valid_classes) < 2:
            raise ValueError(f"Need at least 2 classes with 2+ samples each. Got {len(valid_classes)}")

        valid_indices = set()
        for idxs in valid_classes.values():
            valid_indices.update(idxs)

        self.items = [self.items[i] for i in range(len(self.items)) if i in valid_indices]

        self.class_to_indices = {}
        for new_idx, (cls, pts) in enumerate(self.items):
            self.class_to_indices.setdefault(cls, []).append(new_idx)

    def __len__(self) -> int:
        return len(self.items)

    def _get_positive(self, cls: str, avoid_idx: int) -> int | None:
        idxs = self.class_to_indices[cls]
        candidates = [i for i in idxs if i != avoid_idx]
        if not candidates:
            return None

        if self.train:
            return random.choice(candidates)
        else:
            pos_idx = (idxs.index(avoid_idx) + 1) % len(idxs)
            return idxs[pos_idx]

    def _get_negative(self, cls: str) -> int:
        other_classes = [c for c in self.class_to_indices if c != cls]
        if not other_classes:
            raise ValueError(f"No other class available for negative! Classes: {list(self.class_to_indices.keys())}")
        neg_cls = random.choice(other_classes)
        return random.choice(self.class_to_indices[neg_cls])

    def __getitem__(self, index: int):
        if index >= len(self.items):
            index = index % len(self.items)

        file_a, pts_a = self.items[index]
        idx_p = self._get_positive(file_a, index)
        if idx_p is None:
            return self.__getitem__((index + 1) % len(self))

        idx_n = self._get_negative(file_a)

        _, pts_p = self.items[idx_p]
        _, pts_n = self.items[idx_n]

        if self.train:
            pa = augment(pts_a, self.n_points)
            pp = augment(pts_p, self.n_points)
            pn = augment(pts_n, self.n_points)
        else:
            pa = sample_n(pts_a, self.n_points)
            pp = sample_n(pts_p, self.n_points)
            pn = sample_n(pts_n, self.n_points)

        # (N, 3) -> (3, N)
        return (
            torch.from_numpy(pa.T).float(),
            torch.from_numpy(pp.T).float(),
            torch.from_numpy(pn.T).float(),
        )
