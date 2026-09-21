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


def sample_n(
    points: np.ndarray,
    n_points: int,
    sampling: str = "random",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    n = len(points)
    random_source = rng if rng is not None else np.random

    if n >= n_points and sampling == "fps_baya":
        from src.data.io import _fps_from_bayas
        return _fps_from_bayas(points, n_points)

    if sampling == "fps" and n >= n_points:
        idx = random_source.permutation(n)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[idx])
        return np.asarray(pcd.farthest_point_down_sample(n_points).points, dtype=np.float32)

    if n >= n_points:
        idx = random_source.choice(n, n_points, replace=False)
    else:
        idx = random_source.choice(n, n_points, replace=True)
    return points[idx]


def augment(
    points: np.ndarray,
    n_points: int,
    sampling: str = "random",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    random_source = rng if rng is not None else np.random
    theta = random_source.uniform(-pi, pi)
    R = np.array(
        [[cos(theta), -sin(theta), 0],
         [sin(theta),  cos(theta), 0],
         [0,           0,          1]],
        dtype=np.float32
    )
    pts = points @ R.T

    s = np.float32(random_source.uniform(0.8, 1.25))
    pts *= s

    noise = np.clip(random_source.normal(0, 0.01, pts.shape), -0.05, 0.05).astype(np.float32)
    pts += noise

    keep = max(1, int(len(pts) * random_source.uniform(0.9, 1.0)))
    idx = random_source.choice(len(pts), keep, replace=False)
    pts = pts[idx]

    return sample_n(pts, n_points, sampling, rng=rng)


PointCloudItem = Tuple[str, str, np.ndarray]  # (folder/class, file_path, cloud_np)
LazyPointCloudItem = Tuple[str, str]  # (folder/class, file_path)


class TripletPointCloudDataset(Dataset):
    """
    Dataset que devuelve (anchor, positive, negative) con forma (3, N) cada uno.
    Basado 1:1 en el Colab.
    """

    def __init__(self, all_point_clouds: List[PointCloudItem], n_points: int, train: bool = True, sampling: str = "random"):
        self.n_points = n_points
        self.train = train
        self.sampling = sampling
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
            pa = augment(pts_a, self.n_points, self.sampling)
            pp = augment(pts_p, self.n_points, self.sampling)
            pn = augment(pts_n, self.n_points, self.sampling)
        else:
            pa = sample_n(pts_a, self.n_points, self.sampling)
            pp = sample_n(pts_p, self.n_points, self.sampling)
            pn = sample_n(pts_n, self.n_points, self.sampling)

        # (N, 3) -> (3, N)
        return (
            torch.from_numpy(pa.T).float(),
            torch.from_numpy(pp.T).float(),
            torch.from_numpy(pn.T).float(),
        )


class LazyTripletPointCloudDataset(Dataset):

    def __init__(self, all_point_clouds: List[LazyPointCloudItem], n_points: int, train: bool = True, sampling: str = "random"):
        self.n_points = n_points
        self.train = train
        self.sampling = sampling
        self.items: List[Tuple[str, str]] = []
        self.class_to_indices: Dict[str, List[int]] = {}

        for idx, (folder, path) in enumerate(all_point_clouds):
            self.items.append((folder, path))
            self.class_to_indices.setdefault(folder, []).append(idx)

        valid_classes = {cls: idxs for cls, idxs in self.class_to_indices.items() if len(idxs) >= 2}
        if len(valid_classes) < 2:
            raise ValueError(f"Need at least 2 classes with 2+ samples each. Got {len(valid_classes)}")

        valid_indices = set()
        for idxs in valid_classes.values():
            valid_indices.update(idxs)

        self.items = [self.items[i] for i in range(len(self.items)) if i in valid_indices]

        self.class_to_indices = {}
        for new_idx, (cls, path) in enumerate(self.items):
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

    def _load_from_disk(self, path: str) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return normalize_unit_sphere(pts)

    def __getitem__(self, index: int):
        if index >= len(self.items):
            index = index % len(self.items)

        file_a, path_a = self.items[index]
        idx_p = self._get_positive(file_a, index)
        if idx_p is None:
            return self.__getitem__((index + 1) % len(self))

        idx_n = self._get_negative(file_a)

        _, path_p = self.items[idx_p]
        _, path_n = self.items[idx_n]

        pts_a = self._load_from_disk(path_a)
        pts_p = self._load_from_disk(path_p)
        pts_n = self._load_from_disk(path_n)

        sa = sample_n(pts_a, self.n_points, self.sampling)
        sp = sample_n(pts_p, self.n_points, self.sampling)
        sn = sample_n(pts_n, self.n_points, self.sampling)

        if self.train:
            pa = augment(sa, self.n_points, self.sampling)
            pp = augment(sp, self.n_points, self.sampling)
            pn = augment(sn, self.n_points, self.sampling)
        else:
            pa = sa
            pp = sp
            pn = sn

        return (
            torch.from_numpy(pa.T).float(),
            torch.from_numpy(pp.T).float(),
            torch.from_numpy(pn.T).float(),
        )
