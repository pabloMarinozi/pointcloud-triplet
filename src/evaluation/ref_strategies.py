"""
Estrategias de reference embeddings (centroides, multiprototipo, etc.).
Usado por src.eval para generar estrategias de referencia si faltan (--ref_strategy all).
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from src.evaluation.embed import embed_point_cloud_path

SEED = 42
STRATEGY_REF_BASENAME = "reference_embeddings_{strategy}.npz"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n > 1e-12, n, 1.0)
    return (x / n).astype(np.float32)


def build_references_centroid(
    class_to_embs: Dict[str, List[np.ndarray]], n_samples: int | None, shuffle: bool = True
) -> Dict[str, np.ndarray]:
    """Un vector por clase = media de hasta n_samples embeddings (o todos si n_samples is None)."""
    refs = {}
    rng = random.Random(SEED)
    for cls, embs in class_to_embs.items():
        arr = np.array(embs)
        if len(arr) == 0:
            continue
        idx = list(range(len(arr)))
        if shuffle:
            rng.shuffle(idx)
        if n_samples is not None:
            idx = idx[: min(n_samples, len(idx))]
        refs[cls] = arr[idx].mean(axis=0).astype(np.float32)
    return refs


def build_references_centroid_l2norm(
    class_to_embs: Dict[str, List[np.ndarray]], n_samples: int
) -> Dict[str, np.ndarray]:
    """Media de embeddings L2-normalizados, luego L2-normalizar el resultado."""
    refs = {}
    rng = random.Random(SEED)
    for cls, embs in class_to_embs.items():
        arr = np.array(embs)
        if len(arr) == 0:
            continue
        idx = list(range(len(arr)))
        rng.shuffle(idx)
        idx = idx[: min(n_samples, len(idx))]
        normalized = _l2_normalize(arr[idx])
        mean_emb = normalized.mean(axis=0)
        refs[cls] = _l2_normalize(mean_emb.reshape(1, -1)).reshape(-1)
    return refs


def build_references_multiprototype(
    class_to_embs: Dict[str, List[np.ndarray]], k: int
) -> Dict[str, np.ndarray]:
    """Varios prototipos por clase con k-means (k = min(k, n_samples))."""
    refs = {}
    for cls, embs in class_to_embs.items():
        arr = np.array(embs, dtype=np.float32)
        if len(arr) == 0:
            continue
        n = len(arr)
        n_clusters = min(k, n)
        if n_clusters == 1:
            refs[cls] = arr.mean(axis=0, keepdims=True)
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        kmeans.fit(arr)
        refs[cls] = kmeans.cluster_centers_.astype(np.float32)
    return refs


def build_references_all_train(
    class_to_embs: Dict[str, List[np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Todos los embeddings de train por clase; predicción = mejor score sobre todos."""
    return {
        cls: np.array(embs, dtype=np.float32)
        for cls, embs in class_to_embs.items()
        if len(embs) > 0
    }


def save_all_strategies(exp_dir: str, class_to_embs: Dict[str, List[np.ndarray]]) -> List[str]:
    """Guarda reference_embeddings_<strategy>.npz para todas las estrategias (no incluye 'train')."""
    saved = []
    refs = build_references_centroid(class_to_embs, 5)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_5"))
    np.savez(path, **refs)
    saved.append("centroid_5")

    refs = build_references_centroid(class_to_embs, 10)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_10"))
    np.savez(path, **refs)
    saved.append("centroid_10")

    refs = build_references_centroid(class_to_embs, 20)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_20"))
    np.savez(path, **refs)
    saved.append("centroid_20")

    refs = build_references_centroid(class_to_embs, n_samples=None)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_all"))
    np.savez(path, **refs)
    saved.append("centroid_all")

    refs = build_references_centroid_l2norm(class_to_embs, 5)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_l2norm_5"))
    np.savez(path, **refs)
    saved.append("centroid_l2norm_5")

    refs = build_references_multiprototype(class_to_embs, 5)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="multiprototype_k5"))
    np.savez(path, **refs)
    saved.append("multiprototype_k5")

    refs = build_references_all_train(class_to_embs)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="all_train"))
    np.savez(path, **{k: v for k, v in refs.items()})
    saved.append("all_train")

    return saved


def embed_train_set(
    model,
    train_set: List[Tuple[str, str]],
    n_points: int,
    device,
    show_progress_every: int = 500,
) -> List[Tuple[str, str, np.ndarray]]:
    """train_set: list of (class, path). Returns list of (class, path, embedding)."""
    out = []
    for i, (cls, path) in enumerate(train_set):
        emb = embed_point_cloud_path(
            model=model, ply_path=path, n_points=n_points, device=device
        )
        out.append((cls, path, emb))
        if show_progress_every and (i + 1) % show_progress_every == 0:
            print(f"  ... embedidas {i + 1}/{len(train_set)} muestras", flush=True)
    return out


def ensure_all_strategies_saved(
    exp_dir: str,
    model,
    train_set: List[Tuple[str, str]],
    n_points: int,
    device,
) -> List[str]:
    """
    Si no existen las estrategias adicionales, embedea train, construye class_to_embs
    y guarda reference_embeddings_*.npz para centroid_5, centroid_10, etc.
    Devuelve la lista de estrategias guardadas (puede estar vacía si ya existían).
    """
    # Comprobar si ya hay algo más que "train"
    existing = set()
    if os.path.isdir(exp_dir):
        for fname in os.listdir(exp_dir):
            if fname.startswith("reference_embeddings_") and fname.endswith(".npz"):
                name = fname.replace("reference_embeddings_", "").replace(".npz", "")
                if name != "train":
                    existing.add(name)
    want = {"centroid_5", "centroid_10", "centroid_20", "centroid_all", "centroid_l2norm_5", "multiprototype_k5", "all_train"}
    if existing >= want:
        return []

    print("Calculando estrategias de referencia (embedding train + guardando)...", flush=True)
    train_embeddings = embed_train_set(model, train_set, n_points, device, show_progress_every=500)
    print(f"  Listo: {len(train_embeddings)} embeddings", flush=True)
    class_to_embs = defaultdict(list)
    for cls, _path, emb in train_embeddings:
        class_to_embs[cls].append(emb)
    saved = save_all_strategies(exp_dir, dict(class_to_embs))
    for s in saved:
        print(f"  [OK] {s}", flush=True)
    return saved
