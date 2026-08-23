"""
Estrategias de reference embeddings (centroides, multiprototipo, etc.).
Usado por src.eval para generar estrategias de referencia si faltan (--ref_strategy all).
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans

from src.evaluation.embed import embed_point_cloud_paths
from src.evaluation.embedding_cache import (
    build_cache_manifest,
    cache_is_compatible,
    cache_paths,
    ensure_embedding_cache,
)

SEED = 42
STRATEGY_REF_BASENAME = "reference_embeddings_{strategy}.npz"
STRATEGY_MANIFEST_BASENAME = "reference_embeddings.manifest.json"
STRATEGY_NAMES = {
    "centroid_5",
    "centroid_10",
    "centroid_20",
    "centroid_all",
    "median_all",
    "trimmed_mean_05",
    "trimmed_mean_10",
    "centroid_l2norm_5",
    "multiprototype_k5",
}


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n > 1e-12, n, 1.0)
    return (x / n).astype(np.float32)


def coordinate_median(embeddings: np.ndarray) -> np.ndarray:
    """Agrega embeddings con la mediana de cada coordenada."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("embeddings debe ser una matriz 2D no vacía")
    return np.median(values, axis=0).astype(np.float32)


def coordinate_trimmed_mean(
    embeddings: np.ndarray, proportion_to_cut: float
) -> np.ndarray:
    """Calcula la media por coordenada tras recortar ambas colas."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("embeddings debe ser una matriz 2D no vacía")
    if not 0.0 <= proportion_to_cut < 0.5:
        raise ValueError("proportion_to_cut debe estar en el intervalo [0, 0.5)")

    trim_count = int(np.floor(len(values) * proportion_to_cut))
    if trim_count == 0:
        return values.mean(axis=0).astype(np.float32)

    sorted_values = np.sort(values, axis=0)
    return sorted_values[trim_count:-trim_count].mean(axis=0).astype(np.float32)


def build_references_median(
    class_to_embs: Dict[str, List[np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Un prototipo por clase mediante mediana de cada coordenada."""
    refs = {}
    for cls, embs in class_to_embs.items():
        if len(embs) == 0:
            continue
        refs[cls] = coordinate_median(np.asarray(embs, dtype=np.float32))
    return refs


def build_references_trimmed_mean(
    class_to_embs: Dict[str, List[np.ndarray]], proportion_to_cut: float
) -> Dict[str, np.ndarray]:
    """Un prototipo por clase mediante media recortada por coordenada."""
    refs = {}
    for cls, embs in class_to_embs.items():
        if len(embs) == 0:
            continue
        refs[cls] = coordinate_trimmed_mean(
            np.asarray(embs, dtype=np.float32), proportion_to_cut
        )
    return refs


def build_references_centroid(
    class_to_embs: Dict[str, List[np.ndarray]],
    n_samples: int | None,
    shuffle: bool = True,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """Un vector por clase = media de hasta n_samples embeddings (o todos si n_samples is None)."""
    refs = {}
    rng = random.Random(seed)
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
    class_to_embs: Dict[str, List[np.ndarray]], n_samples: int, seed: int = SEED
) -> Dict[str, np.ndarray]:
    """Media de embeddings L2-normalizados, luego L2-normalizar el resultado."""
    refs = {}
    rng = random.Random(seed)
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
    class_to_embs: Dict[str, List[np.ndarray]], k: int, seed: int = SEED
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
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        kmeans.fit(arr)
        refs[cls] = kmeans.cluster_centers_.astype(np.float32)
    return refs


def save_all_strategies(
    exp_dir: str,
    class_to_embs: Dict[str, List[np.ndarray]],
    seed: int = SEED,
) -> List[str]:
    """Construye y guarda todas las estrategias desde los mismos embeddings."""
    saved = []
    refs = build_references_centroid(class_to_embs, 5, seed=seed)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_5"))
    np.savez(path, **refs)
    saved.append("centroid_5")

    refs = build_references_centroid(class_to_embs, 10, seed=seed)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_10"))
    np.savez(path, **refs)
    saved.append("centroid_10")

    refs = build_references_centroid(class_to_embs, 20, seed=seed)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_20"))
    np.savez(path, **refs)
    saved.append("centroid_20")

    # centroid_all = centroide usando TODOS los embeddings de train (no solo 5/10/20)
    refs = build_references_centroid(class_to_embs, n_samples=None, seed=seed)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_all"))
    np.savez(path, **refs)
    saved.append("centroid_all")

    refs = build_references_median(class_to_embs)
    path = os.path.join(
        exp_dir, STRATEGY_REF_BASENAME.format(strategy="median_all")
    )
    np.savez(path, **refs)
    saved.append("median_all")

    refs = build_references_trimmed_mean(class_to_embs, proportion_to_cut=0.05)
    path = os.path.join(
        exp_dir, STRATEGY_REF_BASENAME.format(strategy="trimmed_mean_05")
    )
    np.savez(path, **refs)
    saved.append("trimmed_mean_05")

    refs = build_references_trimmed_mean(class_to_embs, proportion_to_cut=0.10)
    path = os.path.join(
        exp_dir, STRATEGY_REF_BASENAME.format(strategy="trimmed_mean_10")
    )
    np.savez(path, **refs)
    saved.append("trimmed_mean_10")

    refs = build_references_centroid_l2norm(class_to_embs, 5, seed=seed)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="centroid_l2norm_5"))
    np.savez(path, **refs)
    saved.append("centroid_l2norm_5")

    refs = build_references_multiprototype(class_to_embs, 5, seed=seed)
    path = os.path.join(exp_dir, STRATEGY_REF_BASENAME.format(strategy="multiprototype_k5"))
    np.savez(path, **refs)
    saved.append("multiprototype_k5")

    return saved


def embed_train_set(
    model,
    train_set: List[Tuple[str, str]],
    n_points: int,
    device,
    show_progress_every: int = 500,
    sampling: str = "random",
    seed: int = SEED,
    batch_size: int = 64,
) -> List[Tuple[str, str, np.ndarray]]:
    """train_set: list of (class, path). Returns list of (class, path, embedding)."""
    return embed_point_cloud_paths(
        model=model,
        samples=train_set,
        n_points=n_points,
        device=device,
        sampling=sampling,
        seed=seed,
        batch_size=batch_size,
        show_progress_every=show_progress_every,
    )


def ensure_all_strategies_saved(
    exp_dir: str,
    model,
    train_set: List[Tuple[str, str]],
    n_points: int,
    device,
    *,
    checkpoint_path: str,
    checkpoint_sha256: str,
    sampling: str = "random",
    seed: int = SEED,
    use_augmentation: bool = False,
    batch_size: int = 64,
    views: int = 1,
    view_aggregation: str = "none",
    video_index: Dict[str, str] | None = None,
    runtime_stats: Dict[str, object] | None = None,
) -> List[str]:
    """
    Si no existen las estrategias generables, embedea el train set, construye class_to_embs
    y guarda reference_embeddings_*.npz para centroid_5, centroid_10, etc.
    Devuelve la lista de estrategias guardadas (puede estar vacía si ya existían).
    """
    started = time.perf_counter()
    want = STRATEGY_NAMES
    train_cache_manifest = build_cache_manifest(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        samples=train_set,
        split="train",
        n_points=n_points,
        sampling=sampling,
        seed=seed,
        use_augmentation=use_augmentation,
        batch_size=batch_size,
        views=views,
        view_aggregation=view_aggregation,
    )
    expected_strategy_manifest = {
        "format_version": 3,
        "source_cache": train_cache_manifest,
        "strategies": sorted(want),
    }
    strategy_manifest_path = os.path.join(exp_dir, STRATEGY_MANIFEST_BASENAME)
    existing = set()
    if os.path.isdir(exp_dir):
        for fname in os.listdir(exp_dir):
            if fname.startswith("reference_embeddings_") and fname.endswith(".npz"):
                name = fname.replace("reference_embeddings_", "").replace(".npz", "")
                existing.add(name)
    manifest_matches = False
    try:
        with open(strategy_manifest_path, "r", encoding="utf-8") as file:
            manifest_matches = json.load(file) == expected_strategy_manifest
    except (OSError, json.JSONDecodeError):
        pass
    if existing >= want and manifest_matches:
        cache_path, cache_manifest_path = cache_paths(exp_dir, "train")
        if cache_is_compatible(
            cache_path, cache_manifest_path, train_cache_manifest
        ):
            if runtime_stats is not None:
                runtime_stats.update(
                    {
                        "cache_hit": True,
                        "elapsed_seconds": time.perf_counter() - started,
                        "strategy_count": len(want),
                    }
                )
            return []

    print("Preparando estrategias de referencia desde el caché de train...", flush=True)
    train_cache_stats: Dict[str, object] = {}
    train_embeddings = ensure_embedding_cache(
        cache_dir=exp_dir,
        split="train",
        model=model,
        samples=train_set,
        n_points=n_points,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        sampling=sampling,
        seed=seed,
        use_augmentation=use_augmentation,
        batch_size=batch_size,
        views=views,
        view_aggregation=view_aggregation,
        video_index=video_index,
        runtime_stats=train_cache_stats,
    )
    print(f"  Listo: {len(train_embeddings)} embeddings", flush=True)
    class_to_embs = defaultdict(list)
    for cls, _path, emb in train_embeddings:
        class_to_embs[cls].append(emb)
    saved = save_all_strategies(exp_dir, dict(class_to_embs), seed=seed)
    tmp_manifest_path = f"{strategy_manifest_path}.tmp"
    with open(tmp_manifest_path, "w", encoding="utf-8") as file:
        json.dump(expected_strategy_manifest, file, indent=2, ensure_ascii=False)
    os.replace(tmp_manifest_path, strategy_manifest_path)
    if runtime_stats is not None:
        runtime_stats.update(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "strategy_count": len(saved),
                "train_cache": train_cache_stats,
            }
        )
    for s in saved:
        print(f"  [OK] {s}", flush=True)
    return saved
