"""Caché reproducible de embeddings individuales para evaluación."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.evaluation.embed import (
    aggregate_view_embeddings,
    derive_sample_seed,
    embed_point_cloud_views,
)
from src.evaluation.video_index import get_video_and_capture_form
from src.evaluation.runtime_stats import peak_process_memory_mb


CACHE_FORMAT_VERSION = 2
CACHE_BASENAME = "individual_embeddings_{split}.npz"
MANIFEST_BASENAME = "individual_embeddings_{split}.manifest.json"
REQUIRED_ARRAYS = {
    "path",
    "label",
    "video",
    "capture_form",
    "view_id",
    "seed",
    "embedding",
}


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_samples(samples: Sequence[Tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for label, path in samples:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(os.path.abspath(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_cache_manifest(
    *,
    checkpoint_path: str,
    checkpoint_sha256: str,
    samples: Sequence[Tuple[str, str]],
    split: str,
    n_points: int,
    sampling: str,
    seed: int,
    use_augmentation: bool,
    batch_size: int = 64,
    views: int = 1,
    view_aggregation: str = "none",
) -> Dict[str, object]:
    if views < 1:
        raise ValueError("views debe ser mayor o igual que 1")
    if view_aggregation not in {
        "none",
        "coordinate_mean",
        "coordinate_median",
    }:
        raise ValueError(
            f"Agregación multi-vista no soportada: {view_aggregation}"
        )
    if views > 1 and view_aggregation == "none":
        raise ValueError("Las configuraciones multi-vista requieren agregación")
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "split": split,
        "split_paths_sha256": hash_samples(samples),
        "sample_count": len(samples) * views,
        "n_points": n_points,
        "sampling": sampling,
        "views": views,
        "view_aggregation": view_aggregation,
        "normalization": "unit_sphere",
        "use_augmentation": use_augmentation,
        "batch_size": batch_size,
        "seed": seed,
    }


def cache_paths(cache_dir: str, split: str) -> Tuple[str, str]:
    return (
        os.path.join(cache_dir, CACHE_BASENAME.format(split=split)),
        os.path.join(cache_dir, MANIFEST_BASENAME.format(split=split)),
    )


def cache_is_compatible(
    cache_path: str, manifest_path: str, expected_manifest: Dict[str, object]
) -> bool:
    if not os.path.exists(cache_path) or not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            current_manifest = json.load(file)
        if current_manifest != expected_manifest:
            return False
        with np.load(cache_path, allow_pickle=False) as data:
            if not REQUIRED_ARRAYS.issubset(data.files):
                return False
            count = int(expected_manifest["sample_count"])
            return all(len(data[name]) == count for name in REQUIRED_ARRAYS)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return False


def load_embedding_cache(
    cache_path: str, view_aggregation: str = "none"
) -> List[Tuple[str, str, np.ndarray]]:
    with np.load(cache_path, allow_pickle=False) as data:
        missing = REQUIRED_ARRAYS.difference(data.files)
        if missing:
            raise ValueError(f"Caché incompleto; faltan arrays: {sorted(missing)}")
        raw_embeddings = [
            (
                str(label),
                str(path),
                int(view_id),
                np.asarray(embedding, dtype=np.float32),
            )
            for label, path, view_id, embedding in zip(
                data["label"],
                data["path"],
                data["view_id"],
                data["embedding"],
            )
        ]
    return aggregate_view_embeddings(raw_embeddings, view_aggregation)


def _save_embedding_cache(
    cache_path: str,
    manifest_path: str,
    embeddings: Sequence[Tuple[str, str, int, np.ndarray]],
    manifest: Dict[str, object],
    video_index: Dict[str, str] | None,
) -> None:
    paths = [path for _, path, _, _ in embeddings]
    labels = [label for label, _, _, _ in embeddings]
    view_ids = np.asarray(
        [view_id for _, _, view_id, _ in embeddings], dtype=np.int32
    )
    videos_and_forms = [
        get_video_and_capture_form(path, video_index) for path in paths
    ]
    seeds = np.asarray(
        [
            derive_sample_seed(int(manifest["seed"]), path, int(view_id))
            for path, view_id in zip(paths, view_ids)
        ],
        dtype=np.uint64,
    )
    matrix = np.stack(
        [embedding for _, _, _, embedding in embeddings]
    ).astype(np.float32)

    tmp_cache = f"{cache_path}.tmp.npz"
    tmp_manifest = f"{manifest_path}.tmp"
    np.savez_compressed(
        tmp_cache,
        path=np.asarray(paths),
        label=np.asarray(labels),
        video=np.asarray([item[0] for item in videos_and_forms]),
        capture_form=np.asarray([item[1] for item in videos_and_forms]),
        view_id=view_ids,
        seed=seeds,
        embedding=matrix,
    )
    with open(tmp_manifest, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    os.replace(tmp_cache, cache_path)
    os.replace(tmp_manifest, manifest_path)


def ensure_embedding_cache(
    *,
    cache_dir: str,
    split: str,
    model,
    samples: Sequence[Tuple[str, str]],
    n_points: int,
    device,
    checkpoint_path: str,
    checkpoint_sha256: str,
    sampling: str = "random",
    seed: int = 42,
    use_augmentation: bool = False,
    batch_size: int = 64,
    views: int = 1,
    view_aggregation: str = "none",
    video_index: Dict[str, str] | None = None,
    runtime_stats: Dict[str, object] | None = None,
) -> List[Tuple[str, str, np.ndarray]]:
    """Carga un caché compatible o lo regenera mediante inferencia batcheada."""
    if not samples:
        return []
    started = time.perf_counter()
    peak_rss_before_mb = peak_process_memory_mb()
    os.makedirs(cache_dir, exist_ok=True)
    cache_path, manifest_path = cache_paths(cache_dir, split)
    expected = build_cache_manifest(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        samples=samples,
        split=split,
        n_points=n_points,
        sampling=sampling,
        seed=seed,
        use_augmentation=use_augmentation,
        batch_size=batch_size,
        views=views,
        view_aggregation=view_aggregation,
    )
    if cache_is_compatible(cache_path, manifest_path, expected):
        print(f"  Caché {split}: compatible, reutilizando {cache_path}", flush=True)
        embeddings = load_embedding_cache(cache_path, view_aggregation)
        if runtime_stats is not None:
            runtime_stats.update(
                {
                    "cache_hit": True,
                    "cloud_count": len(embeddings),
                    "view_count": len(samples) * views,
                    "views_per_cloud": views,
                    "view_aggregation": view_aggregation,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_process_rss_before_mb": peak_rss_before_mb,
                    "peak_process_rss_mb": peak_process_memory_mb(),
                    "cache_size_mb": os.path.getsize(cache_path) / (1024 ** 2),
                }
            )
        return embeddings

    if os.path.exists(cache_path) or os.path.exists(manifest_path):
        print(f"  Caché {split}: incompatible o incompleto, regenerando", flush=True)
    else:
        print(f"  Caché {split}: generando", flush=True)
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    embedding_stats: Dict[str, float] = {}
    raw_embeddings = embed_point_cloud_views(
        model=model,
        samples=samples,
        n_points=n_points,
        device=device,
        use_augmentation=use_augmentation,
        sampling=sampling,
        seed=seed,
        view_ids=range(views),
        batch_size=batch_size,
        runtime_stats=embedding_stats,
    )
    _save_embedding_cache(
        cache_path, manifest_path, raw_embeddings, expected, video_index
    )
    embeddings = aggregate_view_embeddings(raw_embeddings, view_aggregation)
    if runtime_stats is not None:
        runtime_stats.update(
            {
                "cache_hit": False,
                "cloud_count": len(embeddings),
                "view_count": len(raw_embeddings),
                "views_per_cloud": views,
                "view_aggregation": view_aggregation,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_process_rss_before_mb": peak_rss_before_mb,
                "peak_process_rss_mb": embedding_stats.get(
                    "peak_process_rss_mb", peak_process_memory_mb()
                ),
                "peak_cuda_memory_mb": embedding_stats.get(
                    "peak_cuda_memory_mb", 0.0
                ),
                "cache_size_mb": os.path.getsize(cache_path) / (1024 ** 2),
            }
        )
    return embeddings
