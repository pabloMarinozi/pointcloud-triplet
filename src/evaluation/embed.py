from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open3d as o3d
import torch

from src.data.dataset import normalize_unit_sphere, sample_n, augment
from src.evaluation.runtime_stats import peak_process_memory_mb


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


def derive_sample_seed(seed: int, path: str, view_id: int = 0) -> int:
    """Deriva una semilla estable sin depender del orden de recorrido."""
    payload = f"{seed}\0{path}\0{view_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def preprocess_point_cloud_path(
    ply_path: str,
    n_points: int,
    use_augmentation: bool = False,
    sampling: str = "random",
    seed: int = 42,
    view_id: int = 0,
) -> np.ndarray:
    """Lee, normaliza y samplea una nube de forma reproducible por path/vista."""
    pts = read_points_from_ply(ply_path)
    pts = normalize_unit_sphere(pts).astype(np.float32)
    return preprocess_normalized_points(
        pts,
        ply_path=ply_path,
        n_points=n_points,
        use_augmentation=use_augmentation,
        sampling=sampling,
        seed=seed,
        view_id=view_id,
    )


def preprocess_normalized_points(
    points: np.ndarray,
    *,
    ply_path: str,
    n_points: int,
    use_augmentation: bool = False,
    sampling: str = "random",
    seed: int = 42,
    view_id: int = 0,
) -> np.ndarray:
    """Muestrea una nube ya normalizada para una vista determinista."""
    rng = np.random.default_rng(derive_sample_seed(seed, ply_path, view_id))
    if use_augmentation:
        return augment(points, n_points, sampling, rng=rng).astype(np.float32)
    return sample_n(points, n_points, sampling, rng=rng).astype(np.float32)


def embed_point_cloud_path(
    model,
    ply_path: str,
    n_points: int,
    device: torch.device,
    use_augmentation: bool = False,
    sampling: str = "random",
    seed: int = 42,
    view_id: int = 0,
) -> np.ndarray:
    """
    Retorna embedding 1D (emb_dim,).
    """
    pts_proc = preprocess_point_cloud_path(
        ply_path=ply_path,
        n_points=n_points,
        use_augmentation=use_augmentation,
        sampling=sampling,
        seed=seed,
        view_id=view_id,
    )
    x = torch.from_numpy(pts_proc.T).unsqueeze(0).float().to(device)

    with torch.no_grad():
        emb = get_embedding_from_model(model, x).squeeze(0).detach().cpu().numpy()

    return emb.astype(np.float32)


def embed_point_cloud_paths(
    model,
    samples: Sequence[Tuple[str, str]],
    n_points: int,
    device: torch.device,
    use_augmentation: bool = False,
    sampling: str = "random",
    seed: int = 42,
    view_id: int = 0,
    batch_size: int = 64,
    show_progress_every: int = 500,
) -> List[Tuple[str, str, np.ndarray]]:
    """Genera embeddings en batches manteniendo orden y semillas por path/vista."""
    view_embeddings = embed_point_cloud_views(
        model=model,
        samples=samples,
        n_points=n_points,
        device=device,
        use_augmentation=use_augmentation,
        sampling=sampling,
        seed=seed,
        view_ids=(view_id,),
        batch_size=batch_size,
        show_progress_every=show_progress_every,
    )
    return [
        (label, path, embedding)
        for label, path, _view_id, embedding in view_embeddings
    ]


def embed_point_cloud_views(
    model,
    samples: Sequence[Tuple[str, str]],
    n_points: int,
    device: torch.device,
    use_augmentation: bool = False,
    sampling: str = "random",
    seed: int = 42,
    view_ids: Sequence[int] = (0,),
    batch_size: int = 64,
    show_progress_every: int = 500,
    runtime_stats: Dict[str, float] | None = None,
) -> List[Tuple[str, str, int, np.ndarray]]:
    """Genera vistas en batches leyendo y normalizando cada PLY una sola vez."""
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor o igual que 1")
    normalized_view_ids = tuple(int(view_id) for view_id in view_ids)
    if not normalized_view_ids or min(normalized_view_ids) < 0:
        raise ValueError("view_ids debe contener enteros no negativos")
    if len(set(normalized_view_ids)) != len(normalized_view_ids):
        raise ValueError("view_ids no puede contener valores repetidos")

    output: List[Tuple[str, str, int, np.ndarray]] = []
    clouds_per_batch = max(1, batch_size // len(normalized_view_ids))
    peak_rss_mb = peak_process_memory_mb()
    for start in range(0, len(samples), clouds_per_batch):
        batch = samples[start : start + clouds_per_batch]
        points = []
        metadata = []
        for label, path in batch:
            normalized = normalize_unit_sphere(
                read_points_from_ply(path)
            ).astype(np.float32)
            for current_view_id in normalized_view_ids:
                points.append(
                    preprocess_normalized_points(
                        normalized,
                        ply_path=path,
                        n_points=n_points,
                        use_augmentation=use_augmentation,
                        sampling=sampling,
                        seed=seed,
                        view_id=current_view_id,
                    )
                )
                metadata.append((label, path, current_view_id))
        x = torch.from_numpy(np.stack(points).transpose(0, 2, 1)).float().to(device)
        with torch.no_grad():
            embeddings = get_embedding_from_model(model, x).detach().cpu().numpy()
        for (label, path, current_view_id), embedding in zip(metadata, embeddings):
            output.append(
                (
                    label,
                    path,
                    current_view_id,
                    np.asarray(embedding, dtype=np.float32),
                )
            )
        completed = start + len(batch)
        peak_rss_mb = max(
            peak_rss_mb, peak_process_memory_mb()
        )
        if show_progress_every and (
            completed == len(samples) or completed % show_progress_every < len(batch)
        ):
            print(
                f"  ... embedidas {completed}/{len(samples)} nubes "
                f"({len(output)} vistas)",
                flush=True,
            )
    if runtime_stats is not None:
        runtime_stats["peak_process_rss_mb"] = peak_rss_mb
        if torch.device(device).type == "cuda":
            runtime_stats["peak_cuda_memory_mb"] = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            )
    return output


def aggregate_view_embeddings(
    view_embeddings: Sequence[Tuple[str, str, int, np.ndarray]],
    aggregation: str,
) -> List[Tuple[str, str, np.ndarray]]:
    """Agrega las vistas de cada path conservando el orden de las nubes."""
    if aggregation not in {"none", "coordinate_mean", "coordinate_median"}:
        raise ValueError(f"Agregación multi-vista no soportada: {aggregation}")

    grouped = OrderedDict()
    for label, path, view_id, embedding in view_embeddings:
        key = (label, path)
        grouped.setdefault(key, []).append((view_id, embedding))

    output = []
    for (label, path), values in grouped.items():
        ordered = [
            embedding for _, embedding in sorted(values, key=lambda item: item[0])
        ]
        matrix = np.stack(ordered).astype(np.float32)
        if aggregation == "none":
            if len(matrix) != 1:
                raise ValueError("aggregation='none' requiere exactamente una vista")
            aggregated = matrix[0]
        elif aggregation == "coordinate_mean":
            aggregated = matrix.mean(axis=0)
        else:
            aggregated = np.median(matrix, axis=0)
        output.append((label, path, np.asarray(aggregated, dtype=np.float32)))
    return output


def embed_point_cloud_paths_multiview(
    model,
    samples: Sequence[Tuple[str, str]],
    n_points: int,
    device: torch.device,
    use_augmentation: bool = False,
    sampling: str = "random",
    seed: int = 42,
    views: int = 1,
    view_aggregation: str = "none",
    batch_size: int = 64,
    show_progress_every: int = 500,
) -> List[Tuple[str, str, np.ndarray]]:
    """Genera y agrega varias vistas deterministas por nube."""
    if views < 1:
        raise ValueError("views debe ser mayor o igual que 1")
    raw = embed_point_cloud_views(
        model=model,
        samples=samples,
        n_points=n_points,
        device=device,
        use_augmentation=use_augmentation,
        sampling=sampling,
        seed=seed,
        view_ids=range(views),
        batch_size=batch_size,
        show_progress_every=show_progress_every,
    )
    return aggregate_view_embeddings(raw, view_aggregation)
