"""Post-procesamientos reproducibles para retrieval de embeddings."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.evaluation.metrics import Method, _score_class
from src.evaluation.ref_strategies import build_strategy_references


EmbeddingSample = Tuple[str, str, np.ndarray]
References = Dict[str, np.ndarray]
Ranking = List[str]


@dataclass(frozen=True)
class WhiteningConfig:
    n_components: int | None
    shrinkage: float

    def as_dict(self) -> Dict[str, int | float | None]:
        return {
            "n_components": self.n_components,
            "shrinkage": self.shrinkage,
        }


@dataclass(frozen=True)
class KReciprocalConfig:
    k1: int
    k2: int
    lambda_value: float

    def as_dict(self) -> Dict[str, int | float]:
        return {
            "k1": self.k1,
            "k2": self.k2,
            "lambda": self.lambda_value,
        }


class PCAWhitening:
    """PCA con whitening regularizado, ajustado exclusivamente sobre train."""

    def __init__(self, config: WhiteningConfig):
        if config.n_components is not None and config.n_components < 1:
            raise ValueError("n_components debe ser positivo o None")
        if config.shrinkage < 0.0:
            raise ValueError("shrinkage no puede ser negativo")
        self.config = config
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "PCAWhitening":
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("PCA whitening requiere una matriz con al menos 2 filas")
        if not np.all(np.isfinite(values)):
            raise ValueError("Los embeddings de train contienen valores no finitos")

        self.mean_ = values.mean(axis=0)
        centered = values - self.mean_
        _, singular_values, right_vectors = np.linalg.svd(
            centered, full_matrices=False
        )
        max_components = min(values.shape)
        requested = self.config.n_components or max_components
        n_components = min(requested, max_components)
        eigenvalues = singular_values[:n_components] ** 2 / (len(values) - 1)
        positive = eigenvalues[eigenvalues > np.finfo(np.float64).eps]
        variance_scale = float(positive.mean()) if len(positive) else 1.0
        regularizer = self.config.shrinkage * variance_scale

        self.components_ = right_vectors[:n_components]
        self.scales_ = np.sqrt(
            np.maximum(eigenvalues + regularizer, np.finfo(np.float64).eps)
        )
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None or self.scales_ is None:
            raise RuntimeError("PCAWhitening debe ajustarse antes de transformar")
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        transformed = ((values - self.mean_) @ self.components_.T) / self.scales_
        return transformed.astype(np.float32)


def transform_samples(
    samples: Sequence[EmbeddingSample], transform: PCAWhitening
) -> List[EmbeddingSample]:
    if not samples:
        return []
    matrix = np.stack([embedding for _, _, embedding in samples])
    transformed = transform.transform(matrix)
    return [
        (label, path, embedding)
        for (label, path, _), embedding in zip(samples, transformed)
    ]


def references_from_samples(
    samples: Sequence[EmbeddingSample], strategy_names: Sequence[str], seed: int
) -> Dict[str, References]:
    class_to_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    for label, _path, embedding in samples:
        class_to_embeddings[label].append(embedding)
    return {
        strategy: build_strategy_references(
            strategy, dict(class_to_embeddings), seed=seed
        )
        for strategy in strategy_names
    }


def rank_samples(
    samples: Sequence[EmbeddingSample], references: References, method: Method
) -> List[Ranking]:
    labels = list(references)
    rankings = []
    for _true_label, _path, embedding in samples:
        scores = np.asarray(
            [_score_class(embedding, references[label], method) for label in labels]
        )
        order = np.argsort(scores, kind="stable")
        if method.maximize:
            order = order[::-1]
        rankings.append([labels[index] for index in order])
    return rankings


def _score_references(a: np.ndarray, b: np.ndarray, method: Method) -> float:
    left = np.atleast_2d(a)
    right = np.atleast_2d(b)
    scores = np.asarray(
        [[method.func(x, y) for y in right] for x in left], dtype=np.float64
    )
    return float(scores.max() if method.maximize else scores.min())


def _scores_to_distances(scores: np.ndarray, maximize: bool) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    distances = values.max() - values if maximize else values - values.min()
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, 0.0)
    scale = distances.max(axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (distances / scale).T.astype(np.float32)


def k_reciprocal_rerank(
    distances: np.ndarray,
    query_count: int,
    config: KReciprocalConfig,
) -> np.ndarray:
    """Implementa k-reciprocal encoding y distancia Jaccard de Zhong et al."""
    original = np.asarray(distances, dtype=np.float32)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise ValueError("distances debe ser una matriz cuadrada")
    if not 0 < query_count < len(original):
        raise ValueError("query_count debe dejar al menos un elemento de galería")
    if config.k1 < 1 or config.k2 < 1:
        raise ValueError("k1 y k2 deben ser positivos")
    if not 0.0 <= config.lambda_value <= 1.0:
        raise ValueError("lambda debe estar entre 0 y 1")

    item_count = len(original)
    k1 = min(config.k1, item_count - 1)
    k2 = min(config.k2, item_count)
    initial_rank = np.argsort(original, axis=1, kind="stable")
    weights = np.zeros_like(original, dtype=np.float32)

    for index in range(item_count):
        forward = initial_rank[index, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.any(backward == index, axis=1)]
        expanded = reciprocal.copy()
        half_k = max(1, int(round(k1 / 2)))
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, : half_k + 1]
            candidate_backward = initial_rank[candidate_forward, : half_k + 1]
            candidate_reciprocal = candidate_forward[
                np.any(candidate_backward == candidate, axis=1)
            ]
            overlap = np.intersect1d(
                candidate_reciprocal, reciprocal, assume_unique=False
            )
            if len(overlap) > (2.0 / 3.0) * len(candidate_reciprocal):
                expanded = np.append(expanded, candidate_reciprocal)
        expanded = np.unique(expanded)
        affinity = np.exp(-original[index, expanded])
        total = affinity.sum()
        if total > 0.0:
            weights[index, expanded] = affinity / total

    if k2 > 1:
        averaged = np.zeros_like(weights)
        for index in range(item_count):
            averaged[index] = weights[initial_rank[index, :k2]].mean(axis=0)
        weights = averaged

    inverse_index = [np.flatnonzero(weights[:, column]) for column in range(item_count)]
    jaccard = np.ones((query_count, item_count), dtype=np.float32)
    for query_index in range(query_count):
        nonzero = np.flatnonzero(weights[query_index])
        minima = np.zeros(item_count, dtype=np.float32)
        for column in nonzero:
            related = inverse_index[column]
            minima[related] += np.minimum(
                weights[query_index, column], weights[related, column]
            )
        jaccard[query_index] -= minima / np.maximum(2.0 - minima, 1e-12)

    combined = (
        (1.0 - config.lambda_value) * jaccard
        + config.lambda_value * original[:query_count]
    )
    return combined[:, query_count:]


def rank_samples_k_reciprocal(
    samples: Sequence[EmbeddingSample],
    references: References,
    method: Method,
    config: KReciprocalConfig,
) -> List[Ranking]:
    """Reordena clases por query usando la vecindad recíproca entre referencias."""
    labels = list(references)
    reference_scores = np.empty((len(labels), len(labels)), dtype=np.float64)
    for left_index, left_label in enumerate(labels):
        for right_index in range(left_index, len(labels)):
            score = _score_references(
                references[left_label], references[labels[right_index]], method
            )
            reference_scores[left_index, right_index] = score
            reference_scores[right_index, left_index] = score

    rankings = []
    for _true_label, _path, embedding in samples:
        query_scores = np.asarray(
            [_score_class(embedding, references[label], method) for label in labels]
        )
        self_score = method.func(embedding, embedding)
        scores = np.empty((len(labels) + 1, len(labels) + 1), dtype=np.float64)
        scores[0, 0] = self_score
        scores[0, 1:] = query_scores
        scores[1:, 0] = query_scores
        scores[1:, 1:] = reference_scores
        distances = _scores_to_distances(scores, method.maximize)
        reranked = k_reciprocal_rerank(distances, 1, config)[0]
        order = np.argsort(reranked, kind="stable")
        rankings.append([labels[index] for index in order])
    return rankings


def reciprocal_rank_fusion(
    rankings_by_source: Mapping[str, Sequence[Ranking]], constant: int = 60
) -> List[Ranking]:
    """Fusiona rankings con sum(1 / (constant + rank)), sin usar etiquetas reales."""
    if constant < 1:
        raise ValueError("La constante de RRF debe ser positiva")
    sources = list(rankings_by_source.values())
    if not sources:
        raise ValueError("Se necesita al menos un ranking para fusionar")
    query_count = len(sources[0])
    if any(len(source) != query_count for source in sources):
        raise ValueError("Todos los rankings deben contener las mismas queries")

    fused = []
    for query_index in range(query_count):
        scores: Dict[str, float] = defaultdict(float)
        for source in sources:
            for rank, label in enumerate(source[query_index], start=1):
                scores[label] += 1.0 / (constant + rank)
        fused.append(
            sorted(scores, key=lambda label: (-scores[label], label))
        )
    return fused


def metrics_from_rankings(
    samples: Sequence[EmbeddingSample], rankings: Sequence[Ranking]
) -> Dict[str, float]:
    if len(samples) != len(rankings):
        raise ValueError("Debe haber un ranking por muestra")
    if not samples:
        return {
            "accuracy": 0.0,
            "top5_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
        }
    ranks = []
    for (true_label, _path, _embedding), ranking in zip(samples, rankings):
        ranks.append(ranking.index(true_label) + 1 if true_label in ranking else 0)
    valid = [rank for rank in ranks if rank > 0]
    count = len(samples)
    return {
        "accuracy": sum(rank == 1 for rank in ranks) / count,
        "top5_accuracy": sum(0 < rank <= 5 for rank in ranks) / count,
        "top10_accuracy": sum(0 < rank <= 10 for rank in ranks) / count,
        "mrr": sum(1.0 / rank for rank in valid) / count,
        "mean_rank": float(np.mean(valid)) if valid else 0.0,
        "median_rank": float(np.median(valid)) if valid else 0.0,
    }
