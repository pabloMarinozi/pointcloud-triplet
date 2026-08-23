"""Post-procesamientos reproducibles para retrieval de embeddings."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

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
            order = np.argsort(-scores, kind="stable")
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


def _candidate(
    metrics: Dict[str, float],
    *,
    strategy: str,
    method: str,
    parameters: Dict[str, object] | None = None,
) -> Dict[str, object]:
    return {
        "strategy": strategy,
        "method": method,
        "parameters": parameters or {},
        "metrics": {name: float(value) for name, value in metrics.items()},
    }


def _best_candidate(candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not candidates:
        raise ValueError("No hay candidatos para seleccionar")

    def selection_key(candidate: Dict[str, object]) -> Tuple[float, ...]:
        metrics = candidate["metrics"]
        assert isinstance(metrics, dict)
        return (
            float(metrics["accuracy"]),
            float(metrics["mrr"]),
            float(metrics["top5_accuracy"]),
            float(metrics["top10_accuracy"]),
            -float(metrics["mean_rank"]),
        )

    return dict(max(candidates, key=selection_key))


def _fit_whitening(
    train_samples: Sequence[EmbeddingSample], config: WhiteningConfig
) -> PCAWhitening:
    train_matrix = np.stack([embedding for _, _, embedding in train_samples])
    return PCAWhitening(config).fit(train_matrix)


def _selected_parameters(
    candidate: Dict[str, object], group: str
) -> Dict[str, object]:
    parameters = candidate["parameters"]
    assert isinstance(parameters, dict)
    selected = parameters[group]
    assert isinstance(selected, dict)
    return selected


def _whitening_from_parameters(parameters: Mapping[str, object]) -> WhiteningConfig:
    n_components = parameters["n_components"]
    return WhiteningConfig(
        n_components=None if n_components is None else int(n_components),
        shrinkage=float(parameters["shrinkage"]),
    )


def _reranking_from_parameters(parameters: Mapping[str, object]) -> KReciprocalConfig:
    return KReciprocalConfig(
        k1=int(parameters["k1"]),
        k2=int(parameters["k2"]),
        lambda_value=float(parameters["lambda"]),
    )


def evaluate_postprocessing(
    *,
    train_samples: Sequence[EmbeddingSample],
    val_samples: Sequence[EmbeddingSample],
    test_samples: Sequence[EmbeddingSample],
    references_by_strategy: Mapping[str, References],
    methods: Mapping[str, Method],
    whitening_configs: Sequence[WhiteningConfig],
    reranking_configs: Sequence[KReciprocalConfig],
    rrf_constants: Sequence[int],
    seed: int = 42,
    fusion_strategies: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    """Selecciona configuraciones en val y las aplica sin cambios sobre test."""
    if not train_samples:
        raise ValueError("El post-procesamiento requiere embeddings de train")
    if not val_samples:
        raise ValueError("El post-procesamiento requiere embeddings de val")
    if not references_by_strategy:
        raise ValueError("No hay estrategias de referencia")
    if not methods:
        raise ValueError("No hay métodos para post-procesar")
    if not whitening_configs or not reranking_configs or not rrf_constants:
        raise ValueError("Las grillas de post-procesamiento no pueden estar vacías")

    strategy_names = list(fusion_strategies or references_by_strategy)
    missing = set(strategy_names).difference(references_by_strategy)
    if missing:
        raise ValueError(
            f"Estrategias pedidas para RRF no disponibles: {sorted(missing)}"
        )

    emit = progress or (lambda _message: None)
    candidates: Dict[str, List[Dict[str, object]]] = {
        "baseline": [],
        "whitening": [],
        "k_reciprocal": [],
        "fusion": [],
        "all": [],
    }
    base_val_rankings: Dict[Tuple[str, str], List[Ranking]] = {}

    emit("Post-procesamiento: baseline")
    for strategy, references in references_by_strategy.items():
        for method_name, method in methods.items():
            rankings = rank_samples(val_samples, references, method)
            base_val_rankings[(strategy, method_name)] = rankings
            candidates["baseline"].append(
                _candidate(
                    metrics_from_rankings(val_samples, rankings),
                    strategy=strategy,
                    method=method_name,
                )
            )

    emit("Post-procesamiento: grilla de PCA whitening")
    whitening_val_cache: Dict[
        WhiteningConfig, Tuple[List[EmbeddingSample], Dict[str, References]]
    ] = {}
    for whitening_config in whitening_configs:
        transform = _fit_whitening(train_samples, whitening_config)
        whitened_train = transform_samples(train_samples, transform)
        whitened_val = transform_samples(val_samples, transform)
        whitened_references = references_from_samples(
            whitened_train, list(references_by_strategy), seed
        )
        whitening_val_cache[whitening_config] = (
            whitened_val,
            whitened_references,
        )
        for strategy, references in whitened_references.items():
            for method_name, method in methods.items():
                rankings = rank_samples(whitened_val, references, method)
                candidates["whitening"].append(
                    _candidate(
                        metrics_from_rankings(whitened_val, rankings),
                        strategy=strategy,
                        method=method_name,
                        parameters={"whitening": whitening_config.as_dict()},
                    )
                )

    emit("Post-procesamiento: grilla k-recíproca")
    for reranking_config in reranking_configs:
        for strategy, references in references_by_strategy.items():
            for method_name, method in methods.items():
                rankings = rank_samples_k_reciprocal(
                    val_samples, references, method, reranking_config
                )
                candidates["k_reciprocal"].append(
                    _candidate(
                        metrics_from_rankings(val_samples, rankings),
                        strategy=strategy,
                        method=method_name,
                        parameters={
                            "k_reciprocal": reranking_config.as_dict()
                        },
                    )
                )

    emit("Post-procesamiento: grilla de Reciprocal Rank Fusion")
    for constant in rrf_constants:
        for method_name in methods:
            source_rankings = {
                strategy: base_val_rankings[(strategy, method_name)]
                for strategy in strategy_names
            }
            rankings = reciprocal_rank_fusion(source_rankings, constant)
            candidates["fusion"].append(
                _candidate(
                    metrics_from_rankings(val_samples, rankings),
                    strategy="rank_fusion",
                    method=method_name,
                    parameters={
                        "rrf": {
                            "constant": constant,
                            "strategies": strategy_names,
                        }
                    },
                )
            )

    selected = {
        variant: _best_candidate(variant_candidates)
        for variant, variant_candidates in candidates.items()
        if variant != "all"
    }
    selected_whitening = _whitening_from_parameters(
        _selected_parameters(selected["whitening"], "whitening")
    )
    selected_reranking = _reranking_from_parameters(
        _selected_parameters(selected["k_reciprocal"], "k_reciprocal")
    )
    selected_rrf = _selected_parameters(selected["fusion"], "rrf")
    whitened_val, whitened_references = whitening_val_cache[selected_whitening]

    emit("Post-procesamiento: whitening + k-recíproco + fusión")
    for method_name, method in methods.items():
        source_rankings = {
            strategy: rank_samples_k_reciprocal(
                whitened_val,
                whitened_references[strategy],
                method,
                selected_reranking,
            )
            for strategy in strategy_names
        }
        rankings = reciprocal_rank_fusion(
            source_rankings, int(selected_rrf["constant"])
        )
        candidates["all"].append(
            _candidate(
                metrics_from_rankings(whitened_val, rankings),
                strategy="whitened_k_reciprocal_rank_fusion",
                method=method_name,
                parameters={
                    "whitening": selected_whitening.as_dict(),
                    "k_reciprocal": selected_reranking.as_dict(),
                    "rrf": {
                        "constant": int(selected_rrf["constant"]),
                        "strategies": strategy_names,
                    },
                },
            )
        )
    selected["all"] = _best_candidate(candidates["all"])

    test_results: Dict[str, Dict[str, object]] = {}
    if test_samples:
        emit("Post-procesamiento: aplicación final sobre test")
        baseline_choice = selected["baseline"]
        baseline_strategy = str(baseline_choice["strategy"])
        baseline_method_name = str(baseline_choice["method"])
        baseline_ranking = rank_samples(
            test_samples,
            references_by_strategy[baseline_strategy],
            methods[baseline_method_name],
        )
        test_results["baseline"] = _candidate(
            metrics_from_rankings(test_samples, baseline_ranking),
            strategy=baseline_strategy,
            method=baseline_method_name,
        )

        whitening_transform = _fit_whitening(train_samples, selected_whitening)
        whitened_train = transform_samples(train_samples, whitening_transform)
        whitened_test = transform_samples(test_samples, whitening_transform)
        final_whitened_references = references_from_samples(
            whitened_train, list(references_by_strategy), seed
        )

        whitening_choice = selected["whitening"]
        whitening_strategy = str(whitening_choice["strategy"])
        whitening_method_name = str(whitening_choice["method"])
        whitening_ranking = rank_samples(
            whitened_test,
            final_whitened_references[whitening_strategy],
            methods[whitening_method_name],
        )
        test_results["whitening"] = _candidate(
            metrics_from_rankings(whitened_test, whitening_ranking),
            strategy=whitening_strategy,
            method=whitening_method_name,
            parameters={"whitening": selected_whitening.as_dict()},
        )

        reranking_choice = selected["k_reciprocal"]
        reranking_strategy = str(reranking_choice["strategy"])
        reranking_method_name = str(reranking_choice["method"])
        reranking_ranks = rank_samples_k_reciprocal(
            test_samples,
            references_by_strategy[reranking_strategy],
            methods[reranking_method_name],
            selected_reranking,
        )
        test_results["k_reciprocal"] = _candidate(
            metrics_from_rankings(test_samples, reranking_ranks),
            strategy=reranking_strategy,
            method=reranking_method_name,
            parameters={"k_reciprocal": selected_reranking.as_dict()},
        )

        fusion_choice = selected["fusion"]
        fusion_method_name = str(fusion_choice["method"])
        fusion_sources = {
            strategy: rank_samples(
                test_samples,
                references_by_strategy[strategy],
                methods[fusion_method_name],
            )
            for strategy in strategy_names
        }
        fusion_ranks = reciprocal_rank_fusion(
            fusion_sources, int(selected_rrf["constant"])
        )
        test_results["fusion"] = _candidate(
            metrics_from_rankings(test_samples, fusion_ranks),
            strategy="rank_fusion",
            method=fusion_method_name,
            parameters={
                "rrf": {
                    "constant": int(selected_rrf["constant"]),
                    "strategies": strategy_names,
                }
            },
        )

        all_choice = selected["all"]
        all_method_name = str(all_choice["method"])
        all_sources = {
            strategy: rank_samples_k_reciprocal(
                whitened_test,
                final_whitened_references[strategy],
                methods[all_method_name],
                selected_reranking,
            )
            for strategy in strategy_names
        }
        all_ranks = reciprocal_rank_fusion(
            all_sources, int(selected_rrf["constant"])
        )
        test_results["all"] = _candidate(
            metrics_from_rankings(whitened_test, all_ranks),
            strategy="whitened_k_reciprocal_rank_fusion",
            method=all_method_name,
            parameters=dict(all_choice["parameters"]),
        )

    return {
        "format_version": 1,
        "protocol": "select_hyperparameters_on_val_then_apply_unchanged_to_test",
        "grids": {
            "whitening": [config.as_dict() for config in whitening_configs],
            "k_reciprocal": [config.as_dict() for config in reranking_configs],
            "rrf_constants": [int(value) for value in rrf_constants],
            "methods": list(methods),
            "fusion_strategies": strategy_names,
        },
        "validation_candidates": candidates,
        "selected_on_val": selected,
        "val": selected,
        "test": test_results,
    }
