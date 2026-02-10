from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


# ----------------------------
# Métricas (distancia pura)
# ----------------------------

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def linf_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


# ----------------------------
# Métricas (similitud)
# ----------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def inv1p_l2_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # similitud: más grande = más cerca
    return 1.0 / (1.0 + l2_distance(a, b))


def inv1p_l1_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 / (1.0 + l1_distance(a, b))


@dataclass(frozen=True)
class Method:
    name: str
    func: Callable[[np.ndarray, np.ndarray], float]
    maximize: bool


def default_methods() -> Dict[str, Method]:
    """
    Métodos recomendados, consistentes con "maximize" correcto.
    """
    methods = [
        Method("Cosine Similarity", cosine_similarity, True),
        Method("Dot Product", dot_product, True),

        # Distancias reales (minimizar)
        Method("L2 Distance", l2_distance, False),
        Method("L1 Distance", l1_distance, False),
        Method("Linf Distance", linf_distance, False),

        # Opcionales: similitudes inversas (maximizar)
        Method("Inv(1+L2) Similarity", inv1p_l2_similarity, True),
        Method("Inv(1+L1) Similarity", inv1p_l1_similarity, True),
    ]
    return {m.name: m for m in methods}


def _score_class(emb: np.ndarray, ref_emb: np.ndarray, method: Method) -> float:
    """
    Mejor score de la query contra la referencia de una clase.
    ref_emb puede ser (dim,) o (k, dim); si es (k, dim) se toma el mejor sobre las filas.
    """
    if ref_emb.ndim == 1:
        return float(method.func(emb, ref_emb))
    # (k, dim): mejor = min para distancias, max para similitudes
    scores = [float(method.func(emb, ref_emb[i])) for i in range(ref_emb.shape[0])]
    if method.maximize:
        return max(scores)
    return min(scores)


def predict_class(
    emb: np.ndarray,
    reference_embeddings: Dict[str, np.ndarray],
    method: Method,
) -> Tuple[str, float]:
    """
    Devuelve (pred_label, score).
    reference_embeddings: clase -> vector (dim,) o matriz (k, dim) de prototipos.
    """
    best_label = None
    best_score = None

    for label, ref_emb in reference_embeddings.items():
        score = _score_class(emb, ref_emb, method)

        if best_label is None:
            best_label, best_score = label, score
            continue

        if method.maximize:
            if score > best_score:
                best_label, best_score = label, score
        else:
            if score < best_score:
                best_label, best_score = label, score

    assert best_label is not None and best_score is not None
    return best_label, float(best_score)


def rank_of_label(
    emb: np.ndarray,
    reference_embeddings: Dict[str, np.ndarray],
    method: Method,
    label: str,
) -> int:
    """
    Posición 1-based de la clase `label` en el ranking de scores (mejor score = 1).
    Si la clase no está en reference_embeddings, devuelve 0.
    """
    if label not in reference_embeddings:
        return 0
    items = [
        (cls, _score_class(emb, ref_emb, method))
        for cls, ref_emb in reference_embeddings.items()
    ]
    items.sort(key=lambda x: x[1], reverse=method.maximize)
    for rank, (cls, _) in enumerate(items, start=1):
        if cls == label:
            return rank
    return 0
