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


def predict_class(
    emb: np.ndarray,
    reference_embeddings: Dict[str, np.ndarray],
    method: Method,
) -> Tuple[str, float]:
    """
    Devuelve (pred_label, score).
    """
    best_label = None
    best_score = None

    for label, ref_emb in reference_embeddings.items():
        score = method.func(emb, ref_emb)

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
