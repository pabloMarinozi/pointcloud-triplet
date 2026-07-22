from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.embed import embed_point_cloud_path
from src.evaluation.metrics import Method, predict_class


UNKNOWN_LABEL = "__unknown__"


def _embed_set(model, samples, n_points: int, device, use_augmentation: bool):
    return [
        (
            label,
            path,
            embed_point_cloud_path(
                model=model,
                ply_path=path,
                n_points=n_points,
                device=device,
                use_augmentation=use_augmentation,
            ),
        )
        for label, path in samples
    ]


def _novelty_score(score: float, method: Method) -> float:
    """Higher always means more likely to be unknown."""
    return -score if method.maximize else score


def _best_threshold(scores: np.ndarray, is_unknown: np.ndarray) -> float:
    unique = np.unique(scores)
    if len(unique) == 1:
        candidates = np.array([np.nextafter(unique[0], -np.inf), unique[0]])
    else:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
        candidates = np.concatenate(
            ([np.nextafter(unique[0], -np.inf)], midpoints, [unique[-1]])
        )
    best = max(
        candidates,
        key=lambda threshold: balanced_accuracy_score(
            is_unknown, scores > threshold
        ),
    )
    return float(best)


def evaluate_open_set(
    model,
    reference_embeddings: Dict[str, np.ndarray],
    known_val_set: List[Tuple[str, str]],
    unknown_val_set: List[Tuple[str, str]],
    known_test_set: List[Tuple[str, str]],
    unknown_test_set: List[Tuple[str, str]],
    methods: Dict[str, Method],
    n_points: int,
    device,
    use_augmentation: bool = False,
    export_csv: bool = False,
    out_dir: str | None = None,
) -> Dict[str, Dict[str, float]]:
    """Calibrate rejection thresholds on val and evaluate once on held-out test."""
    if not known_val_set or not unknown_val_set:
        raise ValueError("Open-set calibration necesita muestras val conocidas y desconocidas.")
    if not known_test_set or not unknown_test_set:
        raise ValueError("Open-set evaluation necesita muestras test conocidas y desconocidas.")
    if export_csv and out_dir is None:
        raise ValueError("Si export_csv=True, out_dir no puede ser None.")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    calibration = _embed_set(
        model, known_val_set + unknown_val_set, n_points, device, use_augmentation
    )
    test = _embed_set(
        model, known_test_set + unknown_test_set, n_points, device, use_augmentation
    )
    known_labels = set(reference_embeddings)
    results: Dict[str, Dict[str, float]] = {}

    for method_name, method in methods.items():
        calibration_scores = []
        calibration_targets = []
        for true_label, _, emb in calibration:
            _, score = predict_class(emb, reference_embeddings, method)
            calibration_scores.append(_novelty_score(score, method))
            calibration_targets.append(int(true_label not in known_labels))
        threshold = _best_threshold(
            np.asarray(calibration_scores), np.asarray(calibration_targets)
        )

        targets = []
        predictions = []
        overall_correct = []
        rows = []
        novelty_scores = []
        for true_label, path, emb in test:
            nearest_label, score = predict_class(emb, reference_embeddings, method)
            novelty = _novelty_score(score, method)
            true_unknown = int(true_label not in known_labels)
            predicted_unknown = int(novelty > threshold)
            predicted_label = UNKNOWN_LABEL if predicted_unknown else nearest_label
            targets.append(true_unknown)
            predictions.append(predicted_unknown)
            novelty_scores.append(novelty)
            correct = predicted_unknown == 1 if true_unknown else predicted_label == true_label
            overall_correct.append(int(correct))
            rows.append(
                {
                    "path": path,
                    "true_label": true_label,
                    "pred_label": predicted_label,
                    "is_unknown": true_unknown,
                    "predicted_unknown": predicted_unknown,
                    "correct": int(correct),
                    "score": float(score),
                    "novelty_score": float(novelty),
                    "novelty_threshold": threshold,
                }
            )

        y_true = np.asarray(targets)
        y_pred = np.asarray(predictions)
        metrics = {
            "threshold": threshold,
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "unknown_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "unknown_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "unknown_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "known_accept_rate": float(np.mean(y_pred[y_true == 0] == 0)),
            "open_set_accuracy": float(np.mean(overall_correct)),
            "auroc": float(roc_auc_score(y_true, novelty_scores)),
            "n_known_test": int(np.sum(y_true == 0)),
            "n_unknown_test": int(np.sum(y_true == 1)),
        }
        results[method_name] = metrics

        if export_csv and out_dir:
            safe_name = method_name.replace(" ", "_").replace("/", "_")
            path = os.path.join(out_dir, f"open_set_predictions_{safe_name}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    return results
