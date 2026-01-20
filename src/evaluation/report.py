from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.evaluation.embed import embed_point_cloud_path
from src.evaluation.metrics import Method, predict_class


def evaluate_run_on_val(
    model,
    reference_embeddings: Dict[str, np.ndarray],
    val_set: List[Tuple[str, str]],
    methods: Dict[str, Method],
    n_points: int,
    device,
    use_augmentation: bool = False,
    export_csv: bool = True,
    out_dir: str | None = None,
) -> Dict[str, float]:
    """
    val_set: lista de (true_label, path)
    Retorna accuracies por método.
    """
    y_true = [cls for cls, _ in val_set]
    results: Dict[str, float] = {}

    if export_csv and out_dir is None:
        raise ValueError("Si export_csv=True, out_dir no puede ser None.")

    if export_csv:
        os.makedirs(out_dir, exist_ok=True)

    for method_name, method in methods.items():
        y_pred = []
        rows = []

        for true_label, path in val_set:
            emb = embed_point_cloud_path(
                model=model,
                ply_path=path,
                n_points=n_points,
                device=device,
                use_augmentation=use_augmentation,
            )

            pred_label, score = predict_class(emb, reference_embeddings, method)
            y_pred.append(pred_label)

            if export_csv:
                rows.append(
                    {
                        "path": path,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "correct": int(pred_label == true_label),
                        "score": float(score),
                    }
                )

        acc = float(accuracy_score(y_true, y_pred))
        results[method_name] = acc

        if export_csv:
            safe_name = method_name.replace(" ", "_").replace("/", "_")
            csv_path = os.path.join(out_dir, f"validation_predictions_{safe_name}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["path", "true_label", "pred_label", "correct", "score"],
                )
                writer.writeheader()
                writer.writerows(rows)

    return results


def summarize_errors_by_class(eval_dir: str) -> pd.DataFrame:
    """
    Lee los CSVs exportados y devuelve una tabla con tasa de error por clase y métrica.
    """
    errors_by_metric = defaultdict(lambda: defaultdict(int))
    total_by_class = defaultdict(int)

    if not os.path.isdir(eval_dir):
        raise ValueError(f"No existe eval_dir: {eval_dir}")

    for fname in os.listdir(eval_dir):
        if not fname.endswith(".csv"):
            continue

        metric_name = (
            fname.replace("validation_predictions_", "")
            .replace(".csv", "")
        )

        df = pd.read_csv(os.path.join(eval_dir, fname))

        # total por clase
        for cls, cnt in df["true_label"].value_counts().items():
            total_by_class[cls] += int(cnt)

        # errores por clase
        df_err = df[df["correct"] == 0]
        for cls, cnt in df_err["true_label"].value_counts().items():
            errors_by_metric[metric_name][cls] += int(cnt)

    classes = sorted(total_by_class.keys())
    metrics = sorted(errors_by_metric.keys())

    rows = []
    for metric in metrics:
        for cls in classes:
            err = errors_by_metric[metric].get(cls, 0)
            total = total_by_class.get(cls, 0)
            rate = (err / total) if total > 0 else 0.0

            rows.append(
                {
                    "metric": metric,
                    "class": cls,
                    "errors": err,
                    "total": total,
                    "error_rate": rate,
                }
            )

    return pd.DataFrame(rows)
