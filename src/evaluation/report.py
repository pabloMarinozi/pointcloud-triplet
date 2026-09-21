from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.evaluation.embed import embed_point_cloud_paths_multiview
from src.evaluation.metrics import Method, predict_class, rank_of_label
from src.evaluation.runtime_stats import peak_process_memory_mb
from src.evaluation.video_index import get_video_and_capture_form


def _ranking_metrics(ranks: List[int], n: int) -> Dict[str, float]:
    """A partir de una lista de rank_true (1-based), calcula métricas de ranking."""
    valid = [r for r in ranks if r >= 1]
    if not valid:
        return {
            "accuracy": 0.0,
            "top5_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
        }
    n_valid = len(valid)
    return {
        "accuracy": sum(1 for r in valid if r == 1) / n,
        "top5_accuracy": sum(1 for r in valid if r <= 5) / n,
        "top10_accuracy": sum(1 for r in valid if r <= 10) / n,
        "mrr": (sum(1.0 / r for r in valid) / n),
        "mean_rank": float(np.mean(valid)),
        "median_rank": float(np.median(valid)),
    }


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
    video_index: Dict[str, str] | None = None,
    sampling: str = "random",
    seed: int = 42,
    batch_size: int = 64,
    views: int = 1,
    view_aggregation: str = "none",
    precomputed_embeddings: List[Tuple[str, str, np.ndarray]] | None = None,
    runtime_stats: Dict[str, object] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    val_set: lista de (true_label, path)
    Retorna por método un dict con: accuracy, top5_accuracy, top10_accuracy, mrr, mean_rank, median_rank.
    Cada muestra de validación se embeddea una sola vez; luego se evalúan todos los métodos.
    """
    if export_csv and out_dir is None:
        raise ValueError("Si export_csv=True, out_dir no puede ser None.")

    if export_csv:
        os.makedirs(out_dir, exist_ok=True)

    # Una sola pasada (o caché): embeddar cada muestra una vez para todos los métodos.
    if precomputed_embeddings is None:
        val_embeddings = embed_point_cloud_paths_multiview(
            model=model,
            samples=val_set,
            n_points=n_points,
            device=device,
            use_augmentation=use_augmentation,
            sampling=sampling,
            seed=seed,
            views=views,
            view_aggregation=view_aggregation,
            batch_size=batch_size,
        )
    else:
        val_embeddings = precomputed_embeddings

    n = len(val_embeddings)
    y_true = [cls for cls, _, _ in val_embeddings]
    results: Dict[str, Dict[str, float]] = {}

    for method_name, method in methods.items():
        started = time.perf_counter()
        y_pred = []
        rows = []
        rank_trues: List[int] = []

        for true_label, path, emb in val_embeddings:
            pred_label, score = predict_class(emb, reference_embeddings, method)
            y_pred.append(pred_label)
            rank_true = rank_of_label(
                emb, reference_embeddings, method, true_label
            )
            rank_trues.append(rank_true)

            if export_csv:
                video_name, capture_form = get_video_and_capture_form(
                    path, video_index
                )
                row = {
                    "path": path,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": int(pred_label == true_label),
                    "score": float(score),
                    "rank_true": rank_true,
                }
                if video_index is not None:
                    row["video"] = video_name
                    row["capture_form"] = capture_form
                rows.append(row)

        acc = float(accuracy_score(y_true, y_pred))
        metrics = _ranking_metrics(rank_trues, n)
        metrics["accuracy"] = acc  # por si hay rank 0, mantener coherencia con accuracy_score
        results[method_name] = metrics
        elapsed_seconds = time.perf_counter() - started
        if runtime_stats is not None:
            runtime_stats[method_name] = {
                "query_count": n,
                "elapsed_seconds": elapsed_seconds,
                "latency_ms_per_query": (
                    elapsed_seconds * 1000.0 / n if n else 0.0
                ),
                "peak_process_rss_mb": peak_process_memory_mb(),
            }

        if export_csv:
            safe_name = method_name.replace(" ", "_").replace("/", "_")
            csv_path = os.path.join(out_dir, f"validation_predictions_{safe_name}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "path",
                    "true_label",
                    "pred_label",
                    "correct",
                    "score",
                    "rank_true",
                ]
                if video_index is not None:
                    fieldnames.extend(["video", "capture_form"])
                writer = csv.DictWriter(f, fieldnames=fieldnames)
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
