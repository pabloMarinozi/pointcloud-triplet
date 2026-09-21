"""Ejecuta y consolida la comparación de reference embeddings de P1."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

from src.evaluation.loader import get_model_version, get_run_info


P1_VIEWS = (1, 4, 8)
P1_STRATEGIES = (
    "centroid_all",
    "median_all",
    "trimmed_mean_05",
    "trimmed_mean_10",
)
P1_METHOD = "L1 Distance"


def parse_args():
    parser = argparse.ArgumentParser(
        "Ejecutar la matriz P1 sobre validación y consolidar sus métricas."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_batch_size", type=int, default=512)
    parser.add_argument(
        "--view_aggregation",
        choices=("coordinate_median", "coordinate_mean"),
        default="coordinate_median",
    )
    parser.add_argument(
        "--only_consolidate",
        action="store_true",
        help="No ejecuta inferencia; consolida reportes P1 que ya existan.",
    )
    return parser.parse_args()


def _variant_dir(versioned_dir: str, views: int, aggregation: str) -> str:
    if views == 1:
        return versioned_dir
    return os.path.join(
        versioned_dir,
        "evaluation_variants",
        f"views_{views}_{aggregation}",
    )


def _evaluation_command(args, views: int) -> List[str]:
    return [
        sys.executable,
        "-m",
        "src.eval",
        "--data_dir",
        args.data_dir,
        "--runs_dir",
        args.runs_dir,
        "--run",
        args.run,
        "--split",
        "val",
        "--seed",
        str(args.seed),
        "--embedding_batch_size",
        str(args.embedding_batch_size),
        "--embedding_views",
        str(views),
        "--view_aggregation",
        args.view_aggregation,
    ]


def _load_variant(report_path: str, views: int) -> Dict[str, object]:
    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    strategies = {}
    val_results = report.get("val") or {}
    for strategy in P1_STRATEGIES:
        metrics = (val_results.get(strategy) or {}).get(P1_METHOD)
        if metrics:
            strategies[strategy] = {
                key: float(metrics[key])
                for key in ("accuracy", "top5_accuracy", "mrr")
            }

    runtime = report.get("runtime") or {}
    reference_runtime = runtime.get("reference_generation") or {}
    train_cache = reference_runtime.get("train_cache") or {}
    val_cache = (runtime.get("embedding_caches") or {}).get("val") or {}
    classification = (runtime.get("classification") or {}).get("val") or {}
    l1_latencies = [
        float(methods[P1_METHOD]["latency_ms_per_query"])
        for strategy, methods in classification.items()
        if strategy in P1_STRATEGIES and P1_METHOD in methods
    ]
    return {
        "views": views,
        "manifest": report.get("evaluation_manifest") or {},
        "strategies": strategies,
        "runtime": {
            "reference_seconds": float(
                reference_runtime.get("elapsed_seconds", 0.0)
            ),
            "train_embedding_seconds": float(train_cache.get("elapsed_seconds", 0.0)),
            "validation_embedding_seconds": float(
                val_cache.get("elapsed_seconds", 0.0)
            ),
            "peak_process_rss_mb": max(
                float(train_cache.get("peak_process_rss_mb", 0.0)),
                float(val_cache.get("peak_process_rss_mb", 0.0)),
            ),
            "peak_cuda_memory_mb": max(
                float(train_cache.get("peak_cuda_memory_mb", 0.0)),
                float(val_cache.get("peak_cuda_memory_mb", 0.0)),
            ),
            "mean_l1_latency_ms_per_query": (
                sum(l1_latencies) / len(l1_latencies) if l1_latencies else 0.0
            ),
        },
    }


def _markdown_report(comparison: Dict[str, object]) -> str:
    lines = [
        "# Comparación P1 sobre validación",
        "",
        "| Vistas | Estrategia | Accuracy | Top-5 | MRR | "
        "Emb. train (s) | Emb. val (s) | Latencia L1 (ms/query) | Pico RAM (MiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in comparison["variants"]:
        runtime = variant["runtime"]
        for strategy, metrics in variant["strategies"].items():
            lines.append(
                f"| {variant['views']} | `{strategy}` | "
                f"{metrics['accuracy']:.4f} | {metrics['top5_accuracy']:.4f} | "
                f"{metrics['mrr']:.4f} | {runtime['train_embedding_seconds']:.1f} | "
                f"{runtime['validation_embedding_seconds']:.1f} | "
                f"{runtime['mean_l1_latency_ms_per_query']:.3f} | "
                f"{runtime['peak_process_rss_mb']:.1f} |"
            )
    lines.extend(
        [
            "",
            "La selección debe hacerse con validation. Este script no ejecuta test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    info = get_run_info(args.runs_dir, args.run)
    version = get_model_version(info.exp_dir)
    versioned_dir = (
        os.path.join(info.exp_dir, f"ep{version}")
        if version is not None
        else info.exp_dir
    )

    if not args.only_consolidate:
        for views in P1_VIEWS:
            command = _evaluation_command(args, views)
            print(f"\n[P1] Ejecutando M={views}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)

    variants = []
    missing = []
    for views in P1_VIEWS:
        report_path = os.path.join(
            _variant_dir(versioned_dir, views, args.view_aggregation),
            "evaluation_report.json",
        )
        if not os.path.exists(report_path):
            missing.append(report_path)
            continue
        variants.append(_load_variant(report_path, views))

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Faltan reportes P1:\n{formatted}")

    comparison = {
        "run": args.run,
        "split": "val",
        "seed": args.seed,
        "view_aggregation": args.view_aggregation,
        "variants": variants,
    }
    json_path = os.path.join(versioned_dir, "p1_comparison.json")
    markdown_path = os.path.join(versioned_dir, "p1_comparison.md")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, ensure_ascii=False)
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write(_markdown_report(comparison))
    print(f"\n[P1] Comparación guardada en {json_path} y {markdown_path}")


if __name__ == "__main__":
    main()
