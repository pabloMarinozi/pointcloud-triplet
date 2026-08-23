from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch

from src.evaluation.loader import (
    index_dataset_by_path,
    resolve_run,
    get_run_info,
    load_val_paths,
    load_test_paths,
    build_val_set,
    build_train_set,
    get_train_split_path,
    load_train_paths,
    list_ref_strategies,
    get_model_version,
)
from src.evaluation.embedding_cache import (
    ensure_embedding_cache,
    hash_samples,
    sha256_file,
)
from src.evaluation.metrics import default_methods
from src.evaluation.open_set import evaluate_open_set
from src.evaluation.report import evaluate_run_on_val, summarize_errors_by_class
from src.evaluation.ref_strategies import ensure_all_strategies_saved
from src.evaluation.video_index import load_video_index
from src.models.triplet import TripletNet
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser("Evaluate trained TripletNet runs on validation and/or test split.")
    p.add_argument("--data_dir", type=str, required=True, help="Directorio raíz con .ply (recursivo).")
    p.add_argument("--runs_dir", type=str, default="runs", help="Carpeta de experimentos.")
    p.add_argument("--run", type=str, default="latest", help="latest | all | <run_name>")
    p.add_argument(
        "--split",
        type=str,
        choices=("val", "test", "both", "select_and_test"),
        default="both",
        help="val | test | both | select_and_test (selecciona mejor por val, reporta solo test de ese run).",
    )
    p.add_argument(
        "--ref_strategy",
        type=str,
        choices=("all",),
        default="all",
        help="Estrategias de reference embeddings: all (todos los .npz en ep<N>/).",
    )
    p.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    p.add_argument("--export_csv", action="store_true", help="Exporta predicciones CSV dentro de cada run.")
    p.add_argument(
        "--open_set",
        action="store_true",
        help="Calibra el umbral con val y evalúa conocidos/desconocidos sobre test.",
    )
    p.add_argument("--use_augmentation", action="store_true", help="Augmentation al embeder (prueba robustez).")
    p.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
        help="Tamaño de batch para generar embeddings y caches.",
    )
    p.add_argument(
        "--embedding_views",
        type=int,
        choices=(1, 4, 8),
        default=1,
        help="Cantidad de vistas deterministas por nube.",
    )
    p.add_argument(
        "--view_aggregation",
        choices=("coordinate_median", "coordinate_mean"),
        default="coordinate_median",
        help="Agregación de embeddings cuando --embedding_views es 4 u 8.",
    )
    p.add_argument(
        "--index_videos",
        type=str,
        default="index_videos.csv",
        help="CSV con columnas video, forma (forma de captura). Si existe, se agregan columnas video y capture_form al CSV de predicciones.",
    )
    return p.parse_args()


def _save_evaluation_report(
    run_name: str,
    split: str,
    run_report_val: dict,
    run_report_test: dict,
    versioned_dir: str,
    evaluation_manifest: dict,
    evaluation_runtime: dict,
) -> None:
    """Guarda evaluation_report.json con el estado actual (val/test por estrategia)."""
    if not run_report_val and not run_report_test:
        return
    report = {
        "run_name": run_name,
        "split": split,
        "val": run_report_val,
        "test": run_report_test,
        "evaluation_manifest": evaluation_manifest,
        "runtime": evaluation_runtime,
    }
    if run_report_val:
        best_s, best_m = max(
            (
                (s, max(r, key=lambda m: r[m]["accuracy"]))
                for s, r in run_report_val.items()
            ),
            key=lambda x: run_report_val[x[0]][x[1]]["accuracy"],
        )
        report["best_val"] = {
            "strategy": best_s,
            "method": best_m,
            "accuracy": run_report_val[best_s][best_m]["accuracy"],
        }
    report_path = os.path.join(versioned_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  [Guardado] {report_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"Seed: {args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    view_aggregation = (
        "none" if args.embedding_views == 1 else args.view_aggregation
    )
    print(
        f"Vistas por nube: {args.embedding_views} | "
        f"agregación: {view_aggregation}"
    )

    # index del dataset (path -> true_label)
    dataset_index = index_dataset_by_path(args.data_dir)
    print(f"Dataset indexado: {len(dataset_index)} clouds")

    methods = default_methods()
    run_names = resolve_run(args.runs_dir, args.run)

    # Se usa siempre para que el caché conserve video y forma de captura.
    video_index = load_video_index(args.index_videos)
    if args.export_csv and video_index:
        print(f"Índice de videos: {len(video_index)} entradas (columnas video, capture_form en CSV)")

    global_best = (None, None, None, -1.0)  # (run, strategy, method, acc)

    for run_name in run_names:
        info = get_run_info(args.runs_dir, run_name)
        print("\n" + "=" * 80)
        print(f"EVALUANDO RUN: {run_name}")
        print("=" * 80)

        # Cargar config
        if not os.path.exists(info.config_path):
            print("⚠ No se encontró config.json. Saltando.")
            continue

        with open(info.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        n_points = int(config["n_points"])
        width = int(config["width"])
        configured_sampling = config.get("sampling")
        sampling = configured_sampling or "random"
        if configured_sampling is None:
            print("Sampling: random (fallback para run antiguo con sampling=null/ausente)")
        print(f"Config: width={width} | n_points={n_points} | sampling={sampling}")

        # Carpeta por versión del modelo (ep<N>) para no pisar evaluaciones anteriores
        version = get_model_version(info.exp_dir)
        if version is not None:
            versioned_dir = os.path.join(info.exp_dir, f"ep{version}")
            os.makedirs(versioned_dir, exist_ok=True)
            print(f"Version del modelo: ep{version} -> {versioned_dir}")
        else:
            versioned_dir = info.exp_dir

        evaluation_dir = versioned_dir
        if args.embedding_views > 1:
            evaluation_dir = os.path.join(
                versioned_dir,
                "evaluation_variants",
                f"views_{args.embedding_views}_{view_aggregation}",
            )
            os.makedirs(evaluation_dir, exist_ok=True)
            print(f"Artefactos multi-vista: {evaluation_dir}")

        # Estrategias de reference embeddings de esta variante de evaluación.
        strategies = list_ref_strategies(evaluation_dir)
        if not strategies:
            print("No hay reference_embeddings_*.npz en ep<N>/ - se generaran desde train_set despues de cargar el modelo.")
        else:
            print(f"Estrategias de referencia: {[s[0] for s in strategies]}")

        # Open-set siempre calibra con val y reporta una única vez sobre test.
        eval_val = not args.open_set and args.split in ("val", "both", "select_and_test")
        eval_test = (
            not args.open_set
            and args.split in ("test", "both")
            and args.split != "select_and_test"
        )

        val_set = []
        test_set = []
        unknown_val_set = []
        unknown_test_set = []
        if args.open_set:
            required_splits = (
                info.val_split_path,
                info.test_split_path,
                info.open_set_val_split_path,
                info.open_set_test_split_path,
                info.open_set_classes_path,
            )
            missing = [path for path in required_splits if not os.path.exists(path)]
            if missing:
                print("⚠ El run no contiene todos los artefactos open-set. Faltan:")
                for path in missing:
                    print(f"  - {path}")
                print("  Entrenalo con --open_set_classes N (N >= 2). Saltando.")
                continue

            val_set = build_val_set(
                dataset_index, load_val_paths(info.val_split_path)
            )
            test_set = build_val_set(
                dataset_index, load_test_paths(info.test_split_path)
            )
            unknown_val_set = build_val_set(
                dataset_index, load_val_paths(info.open_set_val_split_path)
            )
            unknown_test_set = build_val_set(
                dataset_index, load_test_paths(info.open_set_test_split_path)
            )
            print(
                "Open-set known val/test: "
                f"{len(val_set)} / {len(test_set)} samples"
            )
            print(
                "Open-set unknown calibration/test: "
                f"{len(unknown_val_set)} / {len(unknown_test_set)} samples"
            )
            if not all((val_set, test_set, unknown_val_set, unknown_test_set)):
                print("⚠ Uno o más splits open-set están vacíos o no pudieron mapearse al dataset. Saltando.")
                continue

        if eval_val:
            if not os.path.exists(info.val_split_path):
                print("⚠ No se encontró splits/val_paths.json. Saltando.")
                continue
            val_paths = load_val_paths(info.val_split_path)
            val_set = build_val_set(dataset_index, val_paths)
            print(f"Validation set: {len(val_set)} samples")
            if len(val_set) == 0:
                print("⚠ Validation vacío. Saltando.")
                continue

        if eval_test:
            if not os.path.exists(info.test_split_path):
                print("⚠ No se encontró splits/test_paths.json (run sin split 70/15/15?).")
                eval_test = False
            else:
                test_paths = load_test_paths(info.test_split_path)
                test_set = build_val_set(dataset_index, test_paths)
                print(f"Test set: {len(test_set)} samples")
                if len(test_set) == 0:
                    eval_test = False

        if not args.open_set and not eval_val and not eval_test:
            continue

        # Cargar modelo
        if not os.path.exists(info.model_path):
            print("⚠ No se encontró model_best.pt. Saltando.")
            continue

        model = TripletNet(width=width).to(device)
        state_dict = torch.load(info.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        checkpoint_sha256 = sha256_file(info.model_path)
        evaluation_manifest = {
            "checkpoint_sha256": checkpoint_sha256,
            "n_points": n_points,
            "sampling": sampling,
            "seed": args.seed,
            "use_augmentation": args.use_augmentation,
            "embedding_batch_size": args.embedding_batch_size,
            "embedding_views": args.embedding_views,
            "view_aggregation": view_aggregation,
            "query_splits_sha256": {
                name: hash_samples(samples)
                for name, samples in (
                    ("val", val_set),
                    ("test", test_set),
                    ("open_set_unknown_val", unknown_val_set),
                    ("open_set_unknown_test", unknown_test_set),
                )
                if samples
            },
        }
        evaluation_runtime = {
            "reference_generation": {},
            "embedding_caches": {},
            "classification": {"val": {}, "test": {}},
        }

        # Guardar copia del modelo en ep<N>/ para poder recuperar esta versión si seguís entrenando
        if versioned_dir != info.exp_dir:
            model_snapshot_path = os.path.join(versioned_dir, "model.pt")
            shutil.copy2(info.model_path, model_snapshot_path)
            print(f"  Modelo guardado en ep{version}/model.pt")

        # El manifiesto decide si el caché/las referencias se pueden reutilizar.
        train_split_path = get_train_split_path(info.exp_dir)
        if os.path.exists(train_split_path):
            train_paths = load_train_paths(train_split_path)
            train_set = build_train_set(dataset_index, train_paths)
            if train_set:
                evaluation_manifest["train_split_paths_sha256"] = hash_samples(
                    train_set
                )
                ensure_all_strategies_saved(
                    evaluation_dir,
                    model,
                    train_set,
                    n_points,
                    device,
                    checkpoint_path=info.model_path,
                    checkpoint_sha256=checkpoint_sha256,
                    sampling=sampling,
                    seed=args.seed,
                    use_augmentation=args.use_augmentation,
                    batch_size=args.embedding_batch_size,
                    views=args.embedding_views,
                    view_aggregation=view_aggregation,
                    video_index=video_index,
                    runtime_stats=evaluation_runtime["reference_generation"],
                )
                strategies = list_ref_strategies(evaluation_dir)
                print(f"Estrategias de referencia: {[s[0] for s in strategies]}")

        val_embeddings = None
        test_embeddings = None
        if eval_val and val_set:
            val_embeddings = ensure_embedding_cache(
                cache_dir=evaluation_dir,
                split="val",
                model=model,
                samples=val_set,
                n_points=n_points,
                device=device,
                checkpoint_path=info.model_path,
                checkpoint_sha256=checkpoint_sha256,
                sampling=sampling,
                seed=args.seed,
                use_augmentation=args.use_augmentation,
                batch_size=args.embedding_batch_size,
                views=args.embedding_views,
                view_aggregation=view_aggregation,
                video_index=video_index,
                runtime_stats=evaluation_runtime["embedding_caches"].setdefault(
                    "val", {}
                ),
            )
        if eval_test and test_set:
            test_embeddings = ensure_embedding_cache(
                cache_dir=evaluation_dir,
                split="test",
                model=model,
                samples=test_set,
                n_points=n_points,
                device=device,
                checkpoint_path=info.model_path,
                checkpoint_sha256=checkpoint_sha256,
                sampling=sampling,
                seed=args.seed,
                use_augmentation=args.use_augmentation,
                batch_size=args.embedding_batch_size,
                views=args.embedding_views,
                view_aggregation=view_aggregation,
                video_index=video_index,
                runtime_stats=evaluation_runtime["embedding_caches"].setdefault(
                    "test", {}
                ),
            )

        if args.open_set:
            if not strategies:
                print("⚠ No se pudieron crear estrategias de referencia. Saltando.")
                continue

            known_val_embeddings = ensure_embedding_cache(
                cache_dir=evaluation_dir, split="open_set_known_val", model=model,
                samples=val_set, n_points=n_points, device=device,
                checkpoint_path=info.model_path, checkpoint_sha256=checkpoint_sha256,
                sampling=sampling, seed=args.seed,
                use_augmentation=args.use_augmentation,
                batch_size=args.embedding_batch_size,
                views=args.embedding_views, view_aggregation=view_aggregation,
                video_index=video_index,
            )
            unknown_val_embeddings = ensure_embedding_cache(
                cache_dir=evaluation_dir, split="open_set_unknown_val", model=model,
                samples=unknown_val_set, n_points=n_points, device=device,
                checkpoint_path=info.model_path, checkpoint_sha256=checkpoint_sha256,
                sampling=sampling, seed=args.seed,
                use_augmentation=args.use_augmentation,
                batch_size=args.embedding_batch_size,
                views=args.embedding_views, view_aggregation=view_aggregation,
                video_index=video_index,
            )
            known_test_embeddings = ensure_embedding_cache(
                cache_dir=evaluation_dir, split="open_set_known_test", model=model,
                samples=test_set, n_points=n_points, device=device,
                checkpoint_path=info.model_path, checkpoint_sha256=checkpoint_sha256,
                sampling=sampling, seed=args.seed,
                use_augmentation=args.use_augmentation,
                batch_size=args.embedding_batch_size,
                views=args.embedding_views, view_aggregation=view_aggregation,
                video_index=video_index,
            )
            unknown_test_embeddings = ensure_embedding_cache(
                cache_dir=evaluation_dir, split="open_set_unknown_test", model=model,
                samples=unknown_test_set, n_points=n_points, device=device,
                checkpoint_path=info.model_path, checkpoint_sha256=checkpoint_sha256,
                sampling=sampling, seed=args.seed,
                use_augmentation=args.use_augmentation,
                batch_size=args.embedding_batch_size,
                views=args.embedding_views, view_aggregation=view_aggregation,
                video_index=video_index,
            )
            open_set_report = {}
            open_set_report_path = os.path.join(evaluation_dir, "open_set_report.json")
            for strategy_name, ref_path in strategies:
                ref_data = np.load(ref_path)
                reference_embeddings = {k: ref_data[k] for k in ref_data.files}
                out_dir = (
                    os.path.join(evaluation_dir, "evaluation_open_set", strategy_name)
                    if args.export_csv
                    else None
                )
                print(
                    f"\n--- Open-set ref: {strategy_name} "
                    f"({len(reference_embeddings)} clases conocidas) ---"
                )
                results = evaluate_open_set(
                    model=model,
                    reference_embeddings=reference_embeddings,
                    known_val_set=val_set,
                    unknown_val_set=unknown_val_set,
                    known_test_set=test_set,
                    unknown_test_set=unknown_test_set,
                    methods=methods,
                    n_points=n_points,
                    device=device,
                    use_augmentation=args.use_augmentation,
                    export_csv=args.export_csv,
                    out_dir=out_dir,
                    sampling=sampling,
                    seed=args.seed,
                    batch_size=args.embedding_batch_size,
                    views=args.embedding_views,
                    view_aggregation=view_aggregation,
                    calibration_embeddings=(
                        known_val_embeddings + unknown_val_embeddings
                    ),
                    test_embeddings=(
                        known_test_embeddings + unknown_test_embeddings
                    ),
                )
                open_set_report[strategy_name] = results
                for method_name in sorted(
                    results,
                    key=lambda name: results[name]["balanced_accuracy"],
                    reverse=True,
                ):
                    metrics = results[method_name]
                    print(
                        f"  {method_name:<22} "
                        f"bal_acc={metrics['balanced_accuracy']:.4f}  "
                        f"unknown_recall={metrics['unknown_recall']:.4f}  "
                        f"known_accept={metrics['known_accept_rate']:.4f}  "
                        f"open_acc={metrics['open_set_accuracy']:.4f}  "
                        f"auroc={metrics['auroc']:.4f}  "
                        f"threshold={metrics['threshold']:.6g}"
                    )

                with open(open_set_report_path, "w", encoding="utf-8") as f:
                    json.dump(open_set_report, f, indent=2, ensure_ascii=False)

            print(f"\n  [Guardado] {open_set_report_path}")
            continue

        # Cargar reporte existente para reanudar desde donde quedo (no recalcular estrategias ya guardadas)
        run_report_val = {}
        run_report_test = {}
        report_path = os.path.join(evaluation_dir, "evaluation_report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if (
                    existing.get("split") == args.split
                    and existing.get("evaluation_manifest") == evaluation_manifest
                ):
                    run_report_val = existing.get("val") or {}
                    run_report_test = existing.get("test") or {}
                    existing_runtime = existing.get("runtime") or {}
                    for section in ("reference_generation", "embedding_caches"):
                        if existing_runtime.get(section):
                            evaluation_runtime[section] = existing_runtime[section]
                    existing_classification = existing_runtime.get("classification") or {}
                    for split_name in ("val", "test"):
                        evaluation_runtime["classification"][split_name].update(
                            existing_classification.get(split_name) or {}
                        )
                    if run_report_val or run_report_test:
                        print(f"  Reanudando: {len(run_report_val)} estrategias ya evaluadas en val, {len(run_report_test)} en test.")
                    if existing.get("best_val") and run_report_val:
                        b = existing["best_val"]
                        acc = float(b["accuracy"])
                        if acc > global_best[3]:
                            global_best = (run_name, b["strategy"], b["method"], acc)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        for strategy_name, ref_path in strategies:
            if not os.path.exists(ref_path):
                continue
            # Saltar estrategia ya evaluada (reanudar sin recalcular)
            already_val = strategy_name in run_report_val
            already_test = strategy_name in run_report_test
            if already_val and (already_test or not eval_test):
                print(f"\n--- Ref: {strategy_name} (ya evaluada, omitiendo) ---")
                if run_report_val.get(strategy_name) and global_best[0] == run_name:
                    best_m = max(
                        run_report_val[strategy_name],
                        key=lambda m: run_report_val[strategy_name][m]["accuracy"],
                    )
                    acc = run_report_val[strategy_name][best_m]["accuracy"]
                    if acc > global_best[3]:
                        global_best = (run_name, strategy_name, best_m, acc)
                continue
            ref_data = np.load(ref_path)
            reference_embeddings = {k: ref_data[k] for k in ref_data.files}
            print(f"\n--- Ref: {strategy_name} ({len(reference_embeddings)} clases) ---")

            # Evaluar en val
            if eval_val and val_set:
                out_dir_val = os.path.join(evaluation_dir, "evaluation", strategy_name) if args.export_csv else None
                if out_dir_val:
                    os.makedirs(out_dir_val, exist_ok=True)
                results_val = evaluate_run_on_val(
                    model=model,
                    reference_embeddings=reference_embeddings,
                    val_set=val_set,
                    methods=methods,
                    n_points=n_points,
                    device=device,
                    use_augmentation=args.use_augmentation,
                    export_csv=args.export_csv,
                    out_dir=out_dir_val,
                    video_index=video_index,
                    sampling=sampling,
                    seed=args.seed,
                    batch_size=args.embedding_batch_size,
                    views=args.embedding_views,
                    view_aggregation=view_aggregation,
                    precomputed_embeddings=val_embeddings,
                    runtime_stats=evaluation_runtime["classification"]["val"].setdefault(
                        strategy_name, {}
                    ),
                )
                run_report_val[strategy_name] = {
                    k: {mk: float(mv) for mk, mv in v.items()}
                    for k, v in results_val.items()
                }
                print("  VAL:")
                best_method_val = max(
                    results_val, key=lambda m: results_val[m]["accuracy"]
                )
                best_acc_val = results_val[best_method_val]["accuracy"]
                for method_name in sorted(
                    results_val.keys(),
                    key=lambda m: results_val[m]["accuracy"],
                    reverse=True,
                ):
                    met = results_val[method_name]
                    print(
                        f"    {method_name:<22}  acc={met['accuracy']:.4f}  "
                        f"top5={met['top5_accuracy']:.4f}  mrr={met['mrr']:.4f}  mean_rank={met['mean_rank']:.1f}"
                    )
                print(f"  BEST (val): {best_method_val}  acc={best_acc_val:.4f}")
                if best_acc_val > global_best[3]:
                    global_best = (run_name, strategy_name, best_method_val, best_acc_val)
                if args.export_csv and out_dir_val:
                    df_err = summarize_errors_by_class(out_dir_val)
                    df_err_sorted = df_err.sort_values(["error_rate"], ascending=False).head(10)
                    print("  Top-10 (metric,class) por tasa de error (val):")
                    print(df_err_sorted.to_string(index=False))

            # Evaluar en test
            if eval_test and test_set:
                out_dir_test = os.path.join(evaluation_dir, "evaluation_test", strategy_name) if args.export_csv else None
                if out_dir_test:
                    os.makedirs(out_dir_test, exist_ok=True)
                results_test = evaluate_run_on_val(
                    model=model,
                    reference_embeddings=reference_embeddings,
                    val_set=test_set,
                    methods=methods,
                    n_points=n_points,
                    device=device,
                    use_augmentation=args.use_augmentation,
                    export_csv=args.export_csv,
                    out_dir=out_dir_test,
                    video_index=video_index,
                    sampling=sampling,
                    seed=args.seed,
                    batch_size=args.embedding_batch_size,
                    views=args.embedding_views,
                    view_aggregation=view_aggregation,
                    precomputed_embeddings=test_embeddings,
                    runtime_stats=evaluation_runtime["classification"]["test"].setdefault(
                        strategy_name, {}
                    ),
                )
                run_report_test[strategy_name] = {
                    k: {mk: float(mv) for mk, mv in v.items()}
                    for k, v in results_test.items()
                }
                print("  TEST:")
                best_method_test = max(
                    results_test, key=lambda m: results_test[m]["accuracy"]
                )
                best_acc_test = results_test[best_method_test]["accuracy"]
                for method_name in sorted(
                    results_test.keys(),
                    key=lambda m: results_test[m]["accuracy"],
                    reverse=True,
                ):
                    met = results_test[method_name]
                    print(
                        f"    {method_name:<22}  acc={met['accuracy']:.4f}  "
                        f"top5={met['top5_accuracy']:.4f}  mrr={met['mrr']:.4f}  mean_rank={met['mean_rank']:.1f}"
                    )
                print(f"  BEST (test): {best_method_test}  acc={best_acc_test:.4f}")
                if args.export_csv and out_dir_test:
                    df_err = summarize_errors_by_class(out_dir_test)
                    df_err_sorted = df_err.sort_values(["error_rate"], ascending=False).head(10)
                    print("  Top-10 (metric,class) por tasa de error (test):")
                    print(df_err_sorted.to_string(index=False))

            # Guardar reporte al terminar cada estrategia (para no perder resultados si se interrumpe)
            _save_evaluation_report(
                run_name,
                args.split,
                run_report_val,
                run_report_test,
                evaluation_dir,
                evaluation_manifest,
                evaluation_runtime,
            )

    if args.open_set:
        return

    # Global best (o protocolo select_and_test)
    if args.split == "select_and_test" and global_best[0] is not None:
        run_name, best_strategy, best_method_val, best_acc_val = global_best
        info = get_run_info(args.runs_dir, run_name)
        version = get_model_version(info.exp_dir)
        versioned_dir = os.path.join(info.exp_dir, f"ep{version}") if version is not None else info.exp_dir
        evaluation_dir = versioned_dir
        if args.embedding_views > 1:
            evaluation_dir = os.path.join(
                versioned_dir,
                "evaluation_variants",
                f"views_{args.embedding_views}_{view_aggregation}",
            )
        ref_path = os.path.join(
            evaluation_dir, f"reference_embeddings_{best_strategy}.npz"
        )
        if not os.path.exists(info.test_split_path):
            print("\n" + "=" * 80)
            print("SELECCION POR VAL -> TEST")
            print("=" * 80)
            print(f"Selected run: {run_name} (ref={best_strategy}, {best_method_val}, val acc: {best_acc_val:.4f})")
            print("⚠ No se encontró splits/test_paths.json para este run. No se evaluó test.")
        elif not os.path.exists(ref_path):
            print("\n" + "=" * 80)
            print("SELECCION POR VAL -> TEST")
            print("=" * 80)
            print(f"Selected run: {run_name} (ref={best_strategy}, {best_method_val}, val acc: {best_acc_val:.4f})")
            print(f"⚠ No se encontró {ref_path}. No se evaluó test.")
        else:
            with open(info.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            n_points = int(config["n_points"])
            width = int(config["width"])
            sampling = config.get("sampling") or "random"
            test_paths = load_test_paths(info.test_split_path)
            test_set = build_val_set(dataset_index, test_paths)
            if len(test_set) == 0:
                print("\n" + "=" * 80)
                print("SELECCION POR VAL -> TEST")
                print("=" * 80)
                print(f"Selected run: {run_name} (ref={best_strategy}, {best_method_val}, val acc: {best_acc_val:.4f})")
                print("⚠ Test set vacío.")
            else:
                model = TripletNet(width=width).to(device)
                state_dict = torch.load(info.model_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                checkpoint_sha256 = sha256_file(info.model_path)
                test_embeddings = ensure_embedding_cache(
                    cache_dir=evaluation_dir,
                    split="test",
                    model=model,
                    samples=test_set,
                    n_points=n_points,
                    device=device,
                    checkpoint_path=info.model_path,
                    checkpoint_sha256=checkpoint_sha256,
                    sampling=sampling,
                    seed=args.seed,
                    use_augmentation=args.use_augmentation,
                    batch_size=args.embedding_batch_size,
                    views=args.embedding_views,
                    view_aggregation=view_aggregation,
                    video_index=video_index,
                )
                ref_data = np.load(ref_path)
                reference_embeddings = {k: ref_data[k] for k in ref_data.files}
                out_dir_test = os.path.join(evaluation_dir, "evaluation_test", best_strategy) if args.export_csv else None
                if out_dir_test:
                    os.makedirs(out_dir_test, exist_ok=True)
                results_test = evaluate_run_on_val(
                    model=model,
                    reference_embeddings=reference_embeddings,
                    val_set=test_set,
                    methods=methods,
                    n_points=n_points,
                    device=device,
                    use_augmentation=args.use_augmentation,
                    export_csv=args.export_csv,
                    out_dir=out_dir_test,
                    video_index=video_index,
                    sampling=sampling,
                    seed=args.seed,
                    batch_size=args.embedding_batch_size,
                    views=args.embedding_views,
                    view_aggregation=view_aggregation,
                    precomputed_embeddings=test_embeddings,
                )
                best_method_test = max(
                    results_test,
                    key=lambda method: results_test[method]["accuracy"],
                )
                acc_test_selected = results_test[best_method_val]["accuracy"]
                acc_test_best = results_test[best_method_test]["accuracy"]
                print("\n" + "=" * 80)
                print("SELECCION POR VAL -> TEST")
                print("=" * 80)
                print(f"Selected run: {run_name} (ref={best_strategy}, {best_method_val}, val acc: {best_acc_val:.4f})")
                print(f"Test accuracy (métrica usada en val): {best_method_val} = {acc_test_selected:.4f}")
                print(f"Test accuracy (mejor método en test): {best_method_test} = {acc_test_best:.4f}")
    else:
        print("\n" + "=" * 80)
        print("MEJOR RUN GLOBAL")
        print("=" * 80)
        if global_best[0] is None:
            print("No se evaluó ningún run correctamente.")
        else:
            run_name, strategy_name, method_name, acc = global_best
            print(f"{run_name}  |  ref={strategy_name}  |  {method_name}  |  acc={acc:.4f}")


if __name__ == "__main__":
    main()
