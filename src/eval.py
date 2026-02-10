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
from src.evaluation.metrics import default_methods
from src.evaluation.report import evaluate_run_on_val, summarize_errors_by_class
from src.evaluation.ref_strategies import ensure_all_strategies_saved
from src.evaluation.video_index import load_video_index
from src.models.triplet import TripletNet


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
        choices=("all", "train"),
        default="all",
        help="Estrategias de reference embeddings: all (todas los .npz en el run) o train (solo reference_embeddings_train.npz).",
    )
    p.add_argument("--export_csv", action="store_true", help="Exporta predicciones CSV dentro de cada run.")
    p.add_argument("--use_augmentation", action="store_true", help="Augmentation al embeder (prueba robustez).")
    p.add_argument(
        "--index_videos",
        type=str,
        default="index_videos.csv",
        help="CSV con columnas video, forma (forma de captura). Si existe, se agregan columnas video y capture_form al CSV de predicciones.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # index del dataset (path -> true_label)
    dataset_index = index_dataset_by_path(args.data_dir)
    print(f"Dataset indexado: {len(dataset_index)} clouds")

    methods = default_methods()
    run_names = resolve_run(args.runs_dir, args.run)

    # Índice video -> forma de captura (para columnas video y capture_form en --export_csv)
    video_index = load_video_index(args.index_videos) if args.export_csv else None
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
        print(f"Config: width={width} | n_points={n_points}")

        # Carpeta por versión del modelo (ep<N>) para no pisar evaluaciones anteriores
        version = get_model_version(info.exp_dir)
        if version is not None:
            versioned_dir = os.path.join(info.exp_dir, f"ep{version}")
            os.makedirs(versioned_dir, exist_ok=True)
            print(f"Versión del modelo: ep{version} → {versioned_dir}")
        else:
            versioned_dir = info.exp_dir

        # Reference embeddings: en versioned_dir; si no existen, copiar train desde root
        train_ref_path = os.path.join(versioned_dir, "reference_embeddings_train.npz")
        if versioned_dir != info.exp_dir and not os.path.exists(train_ref_path) and os.path.exists(info.ref_emb_path):
            shutil.copy2(info.ref_emb_path, train_ref_path)
            print(f"  Copiado reference_embeddings_train.npz a ep{version}/")

        # Estrategias de reference embeddings (desde versioned_dir)
        if args.ref_strategy == "train":
            strategies = [("train", train_ref_path)] if os.path.exists(train_ref_path) else []
        else:
            strategies = list_ref_strategies(versioned_dir, train_ref_path)
        if not strategies:
            print("⚠ No se encontró ningún reference_embeddings_*.npz. Saltando.")
            continue
        print(f"Estrategias de referencia: {[s[0] for s in strategies]}")

        # Cargar splits según --split (select_and_test: solo val en esta pasada)
        eval_val = args.split in ("val", "both", "select_and_test")
        eval_test = args.split in ("test", "both") and args.split != "select_and_test"

        val_set = []
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

        test_set = []
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

        if not eval_val and not eval_test:
            continue

        # Cargar modelo
        if not os.path.exists(info.model_path):
            print("⚠ No se encontró model_best.pt. Saltando.")
            continue

        model = TripletNet(width=width).to(device)
        state_dict = torch.load(info.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Guardar copia del modelo en ep<N>/ para poder recuperar esta versión si seguís entrenando
        if versioned_dir != info.exp_dir:
            model_snapshot_path = os.path.join(versioned_dir, "model.pt")
            shutil.copy2(info.model_path, model_snapshot_path)
            print(f"  Modelo guardado en ep{version}/model.pt")

        # Si solo existe "train" y queremos todas las estrategias, generarlas ahora
        if (
            args.ref_strategy == "all"
            and len(strategies) == 1
            and strategies[0][0] == "train"
        ):
            train_split_path = get_train_split_path(info.exp_dir)
            if os.path.exists(train_split_path):
                train_paths = load_train_paths(train_split_path)
                train_set = build_train_set(dataset_index, train_paths)
                if train_set:
                    ensure_all_strategies_saved(
                        versioned_dir, model, train_set, n_points, device
                    )
                    strategies = list_ref_strategies(versioned_dir, train_ref_path)
                    print(f"Estrategias de referencia: {[s[0] for s in strategies]}")

        run_report_val = {}
        run_report_test = {}

        for strategy_name, ref_path in strategies:
            if not os.path.exists(ref_path):
                continue
            ref_data = np.load(ref_path)
            reference_embeddings = {k: ref_data[k] for k in ref_data.files}
            print(f"\n--- Ref: {strategy_name} ({len(reference_embeddings)} clases) ---")

            # Evaluar en val
            if eval_val and val_set:
                out_dir_val = os.path.join(versioned_dir, "evaluation", strategy_name) if args.export_csv else None
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
                out_dir_test = os.path.join(versioned_dir, "evaluation_test", strategy_name) if args.export_csv else None
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

        # Guardar resumen de evaluación en el run (siempre)
        if run_report_val or run_report_test:
            report = {
                "run_name": run_name,
                "split": args.split,
                "val": run_report_val,
                "test": run_report_test,
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

    # Global best (o protocolo select_and_test)
    if args.split == "select_and_test" and global_best[0] is not None:
        run_name, best_strategy, best_method_val, best_acc_val = global_best
        info = get_run_info(args.runs_dir, run_name)
        version = get_model_version(info.exp_dir)
        versioned_dir = os.path.join(info.exp_dir, f"ep{version}") if version is not None else info.exp_dir
        ref_path = os.path.join(versioned_dir, "reference_embeddings_train.npz") if best_strategy == "train" else os.path.join(versioned_dir, f"reference_embeddings_{best_strategy}.npz")
        if not os.path.exists(info.test_split_path):
            print("\n" + "=" * 80)
            print("SELECCIÓN POR VAL → TEST")
            print("=" * 80)
            print(f"Selected run: {run_name} (ref={best_strategy}, {best_method_val}, val acc: {best_acc_val:.4f})")
            print("⚠ No se encontró splits/test_paths.json para este run. No se evaluó test.")
        elif not os.path.exists(ref_path):
            print("\n" + "=" * 80)
            print("SELECCIÓN POR VAL → TEST")
            print("=" * 80)
            print(f"Selected run: {run_name} (ref={best_strategy}, {best_method_val}, val acc: {best_acc_val:.4f})")
            print(f"⚠ No se encontró {ref_path}. No se evaluó test.")
        else:
            with open(info.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            n_points = int(config["n_points"])
            width = int(config["width"])
            test_paths = load_test_paths(info.test_split_path)
            test_set = build_val_set(dataset_index, test_paths)
            if len(test_set) == 0:
                print("\n" + "=" * 80)
                print("SELECCIÓN POR VAL → TEST")
                print("=" * 80)
                print(f"Selected run: {run_name} (ref={best_strategy}, {best_method_val}, val acc: {best_acc_val:.4f})")
                print("⚠ Test set vacío.")
            else:
                model = TripletNet(width=width).to(device)
                state_dict = torch.load(info.model_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                ref_data = np.load(ref_path)
                reference_embeddings = {k: ref_data[k] for k in ref_data.files}
                out_dir_test = os.path.join(versioned_dir, "evaluation_test", best_strategy) if args.export_csv else None
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
                )
                best_method_test = max(results_test, key=results_test.get)
                acc_test_selected = results_test[best_method_val]["accuracy"]
                acc_test_best = results_test[best_method_test]["accuracy"]
                print("\n" + "=" * 80)
                print("SELECCIÓN POR VAL → TEST")
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
