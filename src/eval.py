from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.evaluation.loader import (
    index_dataset_by_path,
    resolve_run,
    get_run_info,
    load_val_paths,
    build_val_set,
)
from src.evaluation.metrics import default_methods
from src.evaluation.report import evaluate_run_on_val, summarize_errors_by_class
from src.models.triplet import TripletNet


def parse_args():
    p = argparse.ArgumentParser("Evaluate trained TripletNet runs on validation split.")
    p.add_argument("--data_dir", type=str, required=True, help="Directorio raíz con .ply (recursivo).")
    p.add_argument("--runs_dir", type=str, default="runs", help="Carpeta de experimentos.")
    p.add_argument("--run", type=str, default="latest", help="latest | all | <run_name>")
    p.add_argument("--export_csv", action="store_true", help="Exporta predicciones CSV dentro de cada run.")
    p.add_argument("--use_augmentation", action="store_true", help="Augmentation al embeder (prueba robustez).")
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

    global_best = (None, None, -1.0)  # (run, method, acc)

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

        # Cargar split val
        if not os.path.exists(info.val_split_path):
            print("⚠ No se encontró splits/val_paths.json. Saltando.")
            continue

        val_paths = load_val_paths(info.val_split_path)
        val_set = build_val_set(dataset_index, val_paths)
        print(f"Validation set reconstruido: {len(val_set)} samples")

        if len(val_set) == 0:
            print("⚠ Validation vacío. Saltando.")
            continue

        # Cargar modelo
        if not os.path.exists(info.model_path):
            print("⚠ No se encontró model_best.pt. Saltando.")
            continue

        model = TripletNet(width=width).to(device)
        state_dict = torch.load(info.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Cargar reference embeddings
        if not os.path.exists(info.ref_emb_path):
            print("⚠ No se encontró reference_embeddings_train.npz. Saltando.")
            continue

        ref_data = np.load(info.ref_emb_path)
        reference_embeddings = {k: ref_data[k] for k in ref_data.files}
        print(f"Reference embeddings: {len(reference_embeddings)} clases")

        # Evaluar
        out_dir = os.path.join(info.exp_dir, "evaluation") if args.export_csv else None
        results = evaluate_run_on_val(
            model=model,
            reference_embeddings=reference_embeddings,
            val_set=val_set,
            methods=methods,
            n_points=n_points,
            device=device,
            use_augmentation=args.use_augmentation,
            export_csv=args.export_csv,
            out_dir=out_dir,
        )

        # Resumen por run
        best_method = max(results, key=results.get)
        best_acc = results[best_method]
        for method_name, acc in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {method_name:<22}  acc={acc:.4f}")
        print(f"BEST: {best_method}  acc={best_acc:.4f}")

        if best_acc > global_best[2]:
            global_best = (run_name, best_method, best_acc)

        # Si exportaste CSV, podés calcular error por clase
        if args.export_csv and out_dir is not None:
            df_err = summarize_errors_by_class(out_dir)
            # Mostramos top 10 errores por tasa de error, por ejemplo
            df_err_sorted = df_err.sort_values(["error_rate"], ascending=False).head(10)
            print("\nTop-10 (metric,class) por tasa de error:")
            print(df_err_sorted.to_string(index=False))

    # Global best
    print("\n" + "=" * 80)
    print("MEJOR RUN GLOBAL")
    print("=" * 80)
    if global_best[0] is None:
        print("No se evaluó ningún run correctamente.")
    else:
        run_name, method_name, acc = global_best
        print(f"{run_name}  |  {method_name}  |  acc={acc:.4f}")


if __name__ == "__main__":
    main()
