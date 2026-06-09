"""
Analiza CSVs de predicciones de validación para dar insights sobre nubes mal identificadas.
Incluye rank_true: no solo error (pred != true) sino qué tan atrás queda la clase verdadera en el ranking.
Opcional: comparar las top N combinaciones modelo-estrategia-método para ver si los mismos
videos/capturas/nubes fallan en todas o cada una tiene errores específicos.

Requisito: haber corrido eval con --export_csv y index_videos.csv para columnas video y capture_form.
Uso:
  python -m scripts.analyze_validation_errors --run w8_np512_m0.5_lr3e-4_bs16_seed42 --strategy centroid_all --method "L1 Distance"
  python -m scripts.analyze_validation_errors --top_combos 5 --out reporte_top5.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def _find_ep_dirs(runs_dir: str, run_name: str) -> list[str]:
    """Devuelve carpetas ep<N> existentes en el run, ordenadas por N descendente."""
    run_dir = os.path.join(runs_dir, run_name)
    if not os.path.isdir(run_dir):
        return []
    ep_dirs = []
    for name in os.listdir(run_dir):
        m = re.match(r"^ep(\d+)$", name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            ep_dirs.append((int(m.group(1)), name))
    ep_dirs.sort(key=lambda x: x[0], reverse=True)
    return [os.path.join(run_dir, name) for _, name in ep_dirs]


def _find_one_csv(ep_dir: str, strategy: str, method: str | None) -> str | None:
    """Busca un CSV de validación en ep_dir/evaluation/<strategy>/ para la estrategia y opcionalmente el método."""
    eval_dir = os.path.join(ep_dir, "evaluation", strategy)
    if not os.path.isdir(eval_dir):
        return None
    prefix = "validation_predictions_"
    if method:
        safe = method.replace(" ", "_").replace("/", "_") + ".csv"
        fname = prefix + safe
        path = os.path.join(eval_dir, fname)
        if os.path.isfile(path):
            return path
        return None
    for f in sorted(os.listdir(eval_dir)):
        if f.startswith(prefix) and f.endswith(".csv"):
            return os.path.join(eval_dir, f)
    return None


RANK_SEVERE = 10  # rank_true > este umbral = "muy atrás en el ranking"
RANK_MUY_ALTO = 50  # sección de nubes con rank_true >= esto


def run_analysis(
    csv_path: str,
    top_n: int = 25,
    min_samples_video: int = 10,
    min_samples_capture: int = 20,
) -> str:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "correct" not in df.columns:
        return f"Error: el CSV no tiene columna 'correct'. Columnas: {list(df.columns)}"

    total = len(df)
    errors = df[df["correct"] == 0]
    n_errors = len(errors)
    acc = (total - n_errors) / total if total else 0
    has_rank = "rank_true" in df.columns

    lines = []
    lines.append("=" * 70)
    lines.append("ANALISIS DE NUBES MAL IDENTIFICADAS (VALIDACION)")
    lines.append("=" * 70)
    lines.append(f"CSV: {csv_path}")
    lines.append(f"Total muestras: {total}  |  Correctas: {total - n_errors}  |  Errores: {n_errors}  |  Accuracy: {acc:.4f}")
    if has_rank:
        lines.append(f"Rank_true: media (global)={df['rank_true'].mean():.1f}  mediana={df['rank_true'].median():.1f}")
        if n_errors:
            err_ranks = errors["rank_true"]
            lines.append(f"Rank_true (solo errores): media={err_ranks.mean():.1f}  mediana={err_ranks.median():.1f}  |  con rank_true > {RANK_SEVERE}: {(err_ranks > RANK_SEVERE).sum()}")
    lines.append("")

    has_video = "video" in df.columns
    has_capture = "capture_form" in df.columns

    # --- Nubes con rank_true muy alto (clase verdadera muy atrás) ---
    if has_rank:
        high_rank = df[df["rank_true"] >= RANK_MUY_ALTO].sort_values("rank_true", ascending=False)
        if len(high_rank) > 0:
            lines.append(f"--- NUBES CON RANK_TRUE >= {RANK_MUY_ALTO} (clase verdadera muy atrás en el ranking) ---")
            for _, row in high_rank.head(top_n).iterrows():
                short = os.path.basename(row["path"]) if len(str(row["path"])) > 60 else row["path"]
                lines.append(f"  rank_true={int(row['rank_true']):3d}  correct={int(row['correct'])}  {short}")
            if len(high_rank) > top_n:
                lines.append(f"  ... y {len(high_rank) - top_n} más.")
            lines.append("")

    # --- Nubes mal identificadas (error + orden por rank_true) ---
    lines.append("--- NUBES MAL IDENTIFICADAS (resumen) ---")
    if n_errors == 0:
        lines.append("No hay errores en este CSV.")
    else:
        err_df = errors.copy()
        if has_rank:
            err_df = err_df.sort_values("rank_true", ascending=False)
        paths = err_df["path"].tolist()
        if has_rank:
            ranks = err_df["rank_true"].tolist()
            lines.append(f"Primeras {min(top_n, len(paths))} nubes con error (ordenadas por rank_true descendente, peor primero):")
            for p, r in zip(paths[: top_n], ranks[: top_n]):
                short = os.path.basename(p) if len(p) > 60 else p
                lines.append(f"  rank_true={int(r):3d}  {short}")
        else:
            lines.append(f"Paths con error ({min(top_n, len(paths))} de {len(paths)}):")
            for p in paths[: top_n]:
                short = os.path.basename(p) if len(p) > 60 else p
                lines.append(f"  {short}")
    lines.append("")

    if not has_video and not has_capture:
        lines.append("No hay columnas 'video' ni 'capture_form' (correr eval con index_videos.csv).")
        return "\n".join(lines)

    # --- Por video (con rank_true si existe) ---
    if has_video:
        lines.append("--- VIDEOS PROBLEMATICOS ---")
        agg_dict = {
            "total": ("correct", "count"),
            "errors": ("correct", lambda s: (s == 0).sum()),
        }
        if has_rank:
            agg_dict["mean_rank_true"] = ("rank_true", "mean")
            agg_dict["rank_severe"] = ("rank_true", lambda s: (s > RANK_SEVERE).sum())
        by_video = df.groupby("video").agg(**{k: v for k, v in agg_dict.items()}).reset_index()
        by_video["error_rate"] = by_video["errors"] / by_video["total"].clip(lower=1)
        by_video = by_video[by_video["total"] >= min_samples_video]
        by_video = by_video.sort_values("error_rate", ascending=False)
        lines.append(f"(Videos con al menos {min_samples_video} muestras; ordenados por tasa de error)")
        for _, row in by_video.head(top_n).iterrows():
            msg = f"  video={row['video']}  total={int(row['total'])}  errors={int(row['errors'])}  error_rate={row['error_rate']:.4f}"
            if has_rank:
                msg += f"  mean_rank_true={row['mean_rank_true']:.1f}  rank_true>{RANK_SEVERE}={int(row['rank_severe'])}"
            lines.append(msg)
        lines.append("")

    # --- Por tipo de captura (con rank_true si existe) ---
    if has_capture:
        lines.append("--- TIPOS DE CAPTURA PROBLEMATICOS ---")
        agg_dict = {
            "total": ("correct", "count"),
            "errors": ("correct", lambda s: (s == 0).sum()),
        }
        if has_rank:
            agg_dict["mean_rank_true"] = ("rank_true", "mean")
            agg_dict["rank_severe"] = ("rank_true", lambda s: (s > RANK_SEVERE).sum())
        by_capture = df.groupby("capture_form").agg(**{k: v for k, v in agg_dict.items()}).reset_index()
        by_capture["error_rate"] = by_capture["errors"] / by_capture["total"].clip(lower=1)
        by_capture = by_capture[by_capture["total"] >= min_samples_capture]
        by_capture = by_capture.sort_values("error_rate", ascending=False)
        lines.append(f"(Tipos con al menos {min_samples_capture} muestras)")
        for _, row in by_capture.head(top_n).iterrows():
            msg = f"  capture_form={row['capture_form']}  total={int(row['total'])}  errors={int(row['errors'])}  error_rate={row['error_rate']:.4f}"
            if has_rank:
                msg += f"  mean_rank_true={row['mean_rank_true']:.1f}  rank_true>{RANK_SEVERE}={int(row['rank_severe'])}"
            lines.append(msg)
        lines.append("")

    # --- Por (video, capture_form) ---
    if has_video and has_capture:
        lines.append("--- COMBINACION VIDEO + TIPO CAPTURA (mas problematicos) ---")
        agg_dict = {
            "total": ("correct", "count"),
            "errors": ("correct", lambda s: (s == 0).sum()),
        }
        if has_rank:
            agg_dict["mean_rank_true"] = ("rank_true", "mean")
        by_both = df.groupby(["video", "capture_form"]).agg(**{k: v for k, v in agg_dict.items()}).reset_index()
        by_both["error_rate"] = by_both["errors"] / by_both["total"].clip(lower=1)
        by_both = by_both[by_both["total"] >= 5]
        by_both = by_both.sort_values("error_rate", ascending=False)
        for _, row in by_both.head(top_n).iterrows():
            msg = f"  video={row['video']}  capture_form={row['capture_form']}  total={int(row['total'])}  errors={int(row['errors'])}  error_rate={row['error_rate']:.4f}"
            if has_rank:
                msg += f"  mean_rank_true={row['mean_rank_true']:.1f}"
            lines.append(msg)
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def _get_top_combos(runs_dir: str, n: int) -> list[tuple[str, str, str, str, float]]:
    """Devuelve las top N (run_name, ep_dir, strategy, method, val_accuracy)."""
    combos = []
    for run_name in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        for ep_name in sorted(os.listdir(run_path)):
            m = re.match(r"^ep(\d+)$", ep_name)
            if not m:
                continue
            ep_dir = os.path.join(run_path, ep_name)
            report_path = os.path.join(ep_dir, "evaluation_report.json")
            if not os.path.isfile(report_path):
                continue
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            val = data.get("val") or {}
            for strategy, methods in val.items():
                if not isinstance(methods, dict):
                    continue
                for method, metrics in methods.items():
                    if isinstance(metrics, dict) and "accuracy" in metrics:
                        combos.append((run_name, ep_dir, strategy, method, float(metrics["accuracy"])))
    combos.sort(key=lambda x: x[4], reverse=True)
    return combos[:n]


def _csv_path_for_combo(ep_dir: str, strategy: str, method: str) -> str | None:
    """Ruta al CSV de validación para esa combinación."""
    return _find_one_csv(ep_dir, strategy, method)


def _collect_error_sets(csv_path: str):
    """Carga el CSV y devuelve (set path, set video, set (video, capture_form)) de muestras con error."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "correct" not in df.columns:
        return set(), set(), set()
    err = df[df["correct"] == 0]
    paths = set(err["path"].astype(str).tolist())
    videos = set()
    pairs = set()
    if "video" in err.columns:
        videos = set(err["video"].dropna().astype(str).tolist())
    if "video" in err.columns and "capture_form" in err.columns:
        for _, row in err.iterrows():
            v = str(row.get("video", ""))
            c = str(row.get("capture_form", ""))
            if v or c:
                pairs.add((v, c))
    return paths, videos, pairs


def run_top_combos_analysis(
    runs_dir: str,
    top_n_combos: int = 5,
    top_n: int = 25,
    min_samples_video: int = 10,
    min_samples_capture: int = 20,
) -> str:
    """Obtiene las top N combinaciones por val accuracy, analiza cada una y compara errores comunes vs específicos."""
    combos = _get_top_combos(runs_dir, top_n_combos)
    if not combos:
        return "No se encontraron evaluation_report.json en runs/*/ep*/ para extraer combinaciones."
    import pandas as pd

    lines = []
    lines.append("=" * 70)
    lines.append(f"TOP {top_n_combos} COMBINACIONES (run + estrategia + método) POR VAL ACCURACY")
    lines.append("=" * 70)
    for i, (run_name, ep_dir, strategy, method, acc) in enumerate(combos, 1):
        lines.append(f"  {i}. {run_name}  |  {strategy}  |  {method}  |  val_acc={acc:.4f}")
    lines.append("")

    # Cargar CSVs y conjuntos de errores
    combo_labels = []
    all_paths = []
    all_videos = []
    all_pairs = []
    has_csv = []
    for run_name, ep_dir, strategy, method, acc in combos:
        csv_path = _csv_path_for_combo(ep_dir, strategy, method)
        label = f"{run_name} | {strategy} | {method}"
        combo_labels.append(label)
        if not csv_path or not os.path.isfile(csv_path):
            all_paths.append(set())
            all_videos.append(set())
            all_pairs.append(set())
            has_csv.append(False)
            lines.append(f"[Falta CSV para {label}]")
            continue
        paths, videos, pairs = _collect_error_sets(csv_path)
        all_paths.append(paths)
        all_videos.append(videos)
        all_pairs.append(pairs)
        has_csv.append(True)

    valid_paths = [all_paths[i] for i in range(len(combo_labels)) if has_csv[i]]
    valid_videos = [all_videos[i] for i in range(len(combo_labels)) if has_csv[i]]
    valid_pairs = [all_pairs[i] for i in range(len(combo_labels)) if has_csv[i]]

    # Intersección y unión (solo entre combos con CSV)
    if valid_paths:
        paths_intersection = set.intersection(*valid_paths)
        paths_union = set.union(*valid_paths)
    else:
        paths_intersection = paths_union = set()
    if valid_videos:
        videos_intersection = set.intersection(*valid_videos)
        videos_union = set.union(*valid_videos)
    else:
        videos_intersection = videos_union = set()
    if valid_pairs:
        pairs_intersection = set.intersection(*valid_pairs)
        pairs_union = set.union(*valid_pairs)
    else:
        pairs_intersection = pairs_union = set()

    lines.append("--- ERRORES COMUNES (en las 5 combinaciones) ---")
    lines.append(f"  Nubes (paths) que fallan en todas: {len(paths_intersection)}")
    if paths_intersection and len(paths_intersection) <= top_n:
        for p in sorted(paths_intersection)[: top_n]:
            lines.append(f"    {os.path.basename(p) if len(p) > 60 else p}")
    elif paths_intersection:
        for p in sorted(paths_intersection)[: top_n]:
            lines.append(f"    {os.path.basename(p) if len(p) > 60 else p}")
        lines.append(f"    ... y {len(paths_intersection) - top_n} más.")
    lines.append(f"  Videos que aparecen en errores en todas: {len(videos_intersection)}")
    if videos_intersection:
        lines.append(f"    {sorted(videos_intersection)[:20]}")
    lines.append(f"  Pares (video, capture_form) en errores en todas: {len(pairs_intersection)}")
    if pairs_intersection and len(pairs_intersection) <= 30:
        lines.append(f"    {sorted(pairs_intersection)}")
    lines.append("")

    lines.append("--- ERRORES ESPECIFICOS POR COMBINACION ---")
    lines.append("(Nubes que fallan solo en esa combinación)")
    for i, (label, paths) in enumerate(zip(combo_labels, all_paths)):
        others = set()
        for j, p in enumerate(all_paths):
            if j != i:
                others |= p
        only_here = paths - others
        lines.append(f"  {label}")
        lines.append(f"    Solo aquí: {len(only_here)} nubes.")
        if only_here and len(only_here) <= 15:
            for p in sorted(only_here)[:15]:
                lines.append(f"      {os.path.basename(p) if len(p) > 55 else p}")
        elif only_here:
            for p in sorted(only_here)[:15]:
                lines.append(f"      {os.path.basename(p) if len(p) > 55 else p}")
            lines.append(f"      ... y {len(only_here) - 15} más.")
    lines.append("")

    lines.append("--- RESUMEN POR COMBINACION ---")
    for label, paths, videos, pairs in zip(combo_labels, all_paths, all_videos, all_pairs):
        lines.append(f"  {label}")
        lines.append(f"    Errores: {len(paths)} nubes, {len(videos)} videos distintos, {len(pairs)} pares (video,capture).")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analiza CSVs de predicciones de validacion: nubes, videos y tipos de captura problematicos; opcionalmente compara top N combinaciones."
    )
    parser.add_argument("--runs_dir", type=str, default="runs", help="Carpeta de runs.")
    parser.add_argument("--run", type=str, default=None, help="Nombre del run (obligatorio si no se usa --top_combos).")
    parser.add_argument("--strategy", type=str, default="centroid_all", help="Estrategia de referencia.")
    parser.add_argument("--method", type=str, default="L1 Distance", help='Metodo (ej. "L1 Distance").')
    parser.add_argument("--top_n", type=int, default=25, help="Cuantos items en cada lista de problematicos.")
    parser.add_argument("--min_samples_video", type=int, default=10)
    parser.add_argument("--min_samples_capture", type=int, default=20)
    parser.add_argument("--out", type=str, default=None, help="Guardar reporte en este archivo.")
    parser.add_argument("--top_combos", type=int, default=None, metavar="N", help="Analizar las top N combinaciones run+estrategia+metodo y comparar errores comunes vs especificos.")
    args = parser.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)

    if args.top_combos is not None:
        report = run_top_combos_analysis(
            runs_dir,
            top_n_combos=args.top_combos,
            top_n=args.top_n,
            min_samples_video=args.min_samples_video,
            min_samples_capture=args.min_samples_capture,
        )
        print(report)
        if args.out:
            with open(os.path.abspath(args.out), "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReporte guardado en {args.out}")
        return 0

    if not args.run:
        parser.error("Indica --run <nombre_run> o --top_combos N.")
    ep_dirs = _find_ep_dirs(runs_dir, args.run)
    if not ep_dirs:
        print(f"No se encontraron carpetas ep<N> en {runs_dir}/{args.run}")
        return 1

    csv_path = _find_one_csv(ep_dirs[0], args.strategy, args.method)
    if not csv_path:
        csv_path = _find_one_csv(ep_dirs[0], args.strategy, None)
    if not csv_path:
        print(f"No se encontro CSV para run={args.run}, strategy={args.strategy}, method={args.method}")
        return 1

    report = run_analysis(
        csv_path,
        top_n=args.top_n,
        min_samples_video=args.min_samples_video,
        min_samples_capture=args.min_samples_capture,
    )
    print(report)
    if args.out:
        with open(os.path.abspath(args.out), "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReporte guardado en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
