from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.data.io import find_ply_files


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    exp_dir: str
    config_path: str
    model_path: str
    ref_emb_path: str
    val_split_path: str
    test_split_path: str


def index_dataset_by_path(data_dir: str) -> Dict[str, str]:
    """
    Devuelve un mapping: path_absoluto -> clase
    Convención: clase = nombre de carpeta contenedora del .ply
    """
    ply_files = find_ply_files(data_dir)
    mapping: Dict[str, str] = {}
    for p in ply_files:
        cls = os.path.basename(os.path.dirname(p))
        mapping[os.path.abspath(p)] = cls
    return mapping


def list_runs(runs_dir: str) -> List[str]:
    if not os.path.isdir(runs_dir):
        return []
    runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    runs.sort()
    return runs


def resolve_run(runs_dir: str, run: str) -> List[str]:
    """
    run:
      - "latest" -> último
      - "all"    -> todos
      - "<name>" -> uno específico
    """
    runs = list_runs(runs_dir)
    if not runs:
        return []

    if run == "latest":
        return [runs[-1]]
    if run == "all":
        return runs
    if run in runs:
        return [run]

    # fallback: si el usuario pasa un path
    if os.path.isdir(run):
        return [os.path.basename(run)]

    raise ValueError(f"Run '{run}' no encontrado dentro de {runs_dir}. Disponibles: {runs[:5]} ...")


def get_run_info(runs_dir: str, run_name: str) -> RunInfo:
    exp_dir = os.path.join(runs_dir, run_name)
    return RunInfo(
        run_name=run_name,
        exp_dir=exp_dir,
        config_path=os.path.join(exp_dir, "config.json"),
        model_path=os.path.join(exp_dir, "model_best.pt"),
        ref_emb_path=os.path.join(exp_dir, "reference_embeddings_train.npz"),
        val_split_path=os.path.join(exp_dir, "splits", "val_paths.json"),
        test_split_path=os.path.join(exp_dir, "splits", "test_paths.json"),
    )


def get_model_version(exp_dir: str) -> int | None:
    """
    Devuelve la época asociada a la versión actual del modelo para carpetas ep<N>.
    Orden: last_epoch.json (última época completada) > model_version.json (época del best) >
    checkpoint_last.pt > última fila de metrics.csv.
    """
    # 1) last_epoch.json (trainer lo escribe al terminar el entrenamiento)
    last_epoch_path = os.path.join(exp_dir, "last_epoch.json")
    if os.path.exists(last_epoch_path):
        try:
            with open(last_epoch_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data["epoch"])
        except (json.JSONDecodeError, KeyError):
            pass
    # 2) model_version.json (trainer lo escribe al guardar model_best.pt)
    model_version_path = os.path.join(exp_dir, "model_version.json")
    if os.path.exists(model_version_path):
        try:
            with open(model_version_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data["epoch"])
        except (json.JSONDecodeError, KeyError):
            pass
    # 3) checkpoint_last.pt
    try:
        import torch
    except ImportError:
        torch = None
        ckpt_path = os.path.join(exp_dir, "checkpoint_last.pt")
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict) and "epoch" in ckpt:
                    return int(ckpt["epoch"])
            except Exception:
                pass
    # 4) última fila de metrics.csv
    csv_path = os.path.join(exp_dir, "metrics.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) >= 2:
                last_row = lines[-1].strip().split(",")
                if last_row:
                    return int(float(last_row[0]))
        except (ValueError, IndexError):
            pass
    return None


def list_ref_strategies(exp_dir: str, ref_emb_path: str) -> List[Tuple[str, str]]:
    """
    Lista (nombre_estrategia, path) de reference embeddings en exp_dir.
    Incluye "train" (reference_embeddings_train.npz) primero si existe,
    luego reference_embeddings_<strategy>.npz con strategy != "train".
    """
    strategies: List[Tuple[str, str]] = []
    if os.path.exists(ref_emb_path):
        strategies.append(("train", ref_emb_path))
    if os.path.isdir(exp_dir):
        for fname in sorted(os.listdir(exp_dir)):
            if fname.startswith("reference_embeddings_") and fname.endswith(".npz"):
                name = fname.replace("reference_embeddings_", "").replace(".npz", "")
                if name != "train":
                    strategies.append((name, os.path.join(exp_dir, fname)))
    return strategies


def get_train_split_path(exp_dir: str) -> str:
    return os.path.join(exp_dir, "splits", "train_paths.json")


def load_train_paths(train_split_path: str) -> List[str]:
    with open(train_split_path, "r", encoding="utf-8") as f:
        paths = json.load(f)
    return list(paths)


def build_train_set(dataset_index: Dict[str, str], train_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Devuelve lista de (clase, path_actual) para paths de train que existen en dataset_index.
    Matchea por path absoluto o por (carpeta, nombre_archivo) si cambió el drive.
    """
    by_key: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for full_path, label in dataset_index.items():
        by_key[_path_key(full_path)] = (full_path, label)

    out: List[Tuple[str, str]] = []
    for p in train_paths:
        p_abs = os.path.abspath(p)
        if p_abs in dataset_index:
            out.append((dataset_index[p_abs], p_abs))
            continue
        key = _path_key(p)
        if key in by_key:
            actual_path, label = by_key[key]
            out.append((label, actual_path))
    return out


def load_val_paths(val_split_path: str) -> List[str]:
    with open(val_split_path, "r", encoding="utf-8") as f:
        paths = json.load(f)
    return [os.path.abspath(p) for p in paths]


def load_test_paths(test_split_path: str) -> List[str]:
    with open(test_split_path, "r", encoding="utf-8") as f:
        paths = json.load(f)
    return [os.path.abspath(p) for p in paths]


def _path_key(path: str) -> Tuple[str, str]:
    """(carpeta, nombre_archivo) para matchear aunque cambie unidad/ruta base."""
    return (os.path.basename(os.path.dirname(path)), os.path.basename(path))


def build_val_set(dataset_index: Dict[str, str], val_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Devuelve lista de (true_label, path).
    Primero matchea por path absoluto; si el run tiene paths de otra unidad (p. ej. J:)
    y ahora evaluás con data_dir en D:, matchea por (carpeta, nombre_archivo).
    """
    # Índice por (carpeta, filename) -> (path_actual, label) para matcheo sin depender del drive
    by_key: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for full_path, label in dataset_index.items():
        by_key[_path_key(full_path)] = (full_path, label)

    out: List[Tuple[str, str]] = []
    for p in val_paths:
        p_abs = os.path.abspath(p)
        if p_abs in dataset_index:
            out.append((dataset_index[p_abs], p_abs))
            continue
        key = _path_key(p)
        if key in by_key:
            actual_path, label = by_key[key]
            out.append((label, actual_path))
    return out
