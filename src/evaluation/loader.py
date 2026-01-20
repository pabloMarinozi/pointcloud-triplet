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
    )


def load_val_paths(val_split_path: str) -> List[str]:
    with open(val_split_path, "r", encoding="utf-8") as f:
        paths = json.load(f)

    # normalizamos a abs para matchear index_dataset_by_path
    return [os.path.abspath(p) for p in paths]


def build_val_set(dataset_index: Dict[str, str], val_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Devuelve lista de (true_label, path)
    Solo incluye paths encontrados en el index.
    """
    out: List[Tuple[str, str]] = []
    for p in val_paths:
        p_abs = os.path.abspath(p)
        if p_abs in dataset_index:
            out.append((dataset_index[p_abs], p_abs))
    return out
