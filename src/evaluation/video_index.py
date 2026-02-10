"""
Índice video -> forma de captura (desde index_videos.csv).
Extracción del nombre de video desde el path del .ply.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Tuple


def load_video_index(csv_path: str) -> Dict[str, str]:
    """
    Carga index_videos.csv (columnas: video, forma).
    Devuelve dict: video_id (sin .mp4) -> forma de captura.
    """
    index: Dict[str, str] = {}
    if not os.path.exists(csv_path):
        return index
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row.get("video", "").strip()
            forma = row.get("forma", "").strip()
            if video:
                video_id = video.replace(".mp4", "")
                index[video_id] = forma
    return index


def parse_video_id_from_ply_path(path: str) -> str:
    """
    Extrae el identificador del video desde el nombre del archivo .ply.
    Patrón esperado: clase_VID_YYYYMMDD_HHMMSS_nube_N.ply
    Devuelve ej. "VID_20230322_173631" o "" si no matchea.
    """
    name = os.path.basename(path)
    if not name.endswith(".ply"):
        return ""
    name = name[:-4]  # quitar .ply
    if "_nube_" not in name:
        return ""
    part_before_nube = name.split("_nube_")[0]
    parts = part_before_nube.split("_", 1)  # clase y resto
    if len(parts) != 2:
        return ""
    return parts[1]  # VID_20230322_173631


def get_video_and_capture_form(
    path: str, video_index: Dict[str, str] | None
) -> Tuple[str, str]:
    """
    Para un path de .ply devuelve (video_id, forma_captura).
    Si video_index es None o no hay match, forma_captura es "".
    """
    video_id = parse_video_id_from_ply_path(path)
    if not video_id:
        return "", ""
    if video_index is None:
        return video_id, ""
    return video_id, video_index.get(video_id, "")
