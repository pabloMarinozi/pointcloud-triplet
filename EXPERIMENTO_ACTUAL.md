# Experimento actual – dónde está todo

Este documento resume qué hace el experimento y en qué archivos está implementado.

---

## Flujo del experimento

1. **Entrenar** 6 modelos (width 8/16/32 × lr 1e-3 y 3e-4), 200 épocas fijas, split 70/15/15.
2. **Evaluar** en val y test, con todas las estrategias de reference embeddings, métricas de ranking y opción de exportar CSV (con video, forma de captura, rank_true).
3. Opcional: **seleccionar** el mejor run por val y reportar solo su test (`--split select_and_test`).

---

## Dónde está implementado

### Entrenamiento
| Qué | Dónde |
|-----|--------|
| Entrada CLI, pipeline, splits 70/15/15, checkpoints, ref embeddings del trainer | `src/train.py` → `src/pipeline/trainer.py` |
| Script que lanza los 6 modelos (200 ep, sin early stop) | `scripts/run_6models_70_15_15.py` |
| model_version.json, last_epoch.json (para carpetas ep<N>) | `src/pipeline/trainer.py` |

### Evaluación
| Qué | Dónde |
|-----|--------|
| CLI eval, runs, split val/test/both/select_and_test, ref_strategy all/train | `src/eval.py` |
| Carpetas ep<N>, copia de modelo y reporte por versión | `src/eval.py` (usa `get_model_version` del loader) |
| Lista de estrategias de ref, get_model_version | `src/evaluation/loader.py` |
| Generación de estrategias (centroid_5, …, all_train) si faltan | `src/evaluation/ref_strategies.py` |
| evaluate_run_on_val, métricas de ranking, CSV con path/true/pred/score/rank_true/video/capture_form | `src/evaluation/report.py` |
| rank_of_label, predict_class, métodos (Cosine, L2, …) | `src/evaluation/metrics.py` |
| Carga index_videos.csv, extracción de video y forma de captura desde path .ply | `src/evaluation/video_index.py` |
| Embedding de una nube | `src/evaluation/embed.py` |

### Utilidades
| Qué | Dónde |
|-----|--------|
| Migración de runs antiguos a estructura ep<N>/ | `scripts/migrate_runs_to_ep_folders.py` (ya ejecutado; útil si clonás el repo en otro lado y tenés runs viejos) |

### Documentación
| Qué | Dónde |
|-----|--------|
| Qué se guarda en cada run y en ep<N>/ | `CONTENIDO_POR_RUN.md` |
| Reporte de entrenamiento (val_loss, etc.) | `REPORTE_EXPERIMENTO.md` |

### Datos del experimento
| Qué | Dónde |
|-----|--------|
| Video → forma de captura (para columnas en CSV de predicciones) | `index_videos.csv` (raíz del repo) |

---

## Scripts que se eliminaron (obsoletos)

- **run_best_200ep_early_stop.py** – Un solo run con early stopping; el experimento usa 6 runs a 200 ep fijas.
- **run_sweep.py** – Sweep genérico (val 20%, etc.), no es el diseño 70/15/15 ni los 6 modelos fijos.
- **grid_search.py** – Grid manual (width × n_points, 30 ep), reemplazado por run_6models_70_15_15.
- **compute_ref_and_eval.py** – Ref strategies + eval por run; todo eso está en `src.eval` (estrategias, ep<N>, métricas, CSV).

---

## Comandos típicos

```bash
# Entrenar los 6 modelos
python -m scripts.run_6models_70_15_15 --data_dir D:\ruta\a\ply

# Evaluar todos los runs (val + test, todas las estrategias, reporte en ep<N>/)
python -m src.eval --data_dir D:\ruta\a\ply --run all

# Evaluar y exportar CSV (con video, capture_form, rank_true)
python -m src.eval --data_dir D:\ruta\a\ply --run all --export_csv

# Seleccionar mejor por val y reportar solo su test
python -m src.eval --data_dir D:\ruta\a\ply --run all --split select_and_test
```
