# Qué se guarda en cada run

Cada run es una carpeta dentro de `runs/` (por ejemplo `runs/w16_np512_m0.5_lr1e-3_bs16_seed42/`). Lo que se guarda se reparte en **raíz del run** (común a todo el entrenamiento) y **carpetas por versión** `ep<N>/` (asociadas a “modelo después de N épocas”), para no pisar evaluaciones anteriores si seguís entrenando.

---

## 1. Raíz del run (siempre; no depende de la versión)

| Archivo / carpeta | Cuándo | Contenido |
|-------------------|--------|-----------|
| **config.json** | Al crear el run | Hiperparámetros: n_points, width, lr, margin, epochs, batch_size, seed, val_size, test_size, ref_samples_per_class, early_stopping_patience, etc. |
| **training.log** | Cada época y mensajes | Log de texto: pérdidas, best model, checkpoints, NaN/retry, early stop, referencia al guardar ref embeddings. |
| **metrics.csv** | Cada época | Una fila por época: `epoch`, `train_loss`, `val_loss`, `lr`. |
| **model_best.pt** | Cuando val_loss mejora | Solo `state_dict` del modelo (pesos). Siempre el mejor hasta ahora. |
| **model_version.json** | Cuando se guarda model_best.pt | `{"epoch": N, "val_loss": ...}` — época del modelo guardado en model_best.pt. |
| **last_epoch.json** | Al terminar el entrenamiento | `{"epoch": N}` — última época completada (define la carpeta `ep<N>/` para la eval). |
| **checkpoint_last.pt** | Cada época | Checkpoint completo para reanudar: epoch, model_state_dict, optimizer, scheduler, best_val, lr, etc. |
| **splits/train_paths.json** | Al iniciar entrenamiento | Lista de paths de muestras de entrenamiento (70%). |
| **splits/val_paths.json** | Al iniciar entrenamiento | Lista de paths de validación (15%). |
| **splits/test_paths.json** | Al iniciar entrenamiento | Lista de paths de test (15%). |
| **reference_embeddings_train.npz** | Al terminar las épocas | Referencias por clase (centroide 5 muestras de train). Lo escribe el **trainer**; la eval puede copiarlo a `ep<N>/` para conservarlo. |
| **reference_paths_train.json** | Al terminar las épocas | Qué paths se usaron para ese centroide: clase → lista de paths (5 por clase). |

---

## 2. Carpeta por versión `ep<N>/` (todo lo asociado a “modelo después de N épocas”)

La eval usa la época actual del run (`last_epoch.json` o `model_version.json`) y guarda **toda** la evaluación dentro de `runs/<run_name>/ep<N>/`. Así, si entrenás 200 épocas, evaluás (se crea `ep200/`), luego entrenás 100 más y volvés a evaluar, se crea `ep300/` y no se pisa lo de 200.

| Archivo / carpeta | Contenido |
|-------------------|-----------|
| **model.pt** | Copia del modelo (state_dict) usada en esta evaluación. Si seguís entrenando, `model_best.pt` en la raíz se actualiza; con `ep<N>/model.pt` podés recuperar la versión de esa época. |
| **reference_embeddings_train.npz** | Copia del de la raíz la primera vez que evaluás esta versión (para no perderla si después reentrenás). |
| **reference_embeddings_centroid_5.npz**, **centroid_10**, … | Estrategias extra que genera la eval (si `--ref_strategy all`). |
| **evaluation_report.json** | Resumen: accuracy por estrategia y método; `best_val` (estrategia, método, accuracy). |
| **evaluation/** | Solo con `--export_csv`: subcarpetas por estrategia con CSVs de predicciones por método. |
| **evaluation_test/** | Solo con `--export_csv` y si evaluás test: igual que `evaluation/` para el split de test. |

**Cómo recuperar el modelo de una época:** cargar `runs/<run_name>/ep<N>/model.pt` como state_dict (igual que `model_best.pt`). Ejemplo: `state_dict = torch.load("runs/mi_run/ep200/model.pt", map_location="cpu", weights_only=True)` y luego `model.load_state_dict(state_dict)`.

**Runs viejos (sin `last_epoch.json` ni `model_version.json`):** la eval escribe todo en la **raíz** del run, como antes.

---

## 3. Resumen visual

```
runs/<run_name>/
├── config.json
├── training.log
├── metrics.csv
├── model_best.pt
├── model_version.json        ← época del best model
├── last_epoch.json           ← última época completada
├── checkpoint_last.pt
├── reference_embeddings_train.npz   ← trainer (se puede copiar a ep<N>/)
├── reference_paths_train.json
├── splits/
│   ├── train_paths.json
│   ├── val_paths.json
│   └── test_paths.json
├── ep200/                     ← evaluación “después de 200 épocas”
│   ├── model.pt               ← snapshot del modelo (recuperable si seguís entrenando)
│   ├── reference_embeddings_train.npz
│   ├── reference_embeddings_centroid_5.npz
│   ├── ... (resto de estrategias)
│   ├── evaluation_report.json
│   ├── evaluation/
│   └── evaluation_test/
└── ep300/                     ← si entrenás 100 más y volvés a evaluar
    ├── reference_embeddings_train.npz
    ├── ...
    ├── evaluation_report.json
    └── ...
```

---

## 4. Qué no se guarda

- **Salida de consola** de la eval: no se persiste por defecto (podés redirigir con `> reporte.txt`).
- **Predicciones detalladas** (por muestra): solo si usás `--export_csv`.
