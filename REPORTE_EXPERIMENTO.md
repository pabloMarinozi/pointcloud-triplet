# Reporte del experimento: Pointcloud Triplet (clasificación por embeddings)

## 1. Diseño del experimento

- **Dataset:** 57 180 nubes de puntos (100 clases). Split **70% train / 15% val / 15% test** (≈40 026 train, 8 577 val, 8 577 test).
- **Modelo:** TripletNet (PointNet-based), embedding de dimensión `width`. Clasificación por comparación con reference embeddings (varias estrategias y métricas).
- **Entrenamiento:** 6 configuraciones, 200 épocas fijas, sin early stopping.
  - **Width:** 8, 16, 32  
  - **Learning rate:** 1e-3, 3e-4  
  - **Fijo:** n_points=512, margin=0.5, batch_size=16, seed=42.

| Run | width | lr   | Parámetros (aprox.) |
|-----|-------|------|----------------------|
| w8_np512_m0.5_lr1e-3_bs16_seed42  | 8  | 1e-3 | 69 161 |
| w8_np512_m0.5_lr3e-4_bs16_seed42  | 8  | 3e-4 | 69 161 |
| w16_np512_m0.5_lr1e-3_bs16_seed42 | 16 | 1e-3 | —     |
| w16_np512_m0.5_lr3e-4_bs16_seed42 | 16 | 3e-4 | —     |
| w32_np512_m0.5_lr1e-3_bs16_seed42 | 32 | 1e-3 | —     |
| w32_np512_m0.5_lr3e-4_bs16_seed42 | 32 | 3e-4 | —     |

---

## 2. Resultados de entrenamiento

Métricas al final del entrenamiento (epoch 200) y mejor val_loss guardado (model_best.pt).

| Run | train_loss (ep 200) | val_loss (ep 200) | Best val_loss (checkpoint) |
|-----|----------------------|-------------------|----------------------------|
| w8_np512_m0.5_lr1e-3_bs16_seed42  | 0.1795 | 0.2065 | **0.1812** |
| w8_np512_m0.5_lr3e-4_bs16_seed42  | 0.1826 | 0.1334 | **0.1321** |
| w16_np512_m0.5_lr1e-3_bs16_seed42 | 0.1956 | 0.1638 | **0.1579** |
| w16_np512_m0.5_lr3e-4_bs16_seed42 | 0.1643 | 0.1775 | **0.1692** |
| w32_np512_m0.5_lr1e-3_bs16_seed42 | 0.1344 | 0.2044 | **0.1754** |
| w32_np512_m0.5_lr3e-4_bs16_seed42 | 0.1073 | 0.1951 | **0.1735** |

- **Mejor val_loss global:** **w16_np512_m0.5_lr1e-3_bs16_seed42** con val_loss = **0.1579**.
- w8 con lr 3e-4 también queda muy bajo (0.1321); w16 lr1e-3 es el mejor entre los checkpoints guardados.
- Los modelos más anchos (w32) tienen train_loss más bajo pero val_loss más alto, indicando posible overfitting.

---

## 3. Evaluación (val / test)

La evaluación se ejecutó por consola. Las **accuracy por estrategia de referencia** (train, centroid_5, centroid_10, centroid_20, centroid_all, centroid_l2norm_5, multiprototype_k5, all_train) y por **métrica** (Cosine, L2, etc.) **no están guardadas** en el repo; solo se vieron en la terminal.

Para obtener un reporte completo de evaluación y guardarlo:

1. **Re-ejecutar evaluación y guardar salida en un archivo:**
   ```bash
   python -m src.eval --data_dir D:\ruta\a\tus\ply --run all > reporte_eval.txt 2>&1
   ```
   (reemplazá `D:\ruta\a\tus\ply` por tu `--data_dir`.)

2. **Exportar CSVs por run (predicciones por método):**
   ```bash
   python -m src.eval --data_dir D:\ruta\a\tus\ply --run all --export_csv
   ```
   Se crean carpetas `evaluation/` y `evaluation_test/` dentro de cada run, con CSVs por estrategia.

3. **Protocolo “elegir por val, reportar test”:**
   ```bash
   python -m src.eval --data_dir D:\ruta\a\tus\ply --run all --split select_and_test
   ```
   Evalúa todos en val, elige el mejor (run + estrategia + método) y reporta solo la accuracy en test de ese modelo.

---

## 4. Resumen

- **Entrenamiento:** 6 runs completados a 200 épocas. Mejor val_loss: **w16, lr 1e-3** (0.1579).
- **Evaluación:** Hecha en consola; no hay log de accuracy en el repo. Usar los comandos de la sección 3 para re-evaluar y guardar reporte o CSVs.
- **Recomendación:** Correr `--split select_and_test` y guardar la salida para tener el modelo seleccionado por val y su accuracy final en test en un solo reporte.
