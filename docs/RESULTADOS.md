# Resultados del experimento – Pointcloud Triplet

Documento único de resultados. Consolida y reemplaza a los tres reportes
previos (`REPORTE_EXPERIMENTO.md`, `REPORTE_RESULTADOS_Y_SUGERENCIAS.md`,
`INFORME_EXPERIMENTOS_AVANCE.md`). Ante diferencias entre ellos, se priorizó la
información más reciente; las discrepancias sin resolver están listadas en la
sección 7.

---

## 1. Diseño del experimento

- **Objetivo:** metric learning sobre nubes de puntos 3D (TripletNet / PointNet-style)
  para aprender embeddings que preserven identidad; clasificación por similitud a
  prototipos de referencia.
- **Caso de uso:** identidad por geometría en racimos de uva (nubes densas).
- **Dataset:** 57.180 nubes de puntos, 100 clases (una por carpeta contenedora del `.ply`).
- **Split:** 70% train / 15% val / 15% test (≈40.026 / 8.577 / 8.577).
- **Hiperparámetros fijos:** `n_points=512`, `margin=0.5`, `batch_size=16`, `seed=42`.
- **Entrenamiento:** 6 configuraciones (width × lr), 200 épocas fijas, sin early stopping,
  vía `scripts/run_6models_70_15_15.py`.

| Run | width | lr | Parámetros (aprox.) |
|-----|-------|------|---------------------|
| w8_np512_m0.5_lr1e-3_bs16_seed42  | 8  | 1e-3 | 69.161 |
| w8_np512_m0.5_lr3e-4_bs16_seed42  | 8  | 3e-4 | 69.161 |
| w16_np512_m0.5_lr1e-3_bs16_seed42 | 16 | 1e-3 | 278.857 |
| w16_np512_m0.5_lr3e-4_bs16_seed42 | 16 | 3e-4 | 278.857 |
| w32_np512_m0.5_lr1e-3_bs16_seed42 | 32 | 1e-3 | ~1.1M |
| w32_np512_m0.5_lr3e-4_bs16_seed42 | 32 | 3e-4 | ~1.1M |

---

## 2. Estado de los runs

Los 6 runs corrieron a 200 épocas. **`w8_lr3e-4` se extendió luego a 400 épocas**
(reanudado con `--resume`); su mejor checkpoint es la **época 390**.

| Run | Épocas completadas | Época del best | Best val_loss | Evaluación guardada |
|-----|--------------------|----------------|---------------|---------------------|
| w8_np512_m0.5_lr3e-4_bs16_seed42  | **400** | **390** | **0.1137** | ep200 (*) |
| w16_np512_m0.5_lr1e-3_bs16_seed42 | 200 | 181 | 0.1579 | ep200 |
| w16_np512_m0.5_lr3e-4_bs16_seed42 | 200 | 197 | 0.1692 | ep200 |
| w32_np512_m0.5_lr3e-4_bs16_seed42 | 200 | 193 | 0.1735 | ep200 |
| w32_np512_m0.5_lr1e-3_bs16_seed42 | 200 | 189 | 0.1754 | ep200 |
| w8_np512_m0.5_lr1e-3_bs16_seed42  | 200 | ~120 | 0.1812 | ep200 |

(*) **Importante:** la evaluación guardada de `w8_lr3e-4` corresponde al modelo de
**200 épocas** (`ep200/`), no al best actual de 400. La evaluación de `ep400` aún
**no se corrió**; ver sección 5.

**Conclusiones de entrenamiento:**

- **Mejor val_loss global:** `w8_lr3e-4`, época 390 → **0.1137** (modelo de 400 épocas).
  A 200 épocas su mejor val_loss era 0.1321.
- Los modelos anchos (`w32`) tienen train_loss muy bajo (0.10–0.13) pero val_loss más
  alto (~0.20) → **overfitting** claro.
- `w8_lr1e-3` no mejora con más entrenamiento (su mejor checkpoint es temprano).

---

## 3. Resultados de evaluación (val / test, modelo a ep200)

Reportes en `runs/<run>/ep200/evaluation_report.json`. Para cada run se toma la
**mejor combinación estrategia + método en val** y se reportan las métricas en val y test.

Métricas: `acc` = accuracy top-1; `top5`/`top10` = clase verdadera en top-5/top-10;
`MRR` = Mean Reciprocal Rank; `mean_rank` = rango promedio (1-based) de la clase verdadera.

| Run | Estrategia | Método | Val acc | Val top5 | Val top10 | Val MRR | Val mean_rank | Test acc | Test top5 | Test top10 | Test MRR | Test mean_rank |
|-----|------------|--------|---------|----------|-----------|---------|---------------|----------|-----------|------------|----------|----------------|
| **w8_lr3e-4** | centroid_all | L1 Distance | **19.2%** | **56.3%** | **77.4%** | **0.363** | **7.4** | **19.2%** | **55.6%** | **77.2%** | **0.361** | **7.5** |
| w32_lr3e-4 | multiprototype_k5 | L1 Distance | 18.2% | 43.0% | 61.5% | 0.317 | 9.9 | 17.9% | 43.8% | 61.8% | 0.317 | 9.8 |
| w16_lr1e-3 | centroid_all | Cosine Similarity | 12.2% | 39.8% | 61.9% | 0.265 | 10.4 | 11.9% | 39.1% | 60.7% | 0.263 | 10.5 |
| w16_lr3e-4 | centroid_all | L2 Distance | 11.8% | 39.5% | 61.5% | 0.265 | 10.4 | 12.0% | 39.3% | 61.0% | 0.265 | 10.5 |
| w32_lr1e-3 | multiprototype_k5 | L1 Distance | 12.6% | 37.2% | 55.4% | 0.260 | 11.7 | 12.5% | 36.9% | 55.2% | 0.258 | 11.7 |
| w8_lr1e-3 | centroid_10 | L1 Distance | 11.0% | 36.7% | 57.8% | 0.248 | 11.7 | 10.6% | 35.7% | 56.5% | 0.243 | 12.1 |

**Conclusiones de evaluación:**

- **Mejor run:** `w8_lr3e-4` (centroid_all + L1): ~19.2% acc en val y test, top5 ~56%,
  MRR ~0.36, mean_rank ~7.4.
- Val y test son muy parecidos en todos los runs → poco overfitting a nivel de ranking.
- `L1 Distance` (y en menor medida L2/Cosine) son los mejores métodos; `Dot Product` va mal.
- `centroid_all` y `multiprototype_k5` superan a `centroid_5/10/20`.

---

## 4. Sugerencias

**Modelos a seguir entrenando (por prioridad):**

1. **`w8_lr3e-4`** — mejor val_loss (0.1137 @ ep390) y mejor accuracy. Sigue mejorando con
   más épocas; probar lr decay (cosine annealing / reduce-on-plateau).
2. **`w16_lr1e-3`** — segundo mejor val_loss (0.1579), más capacidad que w8. Buen candidato
   para más épocas.
3. **`w16_lr3e-4`** — val_loss 0.1692, con margen de mejora; lr 3e-4 suele generalizar bien.

**Baja prioridad:** los `w32` (overfitting: train ~0.11–0.13 vs val ~0.20) y `w8_lr1e-3`
(no mejora con más entrenamiento; candidato a early stopping con patience≈20).

**Otros experimentos:** regularización (dropout / weight decay) para w32; ensemble de
`w8_lr3e-4` + `w16_lr1e-3`; data augmentation en los reference embeddings
(`ref_use_augmentation`).

---

## 5. Próximos pasos concretos

1. **Evaluar el modelo actual de `w8_lr3e-4` (400 épocas).** Genera `ep400/` y permite
   compararlo con `ep200`:
   ```bash
   python -m src.eval --data_dir <ruta_a_ply> --runs_dir runs \
       --run w8_np512_m0.5_lr3e-4_bs16_seed42 --export_csv
   ```
2. **Re-evaluar todos y guardar la salida** (hoy las métricas completas solo se vieron
   en consola para algunos runs):
   ```bash
   python -m src.eval --data_dir <ruta_a_ply> --run all > reporte_eval.txt 2>&1
   ```
3. **Protocolo "elegir por val, reportar test"** (selecciona run + estrategia + método por
   val y reporta solo el test del ganador):
   ```bash
   python -m src.eval --data_dir <ruta_a_ply> --run all --split select_and_test
   ```
4. **Análisis de errores** por run/estrategia/método con
   `scripts/analyze_validation_errors.py` (usa los CSVs de `--export_csv`).

---

## 6. Dónde están los resultados

- Métricas por época: `runs/<run>/metrics.csv`.
- Best checkpoint y época: `runs/<run>/model_best.pt`, `model_version.json`, `last_epoch.json`.
- Evaluación: `runs/<run>/ep<N>/evaluation_report.json` (todas las estrategias y métodos).
- CSVs de predicciones (si se usó `--export_csv`): `runs/<run>/ep<N>/evaluation/` y `evaluation_test/`.
- Qué se guarda exactamente en cada run: ver `docs/CONTENIDO_POR_RUN.md`.

---

## 7. Discrepancias detectadas entre los reportes viejos (a verificar)

Al consolidar quedaron diferencias que conviene confirmar contra los `runs/`:

- **Best val_loss de `w8_lr1e-3`:** el reporte viejo de resultados decía 0.1812
  (best en ~ep120), pero el informe de avance lo reportaba como 0.2065 (ep200).
  Verificar en `runs/w8_np512_m0.5_lr1e-3_bs16_seed42/metrics.csv` / `model_version.json`.
- **Mención a "época 291"** para `w8_lr1e-3` en el reporte viejo: imposible en un run de
  200 épocas; probablemente un error de tipeo. Descartado en este doc.
- **top5 de `w8_lr3e-4` en val:** 56.3% (reporte de resultados) vs 52.7% (informe de avance).
  Se tomó 56.3% por venir de la tabla val/test completa; reconfirmar con `evaluation_report.json`.
