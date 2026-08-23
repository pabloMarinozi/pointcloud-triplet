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

Los 6 runs corrieron a 200 épocas. Dos se extendieron luego con `--resume`:
**`w8_lr3e-4` se llevó hasta 1000 épocas** y **`w16_lr3e-4` hasta 400**.

| Run | Épocas completadas | Best val_loss | Evaluación guardada |
|-----|--------------------|---------------|---------------------|
| w8_np512_m0.5_lr3e-4_bs16_seed42  | **1000** | **0.0701** | ep200 / ep400 / ep1000 |
| w16_np512_m0.5_lr1e-3_bs16_seed42 | 200 | 0.1579 | ep200 |
| w16_np512_m0.5_lr3e-4_bs16_seed42 | 400 | 0.1688 | ep200 |
| w32_np512_m0.5_lr3e-4_bs16_seed42 | 200 | 0.1735 | ep200 |
| w32_np512_m0.5_lr1e-3_bs16_seed42 | 200 | 0.1754 | ep200 |
| w8_np512_m0.5_lr1e-3_bs16_seed42  | 200 | 0.1812 | ep200 |

**Conclusiones de entrenamiento:**

- **Mejor modelo global:** `w8_lr3e-4` entrenado a **1000 épocas** → best val_loss **0.0701**.
  Sigue mejorando con más entrenamiento, sin señales de overfitting (val_loss baja de
  forma sostenida: ~0.13 @ep200 → ~0.07 @ep1000).
- Los modelos anchos (`w32`) tienen train_loss muy bajo (0.10–0.13) pero val_loss más
  alto (~0.20) → **overfitting** claro; no se beneficiaron de más entrenamiento.
- `w8_lr1e-3` no mejora con más entrenamiento (su mejor checkpoint es temprano).
- Lección clave: la palanca más efectiva fue **más épocas sobre `w8_lr3e-4`**, no más
  capacidad (width).

---

## 3. Resultados de evaluación (val / test)

Métricas: `acc` = accuracy top-1; `top5`/`top10` = clase verdadera en top-5/top-10;
`MRR` = Mean Reciprocal Rank; `mean_rank` = rango promedio (1-based) de la clase verdadera.
Para cada run se toma la **mejor combinación estrategia + método en val**.

### 3.1 Mejor modelo: progresión de `w8_lr3e-4` con más épocas

El mejor run (`w8_lr3e-4`, estrategia `centroid_all` + `L1 Distance`) mejora de forma
clara al entrenar más épocas. Este es el resultado **vigente** del proyecto:

| Versión | Val acc | Val top5 | Val MRR | Val mean_rank | Test acc | Test top5 | Test MRR | Test mean_rank |
|---------|---------|----------|---------|---------------|----------|-----------|----------|----------------|
| ep200  | 19.2% | 56.3% | 0.363 | 7.4 | 19.2% | 55.6% | 0.361 | 7.5 |
| ep400  | 25.0% | 66.4% | 0.431 | 5.9 | ~23.4% | ~63% | ~0.41 | ~6.3 |
| **ep1000** | **34.1%** | **81.0%** | **0.537** | **3.7** | **34.0%** | **80.3%** | **0.535** | **3.8** |

→ Pasar de 200 a 1000 épocas casi **dobló** la accuracy (19% → 34%) y subió el top5 a ~80%.
Val y test van muy parejos → no hay overfitting.

### 3.2 Comparación entre runs (modelo a ep200)

Tabla histórica con todos los runs evaluados a 200 épocas (única versión evaluada para
los 5 runs restantes). Reportes en `runs/<run>/ep200/evaluation_report.json`.

| Run | Estrategia | Método | Val acc | Val top5 | Val top10 | Val MRR | Val mean_rank | Test acc | Test top5 | Test top10 | Test MRR | Test mean_rank |
|-----|------------|--------|---------|----------|-----------|---------|---------------|----------|-----------|------------|----------|----------------|
| **w8_lr3e-4** | centroid_all | L1 Distance | **19.2%** | **56.3%** | **77.4%** | **0.363** | **7.4** | **19.2%** | **55.6%** | **77.2%** | **0.361** | **7.5** |
| w32_lr3e-4 | multiprototype_k5 | L1 Distance | 18.2% | 43.0% | 61.5% | 0.317 | 9.9 | 17.9% | 43.8% | 61.8% | 0.317 | 9.8 |
| w16_lr1e-3 | centroid_all | Cosine Similarity | 12.2% | 39.8% | 61.9% | 0.265 | 10.4 | 11.9% | 39.1% | 60.7% | 0.263 | 10.5 |
| w16_lr3e-4 | centroid_all | L2 Distance | 11.8% | 39.5% | 61.5% | 0.265 | 10.4 | 12.0% | 39.3% | 61.0% | 0.265 | 10.5 |
| w32_lr1e-3 | multiprototype_k5 | L1 Distance | 12.6% | 37.2% | 55.4% | 0.260 | 11.7 | 12.5% | 36.9% | 55.2% | 0.258 | 11.7 |
| w8_lr1e-3 | centroid_10 | L1 Distance | 11.0% | 36.7% | 57.8% | 0.248 | 11.7 | 10.6% | 35.7% | 56.5% | 0.243 | 12.1 |

**Conclusiones de evaluación:**

- **Mejor run vigente:** `w8_lr3e-4` a **ep1000** (centroid_all + L1): **34% acc** en val y
  test, top5 ~80%, MRR ~0.54, mean_rank ~3.7 (ver 3.1). A ep200 era 19.2%.
- Val y test son muy parecidos en todos los runs → poco overfitting a nivel de ranking.
- `L1 Distance` (y en menor medida L2/Cosine) son los mejores métodos; `Dot Product` va mal.
- `centroid_all` y `multiprototype_k5` superan a `centroid_5/10/20`.

### 3.3 Post-procesamiento de embeddings (ep3000)

La comparación se ejecutó sobre
`w8_np512_m0.5_lr3e-4_bs16_seed42_fps` a ep3000 con seed 42. Los
hiperparámetros se seleccionaron usando solamente val y se aplicaron sin cambios
a test.

| Variante seleccionada en val | Val acc | Val top5 | Val top10 | Val MRR | Test acc | Test top5 | Test top10 | Test MRR |
|-------------------------------|---------|----------|-----------|---------|----------|-----------|------------|----------|
| Baseline | 62.10% | 97.46% | 99.90% | 0.7707 | 61.22% | 97.52% | 99.81% | 0.7645 |
| + PCA whitening | **69.80%** | **98.44%** | 99.85% | **0.8207** | **68.85%** | **98.10%** | 99.71% | **0.8144** |
| + k-recíproco | 52.66% | 90.84% | 97.13% | 0.6896 | 51.95% | 90.49% | 97.18% | 0.6828 |
| + rank fusion (RRF) | 57.25% | 94.02% | 98.19% | 0.7271 | 56.92% | 93.80% | 98.23% | 0.7231 |
| Whitening + k-recíproco + RRF | 59.22% | 93.21% | 97.73% | 0.7364 | 58.94% | 92.83% | 97.52% | 0.7335 |

Whitening fue la única mejora efectiva: sumó **+7.70 puntos absolutos en val** y
**+7.63 en test** sobre el baseline seleccionado, superando la meta de +5 puntos.
La configuración ganadora fue `multiprototype_k5 / L1 Distance`, conservando
todas las componentes PCA y usando shrinkage `1e-4`.

Las demás selecciones de val fueron:

- k-recíproco: `trimmed_mean_10 / L1`, `k1=10`, `k2=3`, `lambda=0.5`;
- RRF: `L1`, constante 20, fusionando las nueve estrategias disponibles;
- stack: Cosine, con los hiperparámetros anteriores y whitening `all/1e-4`.

El deterioro de k-recíproco probablemente se debe a que la galería tiene sólo
100 identidades y el grafo opera a nivel de clase, un régimen mucho más chico y
menos redundante que las galerías de person re-id para las que fue diseñado. RRF
pondera por igual estrategias fuertes y débiles, por lo que diluye la señal de
`multiprototype_k5`; el stack hereda ambos deterioros y no logra conservar la
ganancia de whitening. Próximas pruebas razonables son fusionar sólo las mejores
estrategias de val, usar RRF ponderado y re-rankear prototipos individuales antes
de colapsar por clase.

Hay una inconsistencia con `docs/README_HANDOFF.md`: allí se esperaba
`centroid_all/L1` y aproximadamente 0.50 de accuracy. La corrida reproducible con
los cachés/manifiestos entregados seleccionó `multiprototype_k5/Cosine` y obtuvo
0.6210 val / 0.6122 test. El directorio `ep3000` contiene nueve estrategias, no
los seis archivos indicados en el handoff, y el reporte preexistente ya señalaba
`multiprototype_k5` como ganador; por lo tanto el handoff parece describir una
versión anterior de los artefactos.

Los algoritmos implementados son:

- **PCA whitening:** ajusta media, ejes principales y varianzas exclusivamente
  con embeddings individuales de train. Proyecta train, val y test, y divide
  cada componente por la desviación regularizada antes de reconstruir las
  estrategias de referencia.
- **Re-ranking k-recíproco:** forma vecindades recíprocas entre la query y las
  clases de la galería, las expande y combina distancia Jaccard con la distancia
  original. Funciona también cuando una clase tiene varios prototipos.
- **Reciprocal Rank Fusion:** fusiona los rankings producidos por las estrategias
  de referencia mediante `sum(1 / (k + rank))`. No usa scores incompatibles entre
  estrategias ni consulta las etiquetas verdaderas.

La grilla default compara dimensiones PCA `all/128/256`, shrinkage
`1e-4/1e-2`, `k1=10/20`, `k2=3/6`, `lambda=0.3/0.5` y constante RRF
`20/60`, sobre Cosine, L2 y L1. Primero se elige la mejor configuración de cada
técnica en val; el stack usa esos hiperparámetros y vuelve a elegir solamente el
método en val. Todos los candidatos, selecciones y métricas finales quedan bajo
`postprocessing` en `evaluation_report.json`.

Comando para producir la tabla:

```bash
python -m src.eval \
  --data_dir <ruta_a_ply> \
  --runs_dir runs \
  --run w8_np512_m0.5_lr3e-4_bs16_seed42_fps \
  --split both \
  --seed 42 \
  --embedding_batch_size 512 \
  --postprocess
```

Resumen estructurado para copiar los resultados:

```bash
jq '.postprocessing | {val, test}' \
  runs/w8_np512_m0.5_lr3e-4_bs16_seed42_fps/ep3000/evaluation_report.json
```

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

1. **Seguir explotando `w8_lr3e-4`.** Es la palanca que más rindió (19% → 34% acc subiendo
   épocas). Conviene ver si sigue mejorando más allá de 1000 épocas y/o con lr decay, y
   replicar la receta (lr 3e-4, más épocas) en `w16_lr3e-4`.
2. **Re-evaluar todos y guardar la salida en JSON/CSV, no en logs de consola** (hoy las
   métricas de ep400/ep1000 solo viven en `eval_*.txt`, que son volcados de terminal):
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

## 7. Procedencia de los datos y discrepancias a verificar

**Fuentes:** las cifras de ep200 vienen de los tres reportes de febrero 2026
(`evaluation_report.json` de cada run). Las cifras de **ep400 y ep1000** se extrajeron de
los logs de consola `eval_w8_lr3e-4_ep400.txt`, `eval_w8_lr3e-4_ep1000.txt`,
`train_w8_lr3e-4_to600.txt` (este último llegó a 1000 épocas pese al nombre) y
`train_w16_lr3e-4_to400.txt`, de fines de abril 2026.

> **Sobre esos `.txt`:** son volcados de terminal en UTF-16, con ruido y errores de lectura
> de PLY. **No conviene versionarlos** (siguen en `.gitignore`); su contenido útil ya quedó
> en este doc. La fuente estructurada real es `runs/<run>/ep<N>/evaluation_report.json`. Si
> querés guardar el detalle por método, exportá los JSON/CSV, no los logs.

Diferencias entre reportes viejos que conviene confirmar contra los `runs/`:

- **Best val_loss de `w8_lr1e-3`:** el reporte viejo de resultados decía 0.1812
  (best en ~ep120), pero el informe de avance lo reportaba como 0.2065 (ep200).
  Verificar en `runs/w8_np512_m0.5_lr1e-3_bs16_seed42/metrics.csv` / `model_version.json`.
- **Mención a "época 291"** para `w8_lr1e-3` en el reporte viejo: imposible en un run de
  200 épocas; probablemente un error de tipeo. Descartado en este doc.
- **top5 de `w8_lr3e-4` en val:** 56.3% (reporte de resultados) vs 52.7% (informe de avance).
  Se tomó 56.3% por venir de la tabla val/test completa; reconfirmar con `evaluation_report.json`.
