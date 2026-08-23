# Comparación de embeddings — Fase 0, FPS, epoch 3000

Fecha de ejecución: 2026-08-10  
Run: `w8_np512_m0.5_lr3e-4_bs16_seed42_fps`  
Checkpoint fuente: `runs/w8_np512_m0.5_lr3e-4_bs16_seed42_fps/model_best.pt`  
Snapshot evaluado: `runs/w8_np512_m0.5_lr3e-4_bs16_seed42_fps/ep3000/model.pt`  
SHA-256: `f16c0f2dc575559a12e58e55a6eff149f2771543389923125ce4affdb1296ff3`

## Objetivo

Probar los cambios de Fase 0 sobre el modelo entrenado con FPS durante 3000 epochs y
dejar una comparación reproducible con la evaluación histórica. Esta corrida no
reentrena ni modifica el modelo: vuelve a generar los embeddings de referencia y de
validation con el mismo checkpoint y los mismos splits.

## Configuración de la corrida

| Parámetro | Valor |
|---|---:|
| Checkpoint | epoch 3000 |
| Nubes totales | 57.180 |
| Train | 40.026 |
| Validation | 8.577 |
| Puntos por nube | 512 |
| Width | 8 |
| Sampling | `fps` |
| Seed base | 42 |
| Vistas por nube | 1 |
| Batch de embeddings | 512 |
| Dispositivo | GPU 0 — NVIDIA GeForce GTX 1050, 3 GB |

Comando:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m src.eval \
  --data_dir dataset \
  --run w8_np512_m0.5_lr3e-4_bs16_seed42_fps \
  --split val \
  --embedding_batch_size 512 \
  --embedding_views 1
```

## Antes y ahora

| Aspecto | Antes | Ahora |
|---|---|---|
| Sampling efectivo | Las llamadas desde referencias y evaluación no propagaban el sampling del run; el valor por defecto era `random`, incluso para este modelo FPS. | `src.eval` lee `sampling=fps` de `config.json` y lo propaga a train, validation y open-set. |
| Semillas | El muestreo consumía el estado global de NumPy; el resultado de una nube dependía de qué paths se procesaron antes. | Cada muestra usa una semilla SHA-256 derivada de `(seed, path, view_id)`, independiente del orden de recorrido. |
| Preprocesamiento | Referencias y queries recorrían llamadas separadas y podían usar configuraciones diferentes. | Ambas usan la misma ruta: lectura, normalización a esfera unidad, sampling y augmentation opcional. |
| Inferencia | Un forward por nube, con batch de tamaño 1. | Inferencia batcheada; en esta corrida se usan hasta 512 nubes por forward. |
| Reutilización | Validation se volvía a embeder para cada estrategia de referencia. | Cada split se embede una vez y todas las estrategias consumen el mismo caché. |
| Persistencia | Sólo se guardaban prototipos `reference_embeddings_*.npz`, sin trazabilidad de cómo se obtuvieron. | Se guardan embeddings individuales y un manifiesto con checkpoint/SHA-256, hash del split, sampling, seed, puntos, vistas, augmentation y batch. |
| Compatibilidad | Un `.npz` existente se reutilizaba sólo por nombre, aunque correspondiera a otro preprocesamiento. | El manifiesto se valida; cualquier incompatibilidad regenera automáticamente el caché y las referencias. |
| Metadatos | Los embeddings individuales no quedaban disponibles para auditoría. | Cada registro conserva `path`, `label`, `video`, `capture_form`, `view_id`, `seed` y `embedding`. |
| Estrategias | Centroide 5/10/20/all, centroide L2 y multiprototipo k=5. | Se mantienen las anteriores y se agregan `median_all`, `trimmed_mean_05` y `trimmed_mean_10`, calculadas sobre exactamente el mismo caché de train. |
| Medición operativa | El reporte no registraba costo de generar o clasificar embeddings. | El reporte incluye tiempo, memoria pico, tamaño de caché y latencia de clasificación. |

La diferencia conceptual más importante es que la evaluación histórica no era una
medición fiel de FPS: el modelo había sido entrenado con FPS, pero referencias y
queries podían generarse silenciosamente con random. Por eso los cambios de métricas
de esta corrida no deben interpretarse como cambios en los pesos; miden la corrección
del pipeline de evaluación y el nuevo muestreo reproducible.

## Línea base histórica

El reporte previo fue generado el 2026-08-06 y no contiene `evaluation_manifest` ni
cachés individuales. Métricas de validation con distancia L1:

| Estrategia | Accuracy | Top-5 | MRR |
|---|---:|---:|---:|
| `centroid_5` | 0,4560 | 0,8782 | 0,6320 |
| `centroid_10` | 0,4821 | 0,8986 | 0,6580 |
| `centroid_20` | 0,4947 | 0,8967 | 0,6646 |
| `centroid_all` | 0,4995 | 0,9059 | 0,6729 |
| `centroid_l2norm_5` | 0,0309 | 0,2073 | 0,1414 |
| `multiprototype_k5` | 0,4961 | 0,9331 | 0,6764 |

## Resultado de la corrida nueva

La ejecución finalizó con exit code 0. Se generaron 40.026 embeddings de train y 8.577
de validation, todos de dimensión 512, sin omitir entradas de los splits.

### Comparación directa con la línea base

Las siguientes métricas usan distancia L1 en ambos reportes. `Delta acc.` está
expresado en puntos porcentuales.

| Estrategia | Accuracy antes | Accuracy ahora | Delta acc. | Top-5 antes | Top-5 ahora | MRR antes | MRR ahora |
|---|---:|---:|---:|---:|---:|---:|---:|
| `centroid_5` | 0,4560 | 0,5147 | +5,88 pp | 0,8782 | 0,9092 | 0,6320 | 0,6797 |
| `centroid_10` | 0,4821 | 0,5367 | +5,46 pp | 0,8986 | 0,9173 | 0,6580 | 0,7001 |
| `centroid_20` | 0,4947 | 0,5440 | +4,93 pp | 0,8967 | 0,9305 | 0,6646 | 0,7082 |
| `centroid_all` | 0,4995 | 0,5693 | +6,98 pp | 0,9059 | 0,9341 | 0,6729 | 0,7251 |
| `centroid_l2norm_5` | 0,0309 | 0,0288 | -0,21 pp | 0,2073 | 0,1897 | 0,1414 | 0,1352 |
| `multiprototype_k5` | 0,4961 | 0,6180 | +12,20 pp | 0,9331 | 0,9744 | 0,6764 | 0,7696 |

Cinco de las seis estrategias comparables mejoraron con L1. La excepción es
`centroid_l2norm_5` + L1, una combinación conceptualmente poco alineada: al medir ese
prototipo con cosine en el reporte nuevo alcanza 0,4940 de accuracy. El cambio más
grande aparece en `multiprototype_k5`, que gana 12,20 puntos de accuracy L1.

No se puede atribuir toda la diferencia a una única modificación. El checkpoint y los
splits son los mismos, pero el reporte nuevo corrige simultáneamente el sampling FPS,
la simetría del preprocesamiento, la semilla por path y la regeneración de referencias
incompatibles. Por eso esta tabla compara pipelines de generación de embeddings, no
dos modelos entrenados distintos.

### Estrategias nuevas

| Estrategia | Método | Accuracy | Top-5 | MRR |
|---|---|---:|---:|---:|
| `median_all` | L1 | 0,5737 | 0,9321 | 0,7264 |
| `trimmed_mean_05` | L1 | 0,5743 | 0,9337 | 0,7280 |
| `trimmed_mean_10` | L1 | 0,5772 | 0,9344 | 0,7290 |

Entre los prototipos únicos nuevos, `trimmed_mean_10` fue el mejor y superó a
`centroid_all` por 0,79 puntos de accuracy. El mejor resultado global fue
`multiprototype_k5` con cosine: accuracy 0,6210, Top-5 0,9746 y MRR 0,7707.

### Coste medido y artefactos

| Etapa | Tiempo | Memoria/almacenamiento |
|---|---:|---:|
| Train: embeddings + referencias | 1.993,0 s (33m 13s) | 547,7 MiB CUDA pico; caché 66,15 MiB |
| Validation: embeddings | 431,3 s (7m 11s) | 547,7 MiB CUDA pico; caché 14,19 MiB |
| Clasificación de 9 estrategias × 7 métodos | 678,8 s (11m 19s) | 1.309,7 MiB RSS del proceso |
| Total de etapas registradas | 3.103,1 s (51m 43s) | — |

Artefactos principales:

- `individual_embeddings_train.npz`: 40.026 filas, matriz `(40026, 512)`.
- `individual_embeddings_val.npz`: 8.577 filas, matriz `(8577, 512)`.
- Ambos contienen `path`, `label`, `video`, `capture_form`, `view_id`, `seed` y
  `embedding` con la misma cantidad de registros.
- Los dos manifiestos registran `sampling=fps`, seed 42, una vista, batch 512,
  normalización `unit_sphere`, SHA-256 del checkpoint y hash del split.
- `reference_embeddings.manifest.json` enlaza las nueve estrategias con el manifiesto
  exacto del caché de train.
- `evaluation_report.json` guarda el manifiesto completo, métricas y estadísticas de
  runtime.

Se intentó una segunda ejecución idéntica para observar la reutilización end-to-end,
pero el entorno rechazó la creación del proceso porque el revisor automático del
permiso de GPU agotó su tiempo dos veces. No fue un error del programa. La integridad
se validó directamente: todos los arrays tienen los conteos esperados y los
manifiestos coinciden en checkpoint, hashes, sampling y configuración. Los tests de
compatibilidad/invalidation del caché también pasaron antes de la corrida.

### Advertencia de datos

Open3D informó un fin de archivo inesperado al leer:

```text
dataset/034/034_VID_20230322_135246_nube_8.ply
```

El error apareció en el vértice 18.428. Open3D recuperó suficientes puntos, el batch
se completó y no se omitió la muestra, pero conviene reparar o regenerar ese PLY para
que su lectura no dependa de la tolerancia del parser.

## Validación previa

Antes de lanzar la corrida se ejecutaron:

```text
.venv/bin/python -m unittest discover -s tests -v
19 tests — OK

.venv/bin/python -m compileall -q src scripts experiments tests
sin errores de sintaxis
```
