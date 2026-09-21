# PR: post-procesamiento para evaluación cerrada

## Resumen

- Agrega PCA whitening regularizado, ajustado sólo con embeddings de train, y
  reconstruye todas las referencias en el espacio transformado.
- Agrega re-ranking k-recíproco con expansión de vecinos, distancia Jaccard y
  mezcla con la distancia original; acepta referencias de uno o varios
  prototipos por clase.
- Agrega Reciprocal Rank Fusion para combinar rankings de las estrategias
  existentes sin mezclar escalas de scores.
- Incorpora un protocolo único que selecciona hiperparámetros en val y aplica la
  decisión sin cambios a test. El comportamiento baseline no cambia si no se usa
  `--postprocess`.

## Comparación

La tabla completa está en la sección 3.3 de `docs/RESULTADOS.md`. PCA whitening
fue la mejor variante: pasó de 0.6210 a 0.6980 de accuracy en val y de 0.6122 a
0.6885 en test. K-recíproco, RRF y el stack no mejoraron el baseline; las
hipótesis y próximos experimentos están documentados junto a la tabla.

| Variante | Val acc | Val top5 | Val top10 | Val MRR | Test acc | Test top5 | Test top10 | Test MRR |
|----------|---------|----------|-----------|---------|----------|-----------|------------|----------|
| Baseline | 62.10% | 97.46% | 99.90% | 0.7707 | 61.22% | 97.52% | 99.81% | 0.7645 |
| PCA whitening | **69.80%** | **98.44%** | 99.85% | **0.8207** | **68.85%** | **98.10%** | 99.71% | **0.8144** |
| k-recíproco | 52.66% | 90.84% | 97.13% | 0.6896 | 51.95% | 90.49% | 97.18% | 0.6828 |
| RRF | 57.25% | 94.02% | 98.19% | 0.7271 | 56.92% | 93.80% | 98.23% | 0.7231 |
| Whitening + k-recíproco + RRF | 59.22% | 93.21% | 97.73% | 0.7364 | 58.94% | 92.83% | 97.52% | 0.7335 |

Whitening supera el baseline en **+7.70 puntos absolutos de val** y **+7.63 de
test**, por encima de la meta propuesta de +5 puntos.

## Hiperparámetros y selección

La grilla predeterminada es:

- whitening: componentes `all,128,256`; shrinkage `0.0001,0.01`;
- k-recíproco: `k1=10,20`; `k2=3,6`; `lambda=0.3,0.5`;
- RRF: constante `k=20,60`;
- métodos: Cosine Similarity, L2 Distance y L1 Distance;
- estrategias fusionadas: todas las estrategias generables disponibles.

Cada mejora individual se selecciona por accuracy de val, con MRR, top-5,
top-10 y mean rank como desempates deterministas. El stack reutiliza los mejores
hiperparámetros individuales y selecciona su método final también en val. Test no
participa en ninguna elección.

## Decisiones de diseño

- El re-ranking trabaja sobre clases: para una referencia multiprototipo usa el
  mejor par de prototipos para construir la relación entre clases. Esto conserva
  una salida de una posición por identidad y evita fusionar prototipos como si
  fueran clases diferentes.
- RRF fusiona rankings del mismo método a través de estrategias. Al usar rangos,
  no presupone que scores de centroides y multiprototipos tengan igual escala.
- Los artefactos transformados viven en memoria. El manifiesto registra toda la
  grilla, de modo que reanudar sólo reutiliza reportes exactamente compatibles.
- `--postprocess` requiere val y por ahora no se combina con open-set; el umbral
  de rechazo abierto necesita un protocolo de calibración específico.

## Validación

Validaciones estáticas realizadas durante la implementación:

```bash
python -m compileall src scripts experiments tests
git diff --check
```

No se ejecutó el suite unitario durante la implementación inicial. La evaluación
real completa sí se ejecutó sobre CPU con:

```bash
python -m src.eval \
  --data_dir <ruta_a_ply> \
  --run w8_np512_m0.5_lr3e-4_bs16_seed42_fps \
  --split both \
  --seed 42 \
  --embedding_batch_size 512 \
  --postprocess
```

El bloque post-processing tardó 3.148 segundos y reutilizó los cachés compatibles
de train/val; se generó el caché faltante de test.

## Ubicación de resultados

- Resultado versionado y análisis: `docs/RESULTADOS.md`, sección 3.3.
- Reporte estructurado completo local:
  `runs/w8_np512_m0.5_lr3e-4_bs16_seed42_fps/ep3000/evaluation_report.json`.
- Las grillas, candidatos de val, configuraciones seleccionadas y métricas finales
  están bajo la clave `postprocessing` del JSON.
- `runs/` permanece fuera de Git intencionalmente porque contiene artefactos de
  ejecución; la tabla versionada es el registro que acompaña al PR.

## Dependencia de la rama

Este cambio se desarrolló sobre `fix/eval-fix`, que contiene la versión del
pipeline de evaluación usada por los artefactos ep3000. El PR se abre con esa
rama como base para mantener un diff enfocado; después de integrar esa dependencia,
la base/destino final debe quedar en `main`.
