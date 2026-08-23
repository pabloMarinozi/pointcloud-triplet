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

## Validación propuesta

```bash
python -m compileall src scripts experiments tests
python -m unittest tests.test_postprocessing
python -m unittest discover -s tests
python -m src.eval \
  --data_dir <ruta_a_ply> \
  --run w8_np512_m0.5_lr3e-4_bs16_seed42_fps \
  --split both \
  --seed 42 \
  --embedding_batch_size 512 \
  --postprocess
```
