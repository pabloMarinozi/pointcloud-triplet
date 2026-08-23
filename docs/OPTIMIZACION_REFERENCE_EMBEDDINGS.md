# Optimización de *reference embeddings* para evaluación

## Objetivo y alcance

Este documento analiza cómo se construyen y utilizan actualmente los *reference
embeddings* del proyecto y propone un plan experimental para mejorar la precisión de
clasificación durante la evaluación.

El alcance principal son cambios posteriores al entrenamiento: muestreo de las nubes,
generación de embeddings, construcción de prototipos y elección de la distancia. Las
modificaciones de la arquitectura o de la función de pérdida se consideran una fase
posterior porque exigen volver a entrenar los modelos.

Hay que distinguir dos objetivos:

1. **Mejorar la puntuación bajo el protocolo actual**, manteniendo los mismos splits para
   poder comparar con los resultados históricos.
2. **Mejorar la generalización real**, midiendo el rendimiento sobre videos o sesiones
   que no estuvieron presentes en train.

Todas las variantes deben elegirse utilizando exclusivamente validación. El conjunto de
test debe reservarse para una única evaluación final de la configuración seleccionada.

---

## Diagnóstico

### 1. Estado actual

La implementación genera seis estrategias en
[`src/evaluation/ref_strategies.py`](../src/evaluation/ref_strategies.py):

- `centroid_5`, `centroid_10`, `centroid_20` y `centroid_all`;
- `centroid_l2norm_5`;
- `multiprototype_k5`.

Los centroides son medias de embeddings y el multiprototipo utiliza K-means euclídeo.
Durante la predicción se compara cada query contra un vector o contra el prototipo más
cercano de cada clase mediante las métricas definidas en
[`src/evaluation/metrics.py`](../src/evaluation/metrics.py).

En el mejor run disponible, `w8_np512_m0.5_lr3e-4_bs16_seed42` a 1000 épocas, el mejor
resultado guardado es:

| Split | Estrategia | Distancia | Accuracy | Top-5 | MRR |
|---|---|---|---:|---:|---:|
| Validación | `centroid_all` | L1 | 34,76% | 80,88% | 0,5402 |
| Test | `centroid_all` | L1 | 34,07% | 79,99% | 0,5340 |

Fuente local: `runs/w8_np512_m0.5_lr3e-4_bs16_seed42/ep1000/evaluation_report.json`.

La mejora al aumentar la cantidad de referencias es consistente: en validación, usando
L1, `centroid_5` obtiene 29,52%, `centroid_20` 33,72% y `centroid_all` 34,76%. Esto indica
que reducir el ruido de estimación del prototipo es una palanca relevante.

### 2. La media no es el prototipo natural para L1

Un embedding es solamente un vector numérico. L1 es la regla usada para medir la
distancia entre dos de esos vectores:

\[
d_{L1}(x,y)=\sum_j |x_j-y_j|.
\]

Actualmente el prototipo de cada clase es la media:

\[
p_c=\frac{1}{N_c}\sum_{i=1}^{N_c}z_i.
\]

La media minimiza la suma de distancias euclídeas **cuadráticas**. En cambio, el vector
que minimiza la suma de distancias L1 se obtiene tomando la mediana de cada coordenada:

\[
p_c[j]=\operatorname{median}\{z_1[j],\ldots,z_{N_c}[j]\}.
\]

Ejemplo en una dimensión:

```text
Embeddings: 1, 2, 3, 100
Media:      26,5
Mediana:     2,5
```

El valor atípico `100` desplaza fuertemente la media, mientras que la mediana permanece
cerca del grupo principal. Esto no garantiza por sí solo mayor accuracy, pero produce un
estimador robusto y coherente con la distancia ganadora.

Prototypical Networks justifica la media cuando se combina con divergencias de Bregman,
como la distancia euclídea cuadrática, y muestra que la elección de la distancia es una
parte esencial del clasificador [1]. El pipeline actual combina media con L1, por lo que
`median_all + L1` es el primer experimento que debería realizarse.

### 3. Cada `.ply` se representa con un único muestreo aleatorio

[`src/evaluation/embed.py`](../src/evaluation/embed.py) normaliza la nube, selecciona una
sola muestra de `n_points` y ejecuta el modelo una vez. En consecuencia, el embedding
contiene dos fuentes de variabilidad:

- la geometría y calidad de la nube original;
- el subconjunto aleatorio de puntos elegido durante esa corrida.

Aunque la evaluación fija una semilla global, una sola vista sigue siendo una estimación
Monte Carlo de alta varianza y además depende del orden en el que se recorren las nubes.
PointNet fue diseñado para conjuntos no ordenados y estudió su robustez ante puntos
faltantes y perturbaciones [2], pero eso no implica que diferentes subconjuntos produzcan
exactamente el mismo embedding. Trabajos recientes de adaptación de nubes de puntos usan
explícitamente variaciones de muestreo para aumentar la robustez en evaluación [3].

La alternativa propuesta es obtener varias vistas deterministas de cada nube y agregar
sus embeddings:

\[
z(x)=\operatorname{median}\{f(S_1(x)),\ldots,f(S_M(x))\},
\]

donde `S_m` es un muestreo reproducible. Inicialmente deben probarse `M = 4` y `M = 8`.
La misma operación debe aplicarse a referencias y queries.

### 4. La evaluación no propaga la estrategia de muestreo del run

`embed_point_cloud_path` permite elegir `random`, `fps` o `fps_baya`, pero las llamadas
que generan referencias y queries no pasan el campo `sampling` de `config.json`. Por lo
tanto, terminan usando el valor predeterminado `random`, incluso al evaluar un modelo
entrenado con FPS.

Esto introduce una diferencia entre el preprocesamiento de entrenamiento y evaluación.
Antes de comparar nuevas estrategias de prototipos hay que propagar de manera explícita
la estrategia guardada en la configuración del run, usando `random` solamente como
compatibilidad para runs antiguos cuyo valor sea nulo.

### 5. La normalización actual es asimétrica

`centroid_l2norm_5` normaliza los embeddings de referencia antes y después de calcular la
media, pero las queries continúan en el espacio original. Comparar una query sin
normalizar contra una referencia normalizada mediante L1 o L2 mezcla escalas diferentes.
Esto coincide con la fuerte caída observada de `centroid_l2norm_5 + L1`, que obtiene
alrededor de 4% de accuracy.

SimpleShot mostró que transformaciones sencillas como centrar y normalizar los features
pueden mejorar clasificadores de vecino más cercano [4]. Sin embargo, cualquier
transformación debe ajustarse con train y aplicarse de forma idéntica a:

- embeddings usados para construir referencias;
- prototipos resultantes, cuando corresponda;
- embeddings de validación y test.

Se debe evaluar como una familia separada: `center + L2 normalize + cosine/L2`. No debe
mezclarse una referencia normalizada con una query cruda.

### 6. K-means no está alineado con L1

`multiprototype_k5` utiliza K-means, cuyo centro y función objetivo corresponden a
distancias euclídeas cuadráticas. Después, el mejor prototipo se selecciona mediante L1.
Las alternativas compatibles son:

- K-medians, usando medianas por coordenada;
- K-medoids con distancia Manhattan, donde cada centro es una observación real;
- agrupamiento por video o forma de captura, si esos metadatos representan modos reales
  de adquisición.

También hay que seleccionar `k` en validación y no fijarlo siempre en cinco. Los métodos
con múltiples proxies se utilizan precisamente para representar distribuciones de clase
multimodales [5], pero agregar prototipos sin una regla de calibración puede favorecer a
una clase simplemente por tener más oportunidades de producir una distancia mínima.

Además del mínimo actual deben probarse:

- promedio de las dos distancias más bajas;
- `softmin` con temperatura seleccionada en validación;
- cantidad igual de prototipos válidos por clase.

### 7. La mayoría de las dimensiones aporta muy poca separación entre prototipos

La inspección de `reference_embeddings_centroid_all.npz` del mejor run muestra:

- 100 prototipos de 512 dimensiones;
- 494 dimensiones con desviación estándar entre clases menor que `0.01`;
- las 16 dimensiones de mayor varianza concentran 98,28% de la varianza total entre
  prototipos;
- las normas L2 de los prototipos tienen un coeficiente de variación aproximado de 0,53%.

Esto no demuestra por sí solo que las otras dimensiones deban eliminarse: para decidirlo
también se necesita medir su variación dentro de cada clase. Sí demuestra que todas las
coordenadas no parecen ser igual de discriminativas, mientras que L1 les asigna el mismo
peso.

Debe probarse una distancia L1 ponderada:

\[
d_w(x,p_c)=\sum_j w_j|x_j-p_c[j]|, \qquad w_j\geq0.
\]

Una primera estimación interpretable es el cociente de Fisher diagonal:

\[
w_j=\frac{\operatorname{Var}_{entre\ clases}(j)}
{\operatorname{Var}_{dentro\ de\ clases}(j)+\epsilon}.
\]

También se puede comparar selección de las mejores `16`, `32`, `64` y `128` dimensiones.
LMNN demuestra que aprender una transformación o métrica supervisada puede mejorar
sustancialmente la clasificación por vecinos cercanos [6]. Para evitar sobreajuste en
este dataset, debe comenzar por pesos diagonales regularizados antes de intentar una
matriz Mahalanobis completa de `512 × 512`.

### 8. Todas las nubes tienen el mismo peso, pero los videos no

En train hay aproximadamente 322–434 nubes por clase, 23–31 videos por clase y 8–19
nubes por combinación `(clase, video)`. La media global da más peso a los videos que
generaron más nubes.

Si cada video representa una sesión o condición de captura, una construcción jerárquica
evita esa sobrerrepresentación:

1. agregar los embeddings de cada `(clase, video)`;
2. dar el mismo peso a cada video al construir la referencia de clase.

Las variantes iniciales deberían ser:

- `video_mean -> class_mean`;
- `video_median -> class_median`;
- media recortada, eliminando el 5% o 10% de observaciones más alejadas del centro
  preliminar.

Las propuestas de prototipos ponderados también parten de la limitación de que una media
simple asigna igual importancia a todas las observaciones, incluidas las atípicas [7].

### 9. El caché de referencias no describe cómo fue generado

Los nombres `reference_embeddings_<strategy>.npz` no registran:

- seed;
- estrategia de muestreo;
- número de vistas;
- tipo de agregación;
- normalización o transformación aplicada;
- hash o versión del checkpoint;
- lista de paths utilizada.

Esto permite reutilizar accidentalmente referencias incompatibles con una evaluación
nueva. Cada artefacto debe tener metadatos verificables y el pipeline debe regenerarlo si
no coincide con la configuración solicitada.

### 10. El split actual comparte videos entre train, validación y test

El split se realiza por muestra dentro de cada identidad mediante `train_test_split` en
[`src/data/splits.py`](../src/data/splits.py), sin agrupar por video. En los splits del
mejor run se encontró:

- 100% de los videos de validación también presentes en train;
- 100% de los videos de test también presentes en train;
- 2.544 videos presentes simultáneamente en train, validación y test.

Esto no invalida el resultado si la tarea de despliegue consiste en reconocer nuevas
nubes de videos ya conocidos. Si se espera generalizar a videos o sesiones nuevas, la
puntuación actual es optimista y responde a una tarea diferente.

La documentación de scikit-learn recomienda validación por grupos cuando existen
observaciones relacionadas y ofrece `GroupShuffleSplit` y `StratifiedGroupKFold` para
evitar que un mismo grupo aparezca en ambos lados del split [8]. En este proyecto se
puede realizar un split por video dentro de cada identidad, verificando que cada clase
conocida conserve suficientes videos en train, validación y test.

Se deben mantener dos reportes:

- **protocolo histórico por nube**, para comparar con los runs existentes;
- **protocolo por video no visto**, para estimar generalización real.

---

## Plan de implementación

### Fase 0 — Hacer confiable y eficiente la evaluación

Antes de crear más prototipos, conviene separar dos operaciones:

1. generar y almacenar embeddings individuales;
2. construir estrategias de referencia y calcular métricas a partir del caché.

El caché debería contener como mínimo:

```text
path, label, video, capture_form, view_id, seed, embedding
```

Y un manifiesto JSON asociado:

```json
{
  "checkpoint": "...",
  "checkpoint_sha256": "...",
  "n_points": 512,
  "sampling": "random",
  "views": 8,
  "view_aggregation": "coordinate_median",
  "normalization": "unit_sphere",
  "seed": 42,
  "split_paths_sha256": "..."
}
```

Esto permite probar media, mediana, pesos, selección de dimensiones y multiprototipos
sin volver a ejecutar el modelo para cada estrategia. La inferencia debe procesarse en
batches para reducir considerablemente el tiempo de generación.

También se deben añadir pruebas unitarias pequeñas para garantizar:

- que la mediana se calcula por coordenada;
- que referencias y queries reciben la misma transformación;
- que una configuración de caché incompatible obliga a regenerar;
- que una semilla fija produce exactamente los mismos embeddings y métricas;
- que el `sampling` del run llega hasta todas las llamadas de embedding.

### Fase 1 — Baseline robusto `median_all + L1`

Implementar `build_references_median` y guardarlo como una estrategia independiente, sin
reemplazar `centroid_all`. Comparar ambos usando exactamente los mismos embeddings
individuales cacheados.

Experimentos:

| Estrategia | Agregación | Distancia |
|---|---|---|
| Baseline | media de todas las muestras | L1 |
| `median_all` | mediana por coordenada | L1 |
| `trimmed_mean_05` | media recortada 5% | L1 |
| `trimmed_mean_10` | media recortada 10% | L1 |

Las medias recortadas eliminan por coordenada el 5% o 10% de cada cola antes de
calcular la media. Si una clase no tiene suficientes muestras para recortar al menos un
valor por cola, la estrategia coincide con la media sin recortar.

Criterio de avance: mejora estable en validación y ausencia de degradación relevante en
Top-5 y MRR. No se debe consultar test durante esta selección.

### Fase 2 — Multi-sampling determinista

Extender la generación para producir `M` vistas por nube. Cada vista debe tener una
semilla derivada de forma estable a partir de `(seed, path, view_id)`, no del orden global
del recorrido.

Orden recomendado de experimentos:

1. `random`, con `M ∈ {1, 4, 8}`;
2. FPS determinista, si su implementación conserva el mismo resultado entre corridas;
3. combinación de random y FPS;
4. solamente después, rotaciones o jitter compatibles con el problema físico.

La agregación primaria será mediana por coordenada. También se comparará media. El mismo
número y tipo de vistas se aplicará a la galería y a las queries.

Debe medirse el compromiso entre accuracy y coste:

| Vistas | Accuracy | MRR | Tiempo de referencias | Tiempo/query |
|---:|---:|---:|---:|---:|
| 1 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |

### Fase 3 — Prototipos balanceados y robustos

Incorporar `video` y `capture_form` al caché y construir prototipos jerárquicos. Evaluar:

1. balanceo por video;
2. balanceo por forma de captura;
3. mediana de prototipos de video;
4. rechazo de outliers por distancia al prototipo preliminar.

Los umbrales de rechazo deben elegirse en validación y reportarse. Nunca deben eliminarse
muestras usando su comportamiento sobre test.

### Fase 4 — Selección y ponderación de dimensiones

Guardar todos los embeddings individuales de train para estimar variación intra e
interclase. Implementar progresivamente:

1. ranking de dimensiones por cociente de Fisher;
2. selección `top-k`, con `k ∈ {16, 32, 64, 128, 256, 512}`;
3. L1 ponderada con pesos normalizados y recortados;
4. regularización de los pesos hacia `1`;
5. opcionalmente NCA o LMNN después de establecer un baseline diagonal.

Los pesos se aprenden únicamente con train o mediante un split interno de train. El
número de dimensiones y la regularización se seleccionan con validación.

### Fase 5 — Multiprototipos compatibles con la distancia

Implementar K-medoids Manhattan o K-medians para `k ∈ {2, 3, 5}`. Comparar reglas de
score por clase:

```text
min(distancias)
media(de las 2 menores)
softmin(distancias, temperatura)
```

Agregar además dos baselines que no comprimen la clase en un único centro:

- vecino individual más cercano;
- promedio de los `k` ejemplares más cercanos por clase.

Esto permitirá determinar si el problema está en la estimación del centro o en que cada
identidad tiene varios modos geométricos reales.

### Fase 6 — Normalización simétrica

Crear una interfaz de transformación ajustable sobre train y reutilizable sin cambios:

```text
fit(train_embeddings)
transform(reference_embeddings)
transform(query_embeddings)
```

Comparar como experimentos independientes:

- crudo + L1;
- centrado global + L1;
- centrado global + normalización L2 + coseno;
- centrado global + normalización L2 + L2;
- estandarización robusta por dimensión + L1.

No conservar `centroid_l2norm_5` en su forma asimétrica actual.

### Fase 7 — Evaluación por grupos y reporte final

Construir splits adicionales agrupando por video dentro de cada identidad. Validar que:

- ningún video aparezca en más de un split;
- cada clase conocida tenga suficientes videos en los tres conjuntos;
- las proporciones se aproximen a 70/15/15;
- los splits queden almacenados y versionados.

Para cada protocolo se reportará:

- accuracy Top-1, Top-5 y Top-10;
- MRR, mean rank y median rank;
- accuracy macro por identidad;
- métricas por video y forma de captura;
- media y desviación sobre varias semillas o vistas deterministas;
- intervalo de confianza mediante bootstrap agrupado por video;
- tiempo de construcción de referencias y latencia por query.

La configuración final será la mejor de validación. Test se ejecutará una sola vez y el
resultado se añadirá sin modificar después los hiperparámetros.

### Matriz mínima de experimentos

Para evitar una búsqueda combinatoria grande, se recomienda avanzar de forma acumulativa:

| ID | Vistas | Prototipo | Métrica | Objetivo |
|---|---:|---|---|---|
| E0 | 1 | media global | L1 | Reproducir baseline |
| E1 | 1 | mediana global | L1 | Alinear prototipo y distancia |
| E2 | 4 | mediana global | L1 | Reducir ruido de muestreo |
| E3 | 8 | mediana global | L1 | Medir saturación de multi-sampling |
| E4 | mejor E2/E3 | mediana balanceada por video | L1 | Evitar sesgo por video |
| E5 | mejor anterior | mediana | L1 ponderada | Aprovechar dimensiones discriminativas |
| E6 | mejor anterior | K-medoids `k=2/3/5` | L1 ponderada | Modelar multimodalidad |
| E7 | mejor anterior | mejor prototipo | transformación simétrica | Comparar normalización |

Una variante solo avanza si mejora validación de manera consistente. Si dos alternativas
empatan, debe elegirse la de menor coste y menor cantidad de hiperparámetros.

---

## To do list

### P0 — Corrección y reproducibilidad

- [x] Leer `sampling` desde `config.json` y pasarlo a la generación de referencias.
- [x] Pasar el mismo `sampling` a validación, test y open-set.
- [x] Mantener `random` como fallback documentado para runs antiguos con `sampling=null`.
- [x] Derivar semillas por `(path, view_id)` para eliminar dependencia del orden.
- [x] Guardar un manifiesto con checkpoint, split, seed, muestreo y transformaciones.
- [x] Invalidar automáticamente caches cuyos metadatos no coincidan.
- [x] Agregar tests de determinismo y simetría del preprocesamiento.
- [x] Verificar que dos evaluaciones con la misma configuración sean idénticas.

### P1 — Mejoras de alto retorno sin reentrenar

- [x] Cachear embeddings individuales de train, validación y test.
- [x] Vectorizar/batchear la generación para evitar inferencia muestra por muestra.
- [x] Implementar `median_all`.
- [x] Implementar medias recortadas de 5% y 10%.
- [x] Integrar media, mediana y medias recortadas al mismo pipeline y caché.
- [ ] Ejecutar la comparación en validación y registrar accuracy, Top-5 y MRR.
- [x] Implementar multi-sampling con `M=4` y `M=8`.
- [x] Aplicar la misma agregación multi-vista a referencias y queries.
- [x] Registrar tiempo, memoria y latencia además de accuracy.
- [x] Programar una corrida reproducible que consolide la comparación P1.

### P2 — Balanceo y robustez

- [x] Asociar cada embedding con `video` y `capture_form`.
- [ ] Implementar prototipos jerárquicos balanceados por video.
- [ ] Implementar prototipos balanceados por forma de captura.
- [ ] Medir distancia de cada referencia a su centro de clase.
- [ ] Implementar rechazo o reducción de peso de outliers.
- [ ] Analizar si las muestras eliminadas corresponden a PLY corruptos o capturas difíciles.

### P3 — Métrica y dimensiones

- [ ] Calcular varianza intra e interclase por dimensión usando solamente train.
- [ ] Generar un reporte de dimensiones saturadas o poco discriminativas.
- [ ] Implementar selección `top-k` de dimensiones.
- [ ] Implementar L1 ponderada mediante cociente de Fisher regularizado.
- [ ] Seleccionar `k` y regularización exclusivamente en validación.
- [ ] Comparar posteriormente NCA/LMNN contra el baseline diagonal.

### P4 — Multimodalidad

- [ ] Implementar K-medians o K-medoids con Manhattan.
- [ ] Evaluar `k=2`, `k=3` y `k=5`.
- [ ] Implementar score mínimo, promedio top-2 y `softmin`.
- [ ] Garantizar igual cantidad efectiva de prototipos por clase.
- [ ] Implementar baselines por ejemplares: 1-NN y top-k por clase.
- [ ] Inspeccionar si los clusters se relacionan con videos o formas de captura.

### P5 — Normalización coherente

- [ ] Refactorizar transformaciones con interfaz `fit/transform`.
- [ ] Aplicar siempre la misma transformación a referencias y queries.
- [ ] Evaluar centrado + normalización L2 + coseno/L2.
- [ ] Evaluar estandarización robusta + L1.
- [ ] Retirar o renombrar la estrategia asimétrica `centroid_l2norm_5`.

### P6 — Protocolo de generalización

- [ ] Extraer un identificador de video estable para cada path.
- [ ] Crear splits 70/15/15 sin videos compartidos.
- [ ] Agregar una validación automática de intersección de grupos.
- [ ] Conservar los splits históricos por nube para comparabilidad.
- [ ] Reportar por separado protocolo histórico y protocolo por video no visto.
- [ ] Seleccionar toda configuración con validación y ejecutar test una sola vez.
- [ ] Calcular intervalos de confianza agrupados por video.
- [ ] Documentar el coste computacional de la configuración ganadora.

### P7 — Cambios que requieren reentrenamiento

- [ ] Comparar la triplet loss euclídea actual con una función alineada con la métrica final.
- [ ] Evaluar embeddings normalizados durante entrenamiento, no solamente al evaluar.
- [ ] Probar Proxy Anchor u otra pérdida basada en proxies [9].
- [ ] Analizar si la activación `Sigmoid` final causa saturación de dimensiones.
- [ ] Repetir la mejor evaluación post-hoc sobre cada modelo reentrenado.

---

## Resultado esperado

La hipótesis principal a validar es:

```text
8 vistas deterministas por nube
    -> mediana por nube
    -> agregación balanceada por video
    -> mediana de clase
    -> L1 ponderada regularizada
```

Esta combinación ataca simultáneamente:

- el ruido producido por seleccionar solo 512 puntos;
- los outliers dentro de cada identidad;
- la incompatibilidad entre media y L1;
- la sobrerrepresentación de algunos videos;
- el peso uniforme de dimensiones poco discriminativas.

Debe considerarse una hipótesis, no una conclusión anticipada. La mejora real se aceptará
solo si aparece de forma reproducible en validación, conserva las métricas de ranking y
se confirma una única vez sobre test.

---

## Referencias

1. Snell, J., Swersky, K. y Zemel, R. S. (2017). **Prototypical Networks for
   Few-shot Learning**. NeurIPS 2017.
   [Paper](https://papers.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)

2. Qi, C. R., Su, H., Mo, K. y Guibas, L. J. (2017). **PointNet: Deep Learning on
   Point Sets for 3D Classification and Segmentation**. CVPR 2017.
   [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html)

3. Bahri, A. et al. (2025). **Test-Time Adaptation in Point Clouds: Leveraging
   Sampling Variation with Weight Averaging**. WACV 2025.
   [Paper](https://openaccess.thecvf.com/content/WACV2025/html/Bahri_Test-Time_Adaptation_in_Point_Clouds_Leveraging_Sampling_Variation_with_Weight_WACV_2025_paper.html)

4. Wang, Y., Chao, W.-L., Weinberger, K. Q. y van der Maaten, L. (2019).
   **SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning**.
   [Paper](https://arxiv.org/abs/1911.04623)

5. Yu, Y., Zhang, D., Li, Y. y Zhang, Z. (2022). **Multi-Proxy Learning from an
   Entropy Optimization Perspective**. IJCAI 2022, 1594–1600.
   [Paper](https://www.ijcai.org/proceedings/2022/222)

6. Weinberger, K. Q. y Saul, L. K. (2009). **Distance Metric Learning for Large
   Margin Nearest Neighbor Classification**. Journal of Machine Learning Research,
   10, 207–244.
   [Paper](https://www.jmlr.org/papers/v10/weinberger09a.html)

7. Roy Chowdhury, R. y Bathula, D. R. (2021). **Influential Prototypical Networks
   for Few Shot Learning: A Dermatological Case Study**.
   [Paper](https://arxiv.org/abs/2111.00698)

8. scikit-learn. **Cross-validation: evaluating estimator performance — Grouped
   cross-validation**.
   [Documentación](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)

9. Kim, S., Kim, D., Cho, M. y Kwak, S. (2020). **Proxy Anchor Loss for Deep Metric
   Learning**. CVPR 2020.
   [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)

---

## Registro de ejecución de la Fase 0

### Instrucción registrada

Ejecutar la Fase 0 de este documento, registrar al final lo realizado y sus beneficios
respecto del estado previo y, una vez terminada, actualizar la lista de tareas.

### Realizado

- Se incorporó un caché individual por split en
  `individual_embeddings_<split>.npz`, acompañado por
  `individual_embeddings_<split>.manifest.json`. Cada registro conserva `path`,
  `label`, `video`, `capture_form`, `view_id`, `seed` y `embedding`.
- El manifiesto registra versión de formato, path y SHA-256 del checkpoint, hash del
  split, cantidad de muestras, `n_points`, `sampling`, vistas, agregación,
  normalización, augmentation, tamaño de batch y semilla base.
- El caché se valida antes de reutilizarlo. Un cambio de checkpoint, split, seed,
  muestreo, cantidad de puntos o preprocesamiento provoca su regeneración automática.
- Las referencias tienen además un manifiesto que las vincula con el caché de train.
  Los artefactos históricos sin manifiesto se regeneran una vez con la configuración
  explícita del run.
- `sampling` se lee desde `config.json` y llega a train, validación, test y open-set.
  Para runs antiguos donde el campo falta o vale `null`, se informa y utiliza
  `random`.
- El muestreo dejó de depender del estado aleatorio global: cada vista usa una semilla
  SHA-256 derivada de `(seed, path, view_id)`. Recorrer el split en otro orden produce
  los mismos puntos y embeddings para cada path.
- La normalización, el muestreo y la augmentation se ejecutan mediante una única ruta
  de preprocesamiento compartida por referencias y queries.
- La inferencia se agrupó en batches configurables mediante
  `--embedding_batch_size` (valor predeterminado: `64`). Validación, test y open-set
  generan sus embeddings una vez y todas las estrategias reutilizan el mismo caché.
- Los reportes guardan la configuración y los hashes de los splits; ya no se reanudan
  métricas producidas con otro checkpoint, muestreo, seed o preprocesamiento.
- Se añadieron pruebas para mediana por coordenada, simetría del preprocesamiento,
  semillas estables, independencia del orden, igualdad de métricas, metadatos e
  invalidación del caché.

Validación realizada:

```text
.venv/bin/python -m unittest discover -s tests -v
11 tests ejecutados — OK

.venv/bin/python -m compileall src tests scripts experiments
sin errores de sintaxis
```

### Beneficios respecto del estado previo

- **Comparaciones confiables:** referencias y queries ahora usan el `sampling` real
  del entrenamiento y el mismo preprocesamiento. Antes, una evaluación de un modelo
  entrenado con FPS podía terminar usando `random` silenciosamente.
- **Reproducibilidad independiente del recorrido:** antes una única secuencia aleatoria
  global hacía que cambiar el orden de los paths alterara las muestras posteriores.
  Ahora cada nube queda asociada de forma estable a su propia semilla.
- **Menos inferencia repetida:** con `S` estrategias, validación, test y open-set
  ejecutaban nuevamente el modelo hasta `S` veces sobre las mismas queries. Ahora se
  ejecuta una vez por configuración y luego se calculan todas las estrategias desde el
  caché.
- **Mejor aprovechamiento del dispositivo:** antes cada nube implicaba un forward de
  tamaño uno. Con batch `B`, la cantidad de forwards para generar un split pasa de
  aproximadamente `N` a `ceil(N/B)`; el valor exacto de aceleración dependerá de CPU,
  GPU, disco y memoria disponible.
- **Sin reutilización silenciosa de artefactos incompatibles:** los `.npz` anteriores
  no describían checkpoint, split, semilla o muestreo. Los manifiestos permiten detectar
  esas diferencias antes de calcular o reanudar métricas.
- **Base para las fases siguientes:** media, mediana, selección de dimensiones y
  multiprototipos podrán compararse sobre exactamente los mismos embeddings sin volver
  a ejecutar el modelo para cada alternativa.

La Fase 0 mejora confiabilidad y coste computacional; no se atribuye todavía una mejora
de accuracy. Esa comparación corresponde a los experimentos de las fases siguientes.

---

## Registro de implementación de la Fase 1

### Realizado

- Se agregó `median_all`, que construye un prototipo por clase mediante la mediana de
  cada coordenada usando todos los embeddings de train.
- Se agregaron `trimmed_mean_05` y `trimmed_mean_10`, con recorte simétrico por
  coordenada del 5% y 10% de cada cola.
- Las tres estrategias se construyen junto con `centroid_all` desde la misma carga del
  caché individual de train. Validación reutiliza también un único caché para todas las
  estrategias, por lo que la comparación no incorpora nuevas muestras aleatorias.
- El manifiesto de estrategias pasó a la versión 2 para invalidar artefactos que no
  contengan las nuevas referencias.
- Se añadieron pruebas de mediana por clase, recorte por coordenada, validación de
  proporciones y generación conjunta de todas las estrategias.
- Se implementó multi-sampling determinista con `M=4` y `M=8`. Cada PLY se lee y
  normaliza una sola vez por corrida; después se generan las vistas con semillas
  derivadas de `(seed, path, view_id)`.
- Las vistas se agregan con la misma función para train y validation. La opción
  predeterminada es mediana por coordenada; también se puede comparar media.
- Los reportes guardan tiempo de embedding de train y validation, memoria pico del
  proceso y de CUDA, y latencia de clasificación por query y método.
- Las variantes multi-vista guardan sus artefactos en directorios separados para que
  `M=4` y `M=8` no se sobrescriban ni reutilicen cachés incompatibles.
- Se agregó `scripts/run_p1_comparison.py`, que ejecuta `M=1`, `M=4` y `M=8` sólo
  sobre validation y genera `p1_comparison.json` y `p1_comparison.md`.

Validación realizada:

```text
.venv/bin/python -m unittest discover -s tests -v
19 tests ejecutados — OK

python -m compileall src scripts experiments tests
sin errores de sintaxis
```

### Comando interrumpido y comparación programada

La corrida de una vista que se inició y luego se detuvo por indicación del usuario fue:

```bash
.venv/bin/python -m src.eval \
  --data_dir dataset \
  --run w8_np512_m0.5_lr3e-4_bs16_seed42 \
  --split val \
  --embedding_batch_size 512 \
  --embedding_views 1
```

Se interrumpió durante la generación de train, en 6.144 de 40.026 nubes, antes de que
se guardara el caché. Por lo tanto, no existe un resultado parcial que pueda confundirse
con una evaluación completa.

La comparación completa quedó programada con este comando:

```bash
.venv/bin/python -m scripts.run_p1_comparison \
  --data_dir dataset \
  --run w8_np512_m0.5_lr3e-4_bs16_seed42 \
  --embedding_batch_size 512 \
  --view_aggregation coordinate_median
```

El lanzador ejecuta secuencialmente `M=1`, `M=4` y `M=8`, siempre con `--split val`,
y consolida para `centroid_all`, `median_all`, `trimmed_mean_05` y
`trimmed_mean_10` las métricas L1 de accuracy, Top-5 y MRR, junto con tiempo, memoria y
latencia. Los resultados finales se escriben en:

```text
runs/<run>/ep<N>/p1_comparison.json
runs/<run>/ep<N>/p1_comparison.md
```

Si las tres evaluaciones ya existen, se puede reconstruir la tabla sin inferencia:

```bash
.venv/bin/python -m scripts.run_p1_comparison \
  --data_dir dataset \
  --run w8_np512_m0.5_lr3e-4_bs16_seed42 \
  --only_consolidate
```

La comparación sigue abierta en el todolist porque no se ejecutó completa. El lanzador
no consulta test: la selección de P1 se realiza exclusivamente con validation.
