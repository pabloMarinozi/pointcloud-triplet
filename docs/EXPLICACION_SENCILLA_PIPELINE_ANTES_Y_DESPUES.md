# Explicación sencilla del pipeline: antes, después y beneficios

Este documento explica de punta a punta qué ocurre desde que el programa encuentra
los archivos `.ply` hasta que entrega una métrica de reconocimiento. También separa
qué partes ya existían, qué partes se cambiaron y qué beneficio aporta cada cambio.

La idea central del proyecto es sencilla:

1. cada `.ply` es una nube de puntos de una identidad;
2. el modelo convierte la nube en una lista de números llamada **embedding**;
3. embeddings de la misma identidad deberían quedar cerca;
4. embeddings de identidades diferentes deberían quedar lejos;
5. para reconocer una nube nueva, se la compara contra referencias conocidas.

## Resumen en una frase

Antes, el modelo estaba entrenado con FPS, pero durante la evaluación algunas partes
podían usar muestreo aleatorio sin avisar, repetir miles de cálculos y reutilizar
archivos viejos incompatibles. Ahora toda la evaluación usa la configuración real del
modelo, produce resultados repetibles, guarda cada embedding con su “documento de
identidad” y evita recalcularlo cuando sigue siendo válido.

## El recorrido completo

```text
carpetas de identidades
        ↓
archivos .ply
        ↓
split guardado: train / validation / test
        ↓
leer puntos XYZ
        ↓
centrar y normalizar la nube
        ↓
elegir 512 puntos con FPS
        ↓
modelo PointNet/CPN
        ↓
embedding de 512 números
        ↓
embeddings de train → referencias por identidad
        ↓
embeddings de validation/test → consultas
        ↓
comparar consultas contra referencias
        ↓
predicciones, rankings y métricas
```

La run probada tiene 57.180 nubes y 100 identidades:

| Parte | Cantidad | Uso |
|---|---:|---|
| Train | 40.026 | Entrenar el modelo y construir referencias. |
| Validation | 8.577 | Comparar opciones sin mirar test. |
| Test | 8.577 | Medición final. No se usó en esta corrida de Fase 0. |

## 1. Cómo se obtiene la clase de una nube

La clase es el nombre de la carpeta que contiene el `.ply`.

```text
dataset/034/034_video_nube_8.ply → clase 034
```

### Antes

Ya funcionaba así.

### Ahora

Esto no cambió. La evaluación vuelve a indexar el dataset actual. Si una run fue
creada en otro disco o sistema operativo, intenta encontrar cada archivo por:

```text
(nombre de carpeta, nombre de archivo)
```

### Beneficios

- La identidad no depende de una tabla externa.
- Una run vieja puede evaluarse aunque el dataset se haya movido.
- Se siguen usando los archivos guardados en el split original.

## 2. Cómo se hace el split

El split normal busca aproximadamente:

```text
70% train / 15% validation / 15% test
```

La separación se hace **dentro de cada identidad**. Primero se separa test y después
validation del resto. Se usa la misma seed para que sea repetible.

Cada clase conocida necesita al menos cuatro muestras y debe conservar al menos dos
en train y dos en validation. Esto permite formar tripletas válidas.

### Antes

- El split ya era estratificado por identidad.
- Ya usaba una seed.
- Ya guardaba los paths en `train_paths.json`, `val_paths.json` y `test_paths.json`.
- Si esos archivos existían, no se los sobrescribía.

### Ahora

La forma de dividir train, validation y test **no cambió** con esta mejora.

Lo nuevo es que la evaluación guarda un hash de los paths que realmente reconstruyó.
Ese hash es una huella digital: si cambia una entrada, cambia la huella.

### Beneficios

- Antes y después usan las mismas nubes.
- No se mezcla una evaluación hecha con otro split.
- Si falta un archivo o cambia el conjunto, el caché deja de ser compatible.

## 3. Qué ocurre con open-set

Open-set prueba identidades que el sistema no debería conocer.

Si se activa:

1. se reservan identidades completas como desconocidas;
2. ninguna entra en train/validation/test conocidos;
3. unas identidades desconocidas calibran el umbral;
4. otras identidades desconocidas, separadas, se usan para test.

### Antes

El soporte ya existía en la rama, pero podía recalcular las mismas nubes por caminos
separados y no siempre recibía toda la configuración nueva.

### Ahora

- Cada split open-set puede tener caché propio.
- Se propagan sampling, seed, batch, vistas y agregación.
- Puede usar embeddings ya calculados.

### Beneficios

- No se filtran identidades desconocidas a los splits conocidos.
- Calibración y test usan identidades desconocidas distintas.
- No se repite inferencia por cada referencia.

La corrida FPS analizada usó `open_set_classes=0`, así que esta rama no participó.

## 4. Normalización de la nube

Cada nube se centra restando el promedio XYZ y se escala para que el punto más lejano
quede aproximadamente a distancia 1 del centro. Es como centrar cada objeto dentro de
una esfera de tamaño comparable.

### Antes

La normalización a esfera unidad ya existía.

### Ahora

La fórmula no cambió. Lo importante es que referencias y consultas pasan por una
misma función compartida de preprocesamiento.

### Beneficios

- Se compara forma, no posición o escala bruta.
- Referencias y queries se preparan igual.
- Corregir el camino común corrige ambos lados.

## 5. Por qué se toman 512 puntos

Los `.ply` tienen muchas más coordenadas. La run usa `n_points=512`, por lo que cada
nube se reduce a exactamente esa cantidad. Si tiene menos, se repiten puntos; si tiene
más, se elige un subconjunto.

## 6. Qué es FPS

FPS significa **Farthest Point Sampling**:

1. elige un punto inicial;
2. elige puntos alejados de los ya elegidos;
3. continúa hasta llegar a 512.

Busca cubrir toda la forma, en vez de concentrar puntos en una zona.

Antes de FPS se permuta el array porque Open3D parte de su primer punto. La
permutación cambia el inicio y conserva la idea de cobertura.

### Antes

El entrenamiento registraba `sampling=fps`, pero varias llamadas de evaluación no
pasaban ese valor. Como el valor predeterminado era `random`, un modelo entrenado con
FPS podía evaluarse silenciosamente con puntos al azar.

### Ahora

`src.eval` lee el sampling del `config.json` y lo propaga a embeddings de train,
validation, test, open-set y referencias. Para runs antiguas sin sampling, informa el
fallback a `random`.

### Beneficios

- La evaluación representa el entrenamiento real.
- Desaparece el cambio silencioso de FPS a random.
- Referencias y consultas ven el mismo tipo de puntos.
- Las métricas son más confiables.

## 7. Los tres tipos de sampling

| Sampling | Explicación sencilla |
|---|---|
| `random` | Elige puntos al azar; es rápido pero puede cubrir mal una zona. |
| `fps` | Busca puntos separados para cubrir globalmente la nube. |
| `fps_baya` | Reparte FPS dentro de regiones o grupos llamados bayas. |

La run evaluada usa `fps`, no `fps_baya`.

Además, `sample_n` ahora puede recibir un generador `rng` específico. Antes sólo
consumía el azar global de NumPy.

### Beneficios

- La evaluación controla el azar de cada nube.
- No necesita alterar variables globales.
- Entrenamiento y evaluación pueden compartir la función sin perder su comportamiento.

## 8. Augmentation de entrenamiento

Durante train se crean variaciones:

1. rotación alrededor del eje Z;
2. escala aproximada entre 0,8 y 1,25;
3. ruido pequeño;
4. eliminación de algunos puntos;
5. remuestreo hasta volver a 512.

Validation normal no usa esas transformaciones.

### Antes

Las operaciones ya existían y usaban el azar global.

### Ahora

`augment` puede recibir un `rng`. Si la evaluación pide augmentation, sus decisiones
salen de la seed propia de la nube y vista.

### Beneficios

- Una evaluación con augmentation puede repetirse.
- Cambiar el orden no cambia las demás nubes.
- Sampling y augmentation comparten una fuente controlada.

## 9. Entrenamiento por tripletas

Una tripleta tiene:

- **anchor:** una nube de una identidad;
- **positive:** otra nube de la misma identidad;
- **negative:** una nube de otra identidad.

La loss intenta que anchor-positive quede más cerca que anchor-negative, dejando un
margen. En esta run el margen es 0,5.

### Antes

Éste ya era el entrenamiento. En train se eligen positivos y negativos al azar.

### Ahora

No cambiaron las tripletas, la loss, el optimizador ni los pesos para esta comparación.

### Beneficio de no cambiarlo

Las diferencias observadas se deben a la evaluación, no a reentrenar otro modelo.

## 10. Qué hace el modelo

Recibe datos con forma:

```text
(tamaño de batch, 3 coordenadas, 512 puntos)
```

El CPN inspirado en PointNet:

1. alinea XYZ;
2. extrae características de los puntos;
3. alinea características intermedias;
4. resume la nube con max pooling;
5. usa capas densas;
6. devuelve el embedding.

Con `width=8`, el embedding tiene `64 × 8 = 512` números.

### Antes y ahora

La arquitectura, dimensión y pesos no cambiaron.

### Beneficio

La comparación aísla la forma de preparar, guardar y comparar embeddings.

## 11. Qué checkpoint se evalúa

La run llegó a 3000 épocas, por eso sus artefactos están bajo `ep3000/`. Sin embargo,
`src.eval` carga `model_best.pt`: el checkpoint con mejor loss de validation. Luego lo
copia como `ep3000/model.pt` para dejar la foto usada.

“Run de 3000 épocas” no garantiza que los pesos elegidos sean exactamente los de la
última época; son los del mejor checkpoint guardado.

### Antes

Ya se cargaba `model_best.pt`.

### Ahora

Se calcula su SHA-256 y se guarda en manifiestos.

### Beneficios

- Dos archivos con igual nombre y distintos pesos no se confunden.
- Si cambia un byte, el caché deja de ser compatible.
- Se puede demostrar qué modelo creó cada embedding.

## 12. Qué es un embedding

Es una lista de números que resume una nube. No es una clase ni una probabilidad. Es
como una coordenada en un mapa de 512 dimensiones: capturas de la misma identidad
deberían caer en barrios cercanos.

## 13. El problema del azar global

### Antes

A, B y C consumían una sola secuencia de azar. Si se cambiaba el orden, cada nube
recibía otros números y podía generar puntos y embeddings diferentes.

### Ahora

Cada vista recibe una seed derivada mediante SHA-256 de:

```text
(seed base, path completo, view_id)
```

### Beneficios

- La nube A obtiene la misma muestra aunque B cambie de posición.
- Batch 64 y batch 512 muestrean igual cada path.
- Todas las estrategias comparan exactamente los mismos embeddings.
- El resultado no depende del orden accidental.

## 14. Inferencia por batches

Un forward es una pasada por el modelo.

### Antes

Se preparaba una nube y se hacía un forward de tamaño 1. Para 40.026 nubes eran cerca
de 40.026 forwards.

### Ahora

Se agrupan nubes en batches configurables. Con batch 512:

```text
train:      ceil(40.026 / 512) = 79 forwards
validation: ceil(8.577 / 512)  = 17 forwards
```

### Beneficios

- Menos llamadas repetidas al modelo.
- Mejor uso de GPU.
- Batch ajustable a memoria.
- En la GTX 1050 de 3 GB, el pico fue de unos 548 MiB CUDA.

La lectura de PLY y FPS siguen costando CPU/disco; por eso no desaparece todo el
tiempo de ejecución.

## 15. Una sola inferencia por split

### Antes

Cada estrategia podía volver a generar todos los embeddings de validation.

### Ahora

Validation se embebe una vez y todas las estrategias reciben la misma lista. Lo mismo
se aplica a test y open-set.

### Beneficios

- Menos trabajo de GPU.
- Comparación justa sobre las mismas queries.
- Una diferencia refleja la referencia, no otra selección aleatoria de puntos.

## 16. El nuevo caché individual

Se crean archivos como:

```text
individual_embeddings_train.npz
individual_embeddings_val.npz
individual_embeddings_test.npz
```

Cada fila guarda:

| Campo | Significado |
|---|---|
| `path` | PLY original. |
| `label` | Identidad real. |
| `video` | Video de origen, si se identifica. |
| `capture_form` | Forma de captura disponible. |
| `view_id` | Número de vista. |
| `seed` | Seed exacta de esa vista. |
| `embedding` | Los 512 números del modelo. |

### Antes

Quedaban referencias finales, pero no todos los embeddings que las originaron.

### Ahora

Se guardan los embeddings individuales y las referencias se construyen desde ellos.

### Beneficios

- Se puede auditar una nube concreta.
- Se prueban prototipos nuevos sin volver a inferir 40.026 nubes.
- Se pueden analizar errores por video o captura.
- Todas las estrategias comparten la misma materia prima.

La corrida produjo:

| Caché | Matriz | Tamaño |
|---|---:|---:|
| Train | `(40026, 512)` | 66,15 MiB |
| Validation | `(8577, 512)` | 14,19 MiB |

## 17. El manifiesto del caché

Cada `.npz` tiene un `.manifest.json` que registra:

- versión del formato;
- path y SHA-256 del checkpoint;
- split y hash de sus paths;
- cantidad de muestras y puntos;
- sampling;
- vistas y agregación;
- normalización y augmentation;
- batch y seed.

### Antes

El nombre `reference_embeddings_centroid_all.npz` no decía qué modelo, split,
sampling o seed lo había producido. Podía reutilizarse sólo porque existía.

### Ahora

Se compara el manifiesto esperado con el guardado. También se comprueba que el `.npz`
tenga todos los campos y filas correctos.

### Beneficios

- No se mezclan modelos, splits o sampling.
- No se mezclan una, cuatro u ocho vistas.
- No se reutiliza un archivo incompleto.
- Un cambio relevante provoca regeneración automática.

## 18. Escritura segura

### Ahora

Caché y manifiesto se escriben primero con nombre temporal. Sólo al completar se
renombran al nombre final.

### Beneficio

Si el proceso se corta, no queda un archivo parcial que parezca válido.

## 19. Referencias por identidad

Los embeddings de train se agrupan por identidad y se resumen de varias maneras.

### `centroid_5`, `centroid_10`, `centroid_20`

Promedio reproducible de hasta 5, 10 o 20 embeddings.

**Beneficio:** referencias simples para medir cuántas muestras hacen falta.

### `centroid_all`

Promedio de todos los embeddings de la identidad.

**Beneficio:** usa toda la información y reduce el efecto de una captura.

### `centroid_l2norm_5`

Normaliza cinco embeddings, promedia y normaliza otra vez.

**Beneficio:** pensado para cosine, donde importa la dirección. No necesariamente
sirve con L1: en esta prueba dio 0,4940 con cosine y 0,0288 con L1.

### `multiprototype_k5`

K-means guarda hasta cinco centros por identidad.

**Beneficio:** representa diferentes poses o formas de captura mejor que un único
promedio. Fue el mejor: accuracy 0,6210 con cosine.

## 20. Nuevas referencias robustas

### `median_all`

Toma el valor central en cada coordenada del embedding.

**Por qué y beneficio:** los valores extremos afectan menos. Accuracy L1: 0,5737.

### `trimmed_mean_05`

Quita por coordenada el 5% menor y mayor, y promedia el resto.

**Por qué y beneficio:** conserva el promedio pero limita outliers. Accuracy: 0,5743.

### `trimmed_mean_10`

Quita el 10% de cada extremo.

**Beneficio:** más protección frente a valores raros. Accuracy: 0,5772; fue el mejor
prototipo único nuevo.

## 21. Manifiesto de referencias

`reference_embeddings.manifest.json` registra el caché de train que originó las
referencias y la lista exacta de estrategias.

### Antes

Los `.npz` existentes podían aceptarse aunque vinieran de otro sampling.

### Ahora

Sólo se reutilizan si están todas las estrategias, coincide el manifiesto y el caché
de train sigue siendo compatible.

### Beneficio

Cada referencia queda unida a los embeddings que la produjeron.

## 22. Vistas múltiples

Ahora se pueden generar 1, 4 u 8 muestras reproducibles de una nube. Cada `view_id`
produce otra seed estable. Sus embeddings se juntan con promedio o mediana por
coordenada.

### Antes

Una nube tenía una sola muestra normal de evaluación.

### Ahora

Puede tener varias. El PLY se lee y normaliza una sola vez antes de crearlas.

### Beneficios

- Menor dependencia de una selección FPS.
- Permite medir variación por nube.
- Referencias y queries usan la misma regla.
- Las variantes se guardan separadas para no pisarse.

La corrida principal usó `embedding_views=1`; multivista está disponible pero no se
mezcló en la comparación de Fase 0.

## 23. Clasificación

Para cada query:

1. se compara su embedding con cada identidad;
2. se ordenan identidades de mejor a peor;
3. la primera es la predicción.

Con múltiples prototipos se usa el mejor centro de cada clase.

| Método | Idea | Mejor valor |
|---|---|---|
| L1 | Suma de diferencias absolutas. | Más bajo. |
| L2 | Distancia recta. | Más bajo. |
| Linf | Mayor diferencia individual. | Más bajo. |
| Cosine | Dirección de los vectores. | Más alto. |
| Dot product | Producto entre vectores. | Más alto. |
| Inversa L1/L2 | Distancia convertida en similitud. | Más alto. |

### Antes

Los métodos ya existían.

### Ahora

Usan embeddings precalculados y registran su propio tiempo.

### Beneficios

- Se separa costo de inferencia y clasificación.
- Se compara calidad y velocidad.
- Es visible que multiprototipo mejora, pero cuesta más CPU.

Ejemplo: `centroid_all` L1 tardó cerca de 1,02 ms/query; `multiprototype_k5` L1,
5,18 ms/query.

## 24. Métricas en palabras simples

- **Accuracy:** porcentaje donde la identidad correcta quedó primera.
- **Top-5:** porcentaje donde quedó entre las cinco primeras.
- **Top-10:** porcentaje donde quedó entre las diez primeras.
- **MRR:** premia que la respuesta correcta aparezca muy arriba.
- **Mean rank:** posición promedio; más bajo es mejor.
- **Median rank:** posición central, menos sensible a casos extremos.

## 25. Reporte y reanudación segura

### Antes

Se podía continuar un reporte existente sin comprobar por completo checkpoint, split,
sampling y preprocesamiento.

### Ahora

`evaluation_report.json` guarda un manifiesto con checkpoint, puntos, sampling, seed,
augmentation, batch, vistas y hashes. Sólo reanuda si coincide exactamente.

### Beneficios

- No mezcla resultados viejos y nuevos.
- Un cambio relevante obliga a evaluar coherentemente.
- Cada número queda trazable.

## 26. Tiempo y memoria

### Antes

El reporte se centraba en accuracy y ranking.

### Ahora

También guarda tiempos de embeddings, memoria del proceso, memoria CUDA, tamaño de
cachés y latencia por query/método.

### Beneficio

Una estrategia puede elegirse por calidad y también por costo operativo.

## 27. Cambios por archivo

### `src/data/dataset.py`

`sample_n` y `augment` aceptan `rng`.

**Beneficio:** azar reproducible por nube.

### `src/evaluation/embed.py`

Agrega seed por path/vista, preprocesamiento común, batches, multivista, agregación y
medición de memoria.

**Beneficio:** simetría, reproducibilidad y mejor GPU.

### `src/evaluation/embedding_cache.py` — nuevo

Crea, carga y valida cachés, hashes, manifiestos, metadatos y escritura segura.

**Beneficio:** reutilización confiable y trazabilidad.

### `src/evaluation/ref_strategies.py`

Usa el caché común, agrega estrategias robustas y manifiesto de referencias.

**Beneficio:** comparación justa y más alternativas.

### `src/evaluation/report.py`

Acepta embeddings precalculados y mide latencia/memoria.

**Beneficio:** no repite inferencia.

### `src/evaluation/open_set.py`

Reutiliza embeddings y recibe la configuración completa.

**Beneficio:** open-set coherente con el resto.

### `src/eval.py`

Lee sampling real, administra cachés/manifiestos, separa multivista, evita reanudar
reportes incompatibles y guarda runtime.

**Beneficio:** coordina todo el flujo sin inconsistencias silenciosas.

### `src/evaluation/runtime_stats.py` — nuevo

Mide memoria de proceso y CUDA.

### Tests nuevos

Cubren seeds, orden, preprocesamiento simétrico, campos del caché, invalidación,
multivista, métricas fijas, mediana y medias recortadas.

**Beneficio:** protege estos cambios de regresiones.

## 28. Qué no cambió

- No se reentrenó el modelo.
- No cambió CPN/PointNet.
- No cambió la dimensión 512.
- No cambió la loss ni el margen 0,5.
- No cambió el split guardado.
- No cambió el checkpoint evaluado.
- No cambió la fórmula de normalización.
- No se usó test para elegir estrategia.
- No se usó multivista en la corrida principal.

Cambió la confiabilidad, eficiencia, trazabilidad y variedad de la evaluación.

## 29. Tabla general antes/después

| Tema | Antes | Después | Beneficio |
|---|---|---|---|
| Sampling | FPS podía evaluarse con random. | Usa `config.json`. | Coherencia con train. |
| Azar | Dependía del orden global. | Seed por path/vista. | Repetibilidad. |
| Preprocesamiento | Caminos separados. | Ruta compartida. | Simetría. |
| GPU | Un forward por nube. | Batches. | Menos overhead. |
| Estrategias | Repetían queries. | Un caché por split. | Menos trabajo y comparación justa. |
| Archivos | Se confiaba en nombres. | Manifiestos y hashes. | No mezcla configuraciones. |
| Auditoría | Sólo prototipos. | Embeddings individuales. | Rastreo por muestra. |
| Prototipos | Media y K-means. | También mediana/recortadas. | Resistencia a outliers. |
| Vistas | Una. | Opción 1/4/8. | Menos dependencia de un FPS. |
| Reporte | Calidad. | Calidad, tiempo y memoria. | Decisión completa. |
| Reanudación | Podía mezclar resultados. | Exige manifiesto idéntico. | Consistencia. |

## 30. Resultado real de la run FPS

Comparación L1 en validation:

| Estrategia | Antes | Ahora | Cambio |
|---|---:|---:|---:|
| `centroid_5` | 0,4560 | 0,5147 | +5,88 puntos |
| `centroid_10` | 0,4821 | 0,5367 | +5,46 puntos |
| `centroid_20` | 0,4947 | 0,5440 | +4,93 puntos |
| `centroid_all` | 0,4995 | 0,5693 | +6,98 puntos |
| `centroid_l2norm_5` | 0,0309 | 0,0288 | -0,21 puntos |
| `multiprototype_k5` | 0,4961 | 0,6180 | +12,20 puntos |

Mejor resultado general:

```text
multiprototype_k5 + cosine
accuracy: 0,6210
top-5:    0,9746
MRR:      0,7707
```

Los pesos no “aprendieron más”: son los mismos. Ahora la evaluación usa FPS
correctamente y compara referencias/queries coherentes.

## 31. Coste real

| Etapa | Tiempo |
|---|---:|
| Embeddings train + referencias | 33m 13s |
| Embeddings validation | 7m 11s |
| Clasificación | 11m 19s |
| Total registrado | 51m 43s |

El primer cálculo sigue siendo costoso por lectura, FPS e inferencia. La ganancia del
caché se nota en comparaciones posteriores compatibles, que no deberían repetirlo.

## 32. Advertencia del dataset

Open3D detectó un posible PLY truncado:

```text
dataset/034/034_VID_20230322_135246_nube_8.ply
```

Llegó a un fin inesperado cerca del vértice 18.428. Recuperó suficientes puntos y la
muestra no fue omitida, pero conviene regenerarla.

## 33. Qué se puede concluir

### Sí

- La evaluación nueva usa FPS explícito.
- Referencias y queries se preparan igual.
- Cada path tiene una muestra reproducible.
- Los embeddings quedaron guardados y validados.
- Cinco de seis comparaciones L1 mejoraron.
- Multiprototipo representó mejor la variación en validation.

### Todavía no

- No es el resultado final de test.
- No demuestra que multivista sea mejor; se usó una vista.
- No todo aumento puede atribuirse sólo a FPS; se corrigieron varias inconsistencias.
- El modelo no fue reentrenado.
- Multiprototipo no siempre será la opción operativa ideal: también es más lento.

## 34. Explicación final sin términos técnicos

Antes, el modelo aprendía mirando nubes preparadas de una forma, pero al tomarle examen
el programa podía prepararlas de otra forma sin avisar. Además, repetía trabajo y no
anotaba con suficiente detalle cómo había creado resultados viejos.

Ahora el examen respeta la preparación real, cada archivo recibe una selección estable,
el modelo procesa muchas nubes juntas, los resultados costosos se guardan y cada uno
viene con una ficha que dice exactamente de dónde salió.

Los cuatro beneficios principales son:

1. **confianza:** los números representan la configuración real;
2. **repetibilidad:** la misma nube produce la misma entrada;
3. **eficiencia:** los embeddings se calculan una vez y se reutilizan;
4. **análisis:** se prueban referencias nuevas sobre exactamente los mismos datos.

