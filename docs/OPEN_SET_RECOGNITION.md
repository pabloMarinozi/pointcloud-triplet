# Open-set recognition

## 1. Objetivo

El sistema original realizaba clasificación de conjunto cerrado (*closed set*):
dada una nube de puntos, siempre seleccionaba la clase conocida cuyo prototipo
estuviera más cerca, incluso cuando la muestra pertenecía a una identidad nunca
vista durante el entrenamiento.

La implementación de *open-set recognition* agrega la posibilidad de responder:

> Esta muestra no se parece lo suficiente a ninguna clase conocida.

En ese caso, la predicción devuelta es `__unknown__`.

El cambio tiene dos componentes:

1. reservar identidades completas fuera del entrenamiento;
2. aprender un umbral de rechazo usando datos de validación y aplicarlo después
   sobre un test independiente.

## 2. Por qué se implementó de esta manera

TripletNet aprende un espacio de embeddings, pero no contiene una salida explícita
para la clase “desconocida”. La *triplet loss* solo intenta que las muestras de la
misma identidad estén cerca y las de identidades diferentes estén separadas.

Por eso, excluir clases del entrenamiento no es suficiente: el clasificador por
prototipos seguiría escogiendo la clase conocida más cercana. Se necesita además
un criterio de rechazo. En esta implementación, dicho criterio es un umbral sobre
la distancia o similitud entre el embedding consultado y su mejor prototipo
conocido.

No se añadió una clase artificial `unknown` al entrenamiento porque “desconocido”
no representa una única identidad. Además, entrenar con las mismas identidades que
luego se pretende presentar como desconocidas produciría fuga de información.

## 3. Protocolo sin fuga de información

El split se realiza por identidad, no solamente por muestra.

```text
Todas las clases
├── Clases conocidas
│   ├── train: entrenamiento con triplet loss y creación de prototipos
│   ├── val: calibración del comportamiento de muestras conocidas
│   └── test: evaluación final de muestras conocidas
└── Clases reservadas (unknown)
    ├── unknown calibration: calibración del umbral
    └── unknown test: evaluación final de desconocidos
```

Las identidades de `unknown calibration` y `unknown test` también son disjuntas.
Esto evita ajustar el umbral con otras muestras de las mismas clases desconocidas
que aparecen en el test final.

La selección es reproducible:

- las clases se ordenan;
- se seleccionan con `random.Random(seed)`;
- la misma configuración y el mismo `--seed` generan el mismo protocolo.

Las clases conocidas conservan el split estratificado por clase definido mediante
`--val_size` y `--test_size`. Ninguna muestra de una clase reservada se incorpora al
dataset de tripletas, a los dataloaders ni a los prototipos de referencia.

## 4. Entrenamiento

Se añadieron dos argumentos a `src.train`:

- `--open_set_classes N`: número total de identidades que se reservan como
  desconocidas;
- `--open_set_val_size R`: fracción de esas identidades destinada a calibrar el
  umbral. El valor por defecto es `0.5`.

Ejemplo con 20 clases reservadas:

```bash
python -m src.train \
  --data_dir ./dataset_ply \
  --run_name openset_20_seed42 \
  --open_set_classes 20 \
  --open_set_val_size 0.5 \
  --n_points 512 \
  --width 8 \
  --epochs 30 \
  --seed 42
```

Con esta configuración, aproximadamente 10 identidades desconocidas se usan para
calibración y las otras 10 para test. El redondeo garantiza al menos una identidad
en cada grupo.

La arquitectura y la función de pérdida no cambian: TripletNet continúa entrenándose
exclusivamente con las clases conocidas. Por lo tanto, este cambio implementa el
protocolo open-set y el mecanismo de rechazo, no una nueva función de pérdida
especializada en open-set.

### Restricciones validadas

- `--open_set_classes` debe ser `0` —funcionamiento tradicional— o al menos `2`;
- deben quedar al menos dos clases conocidas para formar negativos;
- `val_size + test_size` debe ser menor que uno;
- cada clase conocida debe conservar al menos dos muestras en train y dos en val,
  porque ambos datasets forman tripletas;
- `open_set_val_size` debe estar entre cero y uno cuando open-set está habilitado.

## 5. Archivos generados

Además de los splits tradicionales, el run genera:

```text
runs/<run>/splits/
├── train_paths.json
├── val_paths.json
├── test_paths.json
├── open_set_val_paths.json
├── open_set_test_paths.json
└── open_set_classes.json
```

`open_set_classes.json` registra:

- `known`: clases que el modelo puede reconocer;
- `unknown`: todas las clases reservadas;
- `unknown_calibration`: identidades usadas para seleccionar el umbral;
- `unknown_test`: identidades desconocidas usadas solo en la evaluación final.

La configuración también guarda `open_set_classes` y `open_set_val_size` en
`config.json`.

## 6. Cómo funciona la evaluación

La evaluación open-set se activa con `--open_set`:

```bash
python -m src.eval \
  --data_dir ./dataset_ply \
  --run openset_20_seed42 \
  --open_set \
  --export_csv \
  --seed 42
```

Para cada estrategia de prototipos y para cada método de comparación, el proceso es:

1. cargar únicamente los prototipos creados a partir del train conocido;
2. obtener embeddings de `val_paths.json` y `open_set_val_paths.json`;
3. calcular el mejor score contra todas las clases conocidas;
4. convertir el score en un valor de novedad donde un valor mayor siempre significa
   “más probablemente desconocido”;
5. seleccionar el umbral que maximiza la *balanced accuracy* de calibración;
6. congelar ese umbral;
7. evaluarlo sobre `test_paths.json` y `open_set_test_paths.json`.

### Score de novedad

Para distancias como L1, L2 o Linf:

```text
novelty_score = mejor_distancia
```

Una distancia grande indica que la muestra está lejos incluso de su clase conocida
más próxima.

Para similitudes como cosine similarity o dot product:

```text
novelty_score = -mejor_similitud
```

Se cambia el signo para mantener una única regla de decisión:

```text
si novelty_score > threshold:
    predicción = __unknown__
si no:
    predicción = clase_con_mejor_score
```

El umbral se busca entre los scores observados en calibración. Se usa *balanced
accuracy* para que el resultado no quede dominado por el grupo que tenga más
muestras.

## 7. Métricas reportadas

El archivo `open_set_report.json` contiene un resultado por estrategia de
referencia y método de comparación:

- `threshold`: umbral calibrado sobre el score de novedad;
- `balanced_accuracy`: promedio entre la tasa de aceptación de conocidos y la tasa
  de rechazo de desconocidos;
- `unknown_recall`: proporción de desconocidos correctamente rechazados;
- `unknown_precision`: proporción de rechazos que realmente eran desconocidos;
- `unknown_f1`: media armónica entre precision y recall de desconocidos;
- `known_accept_rate`: proporción de conocidos que no fueron rechazados;
- `open_set_accuracy`: exactitud completa; un conocido solo cuenta como correcto si
  es aceptado y su identidad es correcta, mientras que un desconocido cuenta como
  correcto si es rechazado;
- `auroc`: capacidad de separar conocidos y desconocidos independientemente de un
  umbral específico;
- `n_known_test` y `n_unknown_test`: cantidad de muestras usadas en cada grupo.

Cuando se usa `--export_csv`, se escribe una fila por muestra con su etiqueta real,
predicción, score original, score de novedad, umbral y decisión de rechazo:

```text
runs/<run>/ep<N>/evaluation_open_set/<strategy>/
└── open_set_predictions_<method>.csv
```

El resumen queda en:

```text
runs/<run>/ep<N>/open_set_report.json
```

## 8. Interpretación de resultados

No conviene seleccionar un modelo mirando únicamente `unknown_recall`. Un umbral
muy estricto puede rechazar casi todos los desconocidos y también muchas muestras
conocidas. Las métricas principales para comparar configuraciones son:

1. `balanced_accuracy`, para equilibrar aceptación y rechazo;
2. `auroc`, para medir la separabilidad global;
3. `open_set_accuracy`, para incluir también la clasificación correcta de la
   identidad conocida;
4. el par `unknown_recall` / `known_accept_rate`, para entender el compromiso
   operativo.

Si el coste de aceptar un desconocido es especialmente alto, el umbral óptimo no
necesariamente debe ser el que maximiza balanced accuracy. En ese escenario se
puede extender la calibración para fijar una tasa máxima de falsos aceptados o usar
una métrica ponderada según el coste del caso de uso.

## 9. Pipeline completo

El protocolo open-set se desarrolla en dos comandos y momentos distintos. La
división de datos se realiza al crear el pipeline de entrenamiento, mientras que
la calibración se realiza posteriormente durante `src.eval`. La calibración no es
parte del entrenamiento de TripletNet.

```text
src.train
   │
   ├── 1. descubrir/cargar todas las muestras
   ├── 2. reservar identidades unknown
   ├── 3. dividir identidades conocidas en train/val/test
   ├── 4. guardar todos los splits
   └── 5. entrenar TripletNet únicamente con train conocido
              │
              ▼
src.eval --open_set
   │
   ├── 6. crear/cargar prototipos desde train conocido
   ├── 7. calibrar un umbral con val conocido + unknown calibration
   ├── 8. congelar el umbral
   └── 9. evaluar con test conocido + unknown test
```

### 9.1. Descubrimiento de las muestras

`src.train` comienza localizando todos los archivos `.ply` bajo `--data_dir`. El
nombre de la carpeta contenedora se utiliza como identidad.

En modo eager, las nubes se leen y se muestrean antes de construir el pipeline.
En modo lazy, en esta etapa solo se descubren las rutas y cada nube se lee después
bajo demanda. Esta diferencia afecta el uso de memoria, pero no altera la
selección de identidades ni los splits.

### 9.2. División inicial por identidad

La división ocurre dentro del constructor de `TripletTrainingPipeline` o
`LazyTripletTrainingPipeline`, antes de crear los datasets, dataloaders y el
modelo. Ambos pipelines llaman a la misma función `split_known_and_unknown`.

Primero se obtiene la lista ordenada de identidades disponibles. A continuación,
`random.Random(seed)` selecciona `open_set_classes` identidades completas y las
reserva como desconocidas. No se seleccionan archivos aislados: si una identidad
es desconocida, todas sus muestras quedan fuera del entrenamiento.

Las identidades reservadas se mezclan de forma reproducible y se dividen según
`open_set_val_size`:

- `unknown calibration`: identidades utilizadas más adelante para elegir el
  umbral;
- `unknown test`: identidades diferentes, utilizadas únicamente en la evaluación
  final.

El redondeo garantiza, cuando open-set está habilitado, que ambos grupos contengan
al menos una identidad. Por eso se requieren al menos dos clases desconocidas.

Las clases restantes pasan a ser conocidas. Cada una se divide por muestras en
train, val y test usando `val_size`, `test_size` y el mismo `seed`. Como resultado
se obtienen cinco conjuntos:

| Conjunto | Identidades | Uso |
| --- | --- | --- |
| `train` | Conocidas | Triplet loss y creación de prototipos |
| `val` | Conocidas | Medir la aceptación de conocidos durante la calibración |
| `test` | Conocidas | Evaluación final de clasificación y aceptación |
| `unknown calibration` | Desconocidas reservadas | Medir el rechazo durante la calibración |
| `unknown test` | Otras desconocidas reservadas | Evaluación final de rechazo |

Los cinco conjuntos y el inventario de clases se guardan inmediatamente dentro
de `runs/<run>/splits/`. Esto permite que la evaluación reproduzca exactamente el
protocolo definido antes del entrenamiento.

### 9.3. Entrenamiento con clases conocidas

Después de realizar y guardar la división, el pipeline crea los datasets y
dataloaders exclusivamente con `train` y `val` conocidos. TripletNet se optimiza
con triplet loss de la misma forma que en closed-set.

Las muestras de `unknown calibration` y `unknown test` no se incorporan a los
datasets, no forman tripletas, no generan gradientes y no intervienen en la
selección del mejor checkpoint. En esta etapa tampoco se calcula ningún umbral de
rechazo.

### 9.4. Construcción de los prototipos conocidos

Al ejecutar `src.eval --open_set`, se carga el modelo entrenado. Para cada
estrategia de referencia, los prototipos existentes se cargan desde el directorio
del run. Si todavía no existen, se generan utilizando únicamente
`train_paths.json`, que contiene muestras de identidades conocidas.

Cada prototipo representa una clase que el sistema está autorizado a reconocer.
Las identidades desconocidas nunca reciben un prototipo ni se incorporan como una
clase adicional.

### 9.5. Calibración del umbral

La calibración comienza después del entrenamiento y se repite independientemente
para cada combinación de estrategia de prototipos y método de comparación.

Se cargan conjuntamente:

- `val_paths.json`, con muestras conocidas que deberían ser aceptadas;
- `open_set_val_paths.json`, con muestras desconocidas que deberían ser
  rechazadas.

Cada muestra se convierte en un embedding y se compara contra todos los
prototipos conocidos. Se conserva el score de la mejor clase candidata y se
transforma en `novelty_score`, donde un valor mayor siempre representa mayor
probabilidad de que la muestra sea desconocida.

Con los scores de ambos grupos se prueban posibles umbrales. Para cada candidato
se aplica:

```text
predicted_unknown = novelty_score > threshold
```

El umbral seleccionado es el que maximiza la *balanced accuracy* de calibración:

```text
balanced_accuracy = (known_accept_rate + unknown_reject_rate) / 2
```

De este modo, la selección considera por igual la capacidad de aceptar conocidos
y de rechazar desconocidos, aunque los dos conjuntos tengan cantidades distintas
de muestras. Esta fase no actualiza pesos, embeddings ni prototipos: solamente
elige un número de corte.

### 9.6. Congelado y evaluación final

Una vez elegido, el umbral queda congelado para esa estrategia y método. No se
vuelve a ajustar mirando el test.

La evaluación final carga:

- `test_paths.json`, con muestras conocidas que no participaron en la
  calibración;
- `open_set_test_paths.json`, con identidades desconocidas que tampoco
  participaron en la calibración.

Para cada muestra se obtiene primero la clase conocida más cercana. Si su score de
novedad supera el umbral congelado, esa clase se descarta y la predicción final es
`__unknown__`. Si no lo supera, se conserva la identidad conocida propuesta.

Una muestra conocida solo cuenta como correcta cuando no es rechazada y además su
identidad es la correcta. Una muestra desconocida cuenta como correcta cuando es
rechazada. Los resultados se escriben en `open_set_report.json` y, con
`--export_csv`, también se guarda la decisión individual de cada muestra.

### 9.7. Separación de responsabilidades

En resumen, cada dato tiene una única función dentro del protocolo:

- los pesos se aprenden con `train` conocido;
- los prototipos se construyen con `train` conocido;
- el umbral se elige con `val` conocido y `unknown calibration`;
- las métricas finales se calculan con `test` conocido y `unknown test`.

Esta separación evita ajustar el modelo o el umbral con los mismos datos usados
para reportar el resultado final.

## 10. MODIFICACIONES

La implementación incorpora cuatro archivos nuevos y modifica cuatro archivos
existentes. No se modifica la arquitectura de `TripletNet` ni la función de
pérdida: los cambios se concentran en la creación de los splits, la integración
con los pipelines y el protocolo de evaluación.

### 10.1. Archivos nuevos

#### `src/data/splits.py`

Centraliza la función `split_known_and_unknown`, utilizada tanto por el pipeline
eager como por el lazy. Sus responsabilidades son:

- validar los tamaños de `val`, `test` y calibración open-set;
- seleccionar de forma determinística las identidades desconocidas usando el
  `seed`;
- dividir las identidades desconocidas en dos grupos disjuntos: calibración y
  test;
- realizar el split train/val/test solamente sobre las identidades conocidas;
- impedir que una identidad reservada aparezca en los datasets de tripletas;
- comprobar que queden suficientes clases y muestras conocidas para formar
  tripletas;
- devolver los cinco conjuntos de muestras y las listas de clases conocidas y
  desconocidas.

Con `open_set_classes=0`, la misma función produce únicamente los splits
tradicionales y devuelve vacíos los dos conjuntos desconocidos.

#### `src/evaluation/open_set.py`

Implementa el núcleo de la evaluación open-set:

- genera los embeddings de los conjuntos de calibración y test;
- obtiene, para cada método, el mejor score contra los prototipos conocidos;
- convierte distancias y similitudes a una escala común de novedad;
- busca el umbral que maximiza la *balanced accuracy* en calibración;
- congela el umbral y aplica la regla de rechazo sobre el test independiente;
- devuelve `__unknown__` cuando el score de novedad supera el umbral;
- calcula las métricas open-set descritas en la sección anterior;
- exporta opcionalmente un CSV por método con scores, umbral, decisión y
  predicción de cada muestra.

La función principal es `evaluate_open_set`; `_best_threshold` contiene la
selección del umbral y `_novelty_score` normaliza el sentido de los distintos
métodos de comparación.

#### `tests/test_open_set.py`

Agrega pruebas unitarias para verificar:

- que las clases desconocidas no aparezcan en train, val o test conocidos;
- que las identidades desconocidas de calibración y test sean disjuntas;
- que la selección de clases sea reproducible para un mismo `seed`;
- que la calibración encuentre un umbral entre scores conocidos y desconocidos;
- que una muestra distante sea rechazada como desconocida y produzca las métricas
  esperadas.

#### `docs/OPEN_SET_RECOGNITION.md`

Documenta el objetivo, las decisiones de diseño, el protocolo sin fuga de
información, los argumentos de línea de comandos, los archivos generados, el
procedimiento de calibración, las métricas y la compatibilidad con el modo
tradicional.

### 10.2. Archivos modificados

#### `src/train.py`

Extiende el CLI de entrenamiento con:

- `--open_set_classes`, desactivado por defecto con valor `0`;
- `--open_set_val_size`, con valor por defecto `0.5`.

Ambos valores se pasan al pipeline seleccionado, por lo que el comportamiento es
el mismo en los modos eager y lazy.

#### `src/pipeline/trainer.py`

Reemplaza la lógica duplicada del split tradicional por
`split_known_and_unknown` en `TripletTrainingPipeline` y
`LazyTripletTrainingPipeline`. Además:

- recibe la configuración open-set desde `src.train`;
- conserva en train, val y test solamente las identidades conocidas;
- mantiene fuera de los datasets y dataloaders las muestras desconocidas;
- guarda `open_set_classes` y `open_set_val_size` en `config.json`;
- persiste `open_set_val_paths.json`, `open_set_test_paths.json` y
  `open_set_classes.json` cuando open-set está habilitado;
- registra en el log la cantidad de clases conocidas y la distribución de
  muestras desconocidas entre calibración y test;
- conserva el split tradicional cuando `open_set_classes=0`.

#### `src/evaluation/loader.py`

Amplía `RunInfo` con las rutas de los tres artefactos open-set. De este modo,
`src.eval` puede localizar los splits de calibración, test y el inventario de
clases sin construir rutas manualmente ni depender del directorio actual.

#### `src/eval.py`

Extiende el CLI con `--open_set` e integra el protocolo completo de evaluación:

- comprueba que el run contenga todos los artefactos open-set requeridos;
- carga conocidos de `val_paths.json` y `test_paths.json`;
- carga desconocidos de `open_set_val_paths.json` y
  `open_set_test_paths.json`;
- verifica que ninguno de los cuatro conjuntos esté vacío después de mapear los
  paths contra el dataset actual;
- carga o genera los prototipos exclusivamente desde el train conocido;
- ejecuta `evaluate_open_set` para cada estrategia de referencia y método de
  comparación;
- muestra por consola el umbral y las métricas principales;
- guarda los resultados en `open_set_report.json`;
- crea los CSV bajo `evaluation_open_set/<strategy>/` cuando se usa
  `--export_csv`;
- omite con un mensaje explicativo los runs antiguos que no tienen splits
  open-set;
- mantiene intacto el flujo de evaluación closed-set cuando no se especifica
  `--open_set`.

## 11. Compatibilidad

El comportamiento anterior se conserva cuando no se especifica
`--open_set_classes`, ya que su valor por defecto es cero. Del mismo modo, la
evaluación tradicional sigue disponible sin `--open_set`.
