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

## 9. Código involucrado

- `src/data/splits.py`: selección determinística y splits por identidad;
- `src/pipeline/trainer.py`: integración de los splits en los pipelines eager y lazy;
- `src/train.py`: argumentos del entrenamiento;
- `src/evaluation/open_set.py`: calibración, rechazo, métricas y CSV;
- `src/eval.py`: carga de splits y ejecución de la evaluación open-set;
- `tests/test_open_set.py`: pruebas de determinismo, ausencia de fuga y rechazo.

## 10. Compatibilidad

El comportamiento anterior se conserva cuando no se especifica
`--open_set_classes`, ya que su valor por defecto es cero. Del mismo modo, la
evaluación tradicional sigue disponible sin `--open_set`.
