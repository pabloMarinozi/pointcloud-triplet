````markdown
# Point Cloud Triplet Learning (CPN / PointNet-style) for 3D Identity Experiments

Este repositorio implementa un pipeline reproducible de **metric learning** para **nubes de puntos 3D** densas, con el objetivo de aprender un **embedding** (representación vectorial) que preserve identidad: muestras de la misma “clase/identidad” quedan cerca en el espacio embebido, y muestras de clases distintas quedan lejos.

El caso de uso principal de este proyecto es **identidad por geometría** en **racimos de uva** (3D point clouds densas), pero la implementación es **general** y puede aplicarse a cualquier dataset con nubes de puntos 3D etiquetadas por clase (por ejemplo: objetos, caras 3D, piezas industriales, etc.).

---

## Origen del proyecto

El código se basa en una arquitectura estilo **PointNet** con módulos **T-Net** (transformaciones aprendidas) para alinear la nube de entrada y sus features, y entrena el modelo mediante **Triplet Loss** para verificación por identidad.

Este repositorio adapta ese enfoque para:
- correr localmente
- y permitir experimentación sistemática variando hiperparámetros (`width`, `n_points`, etc.).

---


## Arquitectura de la red (CPN + Triplet Network)

Este proyecto implementa una arquitectura tipo **PointNet** con transformaciones aprendidas, entrenada por **Triplet Loss**.

### 1) CPN (Cloud Processing Network)

[CPN.png](https://postimg.cc/ftdFyMpF)

El backbone **CPN** recibe una nube de puntos `X ∈ R^{3×N}` y produce un embedding `z ∈ R^{D}`:

1. **Input T-Net (3×3)**
   Aprende una matriz `A ∈ R^{3×3}` para alinear la nube de entrada antes de extraer features.

2. **Forward Network-1 (Conv1D 1×1)**
   Bloques `Conv1D + BatchNorm + ReLU` para obtener features locales por punto.

3. **Feature T-Net (64×64)**
   Aprende una matriz `A ∈ R^{64×64}` para alinear el espacio de features (robustez geométrica).

4. **Forward Network-2 (Conv1D 1×1 más profunda)**
   Extrae features de alto nivel.

5. **Max Pooling global**
   Agregación invariante a la permutación → vector global.

6. **MLP final (FC + Dropout)**
   Produce el embedding final (paper-like: **4096 dimensiones**) con activación **Sigmoid** al final.

> Nota: Con `width=64`, este repositorio produce embeddings de dimensión `D = 64 * width = 4096`.

### 2) Triplet Network

Se ejecuta el backbone CPN en paralelo sobre:

* `anchor` (misma clase que positive),
* `positive` (misma clase que anchor),
* `negative` (clase distinta)

Y se optimiza:

[
\mathcal{L} = \max(0, ||z_a - z_p||^2 - ||z_a - z_n||^2 + m)
]

donde `m` es el margen (paper-like: **0.5**).
---

## Usos y extensiones

Este repositorio está pensado para experimentos de identidad por geometría.
Con cambios mínimos se puede extender a:

* Verificación por ROC/FAR (paper-style)
* Hard-negative mining
* Métricas adicionales (Top-k, confusion matrix)
* Normalización L2 del embedding (útil para cosine distance)

---

## Referencia (paper base)

Este repositorio se inspira en el enfoque propuesto en el siguiente trabajo:

* DOI: [https://doi.org/10.1007/s11042-020-10160-9](https://doi.org/10.1007/s11042-020-10160-9)

```


## Requisitos

- **Python 3.10+**
- **GPU NVIDIA** (opcional pero recomendado)
  - Probado con **RTX 3050**
- Drivers NVIDIA instalados correctamente (`nvidia-smi` debe funcionar)

---

## Instalación

### 1) Crear entorno virtual

#### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 3) Instalar PyTorch con CUDA (recomendado)

Ejemplo (PyTorch CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verificación GPU:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

> Nota: PyTorch incluye su propio runtime CUDA. No es necesario instalar el CUDA Toolkit completo para entrenar.

### 4) Instalar dependencias restantes

```bash
pip install -r requirements.txt
```

---

## Dataset esperado

El repositorio asume que la **clase/identidad** de cada nube está dada por el **nombre de la carpeta contenedora** del archivo `.ply`.

Ejemplo:

```
dataset_ply/
├─ cluster_001/
│  ├─ sample_0001.ply
│  ├─ sample_0002.ply
├─ cluster_002/
│  ├─ sample_0001.ply
│  ├─ sample_0002.ply
...
```

* Cada carpeta corresponde a una **identidad/clase**
* Debe haber **mínimo 2 muestras por clase** para poder formar tripletas (anchor/positive/negative)

---

## Entrenamiento

El comando principal de entrenamiento es:

```bash
python -m src.train --data_dir "./dataset_ply" --n_points 2048 --width 64 --epochs 30 --batch_size 16
```

### Argumentos principales

* `--data_dir`: carpeta raíz del dataset (recursivo)
* `--n_points`: puntos sampleados por nube (paper-like: **2048**)
* `--width`: escala del modelo (paper-like: **64**)
* `--epochs`: épocas de entrenamiento
* `--batch_size`: tamaño de batch
* `--margin`: margen de Triplet Loss (paper-like: **0.5**)

Ejemplo paper-like:

```bash
python -m src.train --data_dir "./dataset_ply" --n_points 2048 --width 64 --epochs 200 --batch_size 32 --lr 1e-4 --margin 0.5
```

> Nota: el paper original usa configuraciones más largas (ej. 500 épocas). En datasets distintos, el tiempo necesario puede variar significativamente.

---

## Salidas del entrenamiento

Cada ejecución crea una carpeta nueva dentro de `runs/`:

```
runs/run_YYYY-MM-DD_HH-MM-SS/
├─ config.json
├─ training.log
├─ metrics.csv
├─ model_best.pt
├─ reference_embeddings_train.npz
├─ reference_paths_train.json
└─ splits/
   ├─ train_paths.json
   └─ val_paths.json
```

* `model_best.pt`: checkpoint con menor loss de validación
* `reference_embeddings_train.npz`: embedding promedio por clase (prototipo)
* `splits/*.json`: paths utilizados para reproducir validación

---

## Evaluación

La evaluación actual implementa un enfoque práctico de **clasificación por prototipos**:

* calcula embeddings para el conjunto de validación,
* compara cada embedding contra un embedding de referencia por clase,
* predice la clase por **máxima similitud** o **mínima distancia** según la métrica,
* exporta CSVs de predicción por método.

### Evaluar el último experimento

```bash
python -m src.eval --data_dir "./dataset_ply" --runs_dir runs --run latest --export_csv
```

### Evaluar todos los experimentos

```bash
python -m src.eval --data_dir "./dataset_ply" --runs_dir runs --run all --export_csv
```

### Outputs

Dentro de cada run evaluado se guarda:

```
runs/<run_name>/evaluation/
├─ validation_predictions_Cosine_Similarity.csv
├─ validation_predictions_L2_Distance.csv
...
```

Cada CSV contiene:

* `path`, `true_label`, `pred_label`, `correct`, `score`

---
