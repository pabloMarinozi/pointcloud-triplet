# Análisis de consumo de RAM — actual vs lazy loading

## Datos del dataset

| Métrica | Valor |
|---------|-------|
| Archivos `.ply` totales | 57,180 |
| Tamaño total en disco | 79.3 GB |
| Tamaño promedio por archivo | 1.4 MB |
| Puntos promedio por nube | 25,200 |
| Clases (carpetas) | 100 |
| Archivos por clase | ~572 |

---

## 1. Flujo de datos actual (todo en RAM)

```
disco (.ply ASCII, ~1.4 MB c/u, ~25k pts)
    │
    ▼  build_all_point_clouds() — 57,180 iteraciones
    │
    │  for cada archivo:
    │    ├─ open3d.io.read_point_cloud(path)     → carga 25k pts (295 KB) a RAM
    │    ├─ sample_point_cloud()                 → reduce a n_points pts
    │    └─ guarda tupla (folder, path, array)   → el array de 25k se libera
    │
    ▼  Lista all_point_clouds: 57k tuplas con arrays de (n_points, 3) float32
    │
    │
    ▼  TripletTrainingPipeline.__init__()
    │
    │  ├─ Split estratificado 70/15/15 → referencias a los mismos arrays
    │  │
    │  ├─ TripletPointCloudDataset(train_clouds)
    │  │    normalize_unit_sphere() → CREA NUEVOS arrays normalizados
    │  │    self.items = [(cls, pts_norm), ...]  (~40k items, 70%)
    │  │
    │  └─ TripletPointCloudDataset(val_clouds)
    │       normalize_unit_sphere() → CREA NUEVOS arrays normalizados
    │       self.items = [(cls, pts_norm), ...]  (~9k items, 15%)
    │
    ▼  RAM: 1 copia original sampleada + 1 copia normalizada (~85%)
    │
    │
    ▼  DataLoader(num_workers=2)
    │
    │  fork() × 2 → cada worker hereda TODA la memoria del proceso padre
    │  Python rompe COW (reference counting) → cada worker duplica el dataset
    │
    ▼  RAM: ~×3 el consumo del proceso principal
    │
    │
    ▼  __getitem__ → augment() o sample_n() en cada batch
    │      pa = augment(pts_a, n_points)  → rota, escala, jitter, dropout, samplea
    │
    ▼  Tensor (3, n_points) float32 → GPU
```

### RAM actual por `n_points`

**Cálculo por componente:**

| Componente | ¿Cuántas nubes guarda? | Tamaño por nube |
|-----------|----------------------|----------------|
| `all_point_clouds` (sampleadas) | 57,180 | `n_points × 3 × 4` bytes |
| `train_ds.items` (normalizadas) | ~40,000 | `n_points × 3 × 4` bytes |
| `val_ds.items` (normalizadas) | ~8,600 | `n_points × 3 × 4` bytes |
| Workers (×2, fork) | ~80,000 c/u | `n_points × 3 × 4` bytes |

> Los workers heredan todo el proceso vía `fork()`. Aunque el SO lo comparte con COW, Python modifica los reference counts de cada objeto al leerlo, lo que rompe el COW página por página. En la práctica cada worker termina con una copia casi completa de los arrays del dataset.

**Tabla de RAM estimada:**

| `n_points` | Nubes en RAM | Datos numpy | Python overhead | Model/OS/CUDA | Workers (×2) | **RAM total** |
|-----------|-------------|-------------|-----------------|---------------|-------------|-------------|
| 512 | 57k + 49k + 80k×2 | 1.0 GB | 0.1 GB | 1.5 GB | 2.6 GB | **~5.2 GB** |
| 1024 | 57k + 49k + 80k×2 | 2.0 GB | 0.2 GB | 1.5 GB | 5.2 GB | **~8.9 GB** |
| 2048 | 57k + 49k + 80k×2 | 4.1 GB | 0.3 GB | 1.5 GB | 10.5 GB | **~16.4 GB** |

> **Nota:** con 8 GB de RAM, 512 entra, 1024 está al borde (depende del ancho del modelo), y 2048 es imposible. Los workers duplican el dataset y eso lo hace explotar.

---

## 2. Por qué explota con 2048 puntos

Tres factores se combinan:

1. **Dos copias del dataset**: `all_point_clouds` (original sampleada) + `self.items` (normalizada). Para 2048 pts son ~2.7 GB solo en arrays numpy.

2. **Multiplicación por workers**: cada worker es un `fork()` del proceso principal. Python, al tocar cualquier objeto (incluso para leerlo), modifica su reference count. Esto rompe el copy-on-write del SO y cada worker termina con una copia propia de los arrays. Con `num_workers=2`, la RAM se triplica (~2.7 → ~8+ GB solo en numpy).

3. **Espacio para el SO, CUDA runtime y bibliotecas**: ~1.5 GB adicionales.

El total supera los 8 GB y el SO mata el proceso (OOM killer).

---

## 3. Flujo de datos post lazy loading

```
disco (.ply ASCII, ~1.4 MB c/u, ~25k pts)
    │
    ▼  discover_point_clouds() — solo os.walk, sin abrir archivos
    │
    └─ Lista de (folder, path) → ~15 MB en RAM (strings)
    │
    │
    ▼  TripletTrainingPipeline.__init__()
    │
    │  ├─ Split estratificado → referencias a las mismas tuplas (15 MB)
    │  │
    │  ├─ TripletPointCloudDataset(train_paths)
    │  │    self.items = [(cls, path), ...]   → SIN cargar arrays
    │  │    self.class_to_indices = {cls: [idx, ...]}  → solo índices
    │  │
    │  └─ TripletPointCloudDataset(val_paths)
    │       self.items = [(cls, path), ...]
    │
    ▼  RAM: ~17 MB total (paths + diccionarios de índices)
    │
    │
    ▼  DataLoader(num_workers=4)
    │
    │  fork() × 4 → cada worker hereda solo 17 MB (insignificante)
    │  No hay arrays grandes que duplicar
    │
    ▼  __getitem__ (CADA BATCH lee del disco):
    │
    │  for triplet (anchor, positive, negative):
    │    ├─ open3d.io.read_point_cloud(path) → 25k pts (295 KB) desde disco
    │    ├─ normalize_unit_sphere(points)     → centrar + escalar
    │    ├─ augment(points, n_points)          → rota, escala, jitter, dropout
    │    │    └─ sample_n() → reduce a n_points
    │    └─ libera arrays intermedios
    │
    │  pico de RAM durante 1 batch: 4 workers × 3 nubes × 295 KB ≈ 3.5 MB
    │
    ▼  Tensor (3, n_points) float32 → GPU
```

### RAM post lazy loading

| Componente | Tamaño |
|-----------|--------|
| `self.items` (paths + strings) | ~15 MB |
| `class_to_indices` (dict de índices) | ~2 MB |
| Workers (×4, solo copian 17 MB c/u) | ~70 MB |
| Pico de carga por batch (4 workers × 3 nubes × 295 KB) | ~3.5 MB |
| Model/OS/CUDA | ~1.5 GB |
| **RAM total** | **~1.6 GB** |

> **No depende de `n_points`.** La nube cruda que se lee del disco siempre tiene ~25k pts (295 KB). El sampleo a `n_points` ocurre al final del augment, cuando todo lo demás ya se liberó.

---

## 4. Comparación

| `n_points` | Actual | Lazy loading | Reducción |
|-----------|--------|-------------|-----------|
| 512 | 5.2 GB | 1.6 GB | **3.3×** |
| 1024 | 8.9 GB | 1.6 GB | **5.6×** |
| 2048 | 16.4 GB (crash) | 1.6 GB | **10.3×** |

Con lazy loading, 2048 puntos en 8 GB de RAM es trivial. Incluso 4096 o más no cambian el consumo.

---

## 5. Impacto del lazy loading en el augment

### Por qué el augment es más efectivo con lazy loading

**Actualmente**, `build_all_point_clouds` elige una sola vez, al inicio, los `n_points` que se guardan de cada nube. Esa selección es fija para todo el entrenamiento. El augment después solo aplica rotación, escalado y ruido sobre **los mismos puntos una y otra vez** durante 100 épocas. El modelo nunca ve los otros ~24,000 puntos de la nube original.

**Con lazy loading**, cada vez que una nube entra al `__getitem__` se leen los 25k puntos completos del disco. El augment opera sobre la nube entera, el dropout descarta regiones aleatorias de la superficie, y el sampleo final elige un subset distinto cada vez.

### Secuencia de augment: actual vs lazy loading

**Actual:**

```
RAM → 1024 pts (ya normalizados y pre-sampleados del arranque)
         │
         └─ augment(1024 pts)
              ├─ rotación (sobre 1024)
              ├─ escalado (sobre 1024)
              ├─ jitter   (sobre 1024)
              ├─ dropout  (1024 → ~922)
              └─ sample_n(~922 → 1024, con reemplazo)
                                              │
                                              ▼  GPU
```

**Lazy loading:**

```
disco → 25k pts → normalize
                     │
                     └─ augment(25k pts)
                          ├─ rotación (sobre 25k)
                          ├─ escalado (sobre 25k)
                          ├─ jitter   (sobre 25k)
                          └─ dropout  (25k → ~22.5k)
                                          │
                                 sample_n(~22.5k → 1024)
                                          │
                                          ▼  GPU
```

### Diferencia clave

| | Actual | Lazy loading |
|---|---|---|
| Puntos que ve el augment | Mismos `n_points` fijos desde el arranque | 25k pts completos cada vez |
| Dropout descarta | ~100 pts de los mismos 1024 | ~2,500 pts de regiones aleatorias de la superficie |
| Sampleo repone desde | ~922 pts remanentes (con reemplazo) | ~22,500 pts (sin repetir casi nunca) |
| Varía entre épocas | Solo rotación/escala/ruido | Subset de puntos **distinto** en cada acceso |
| Exposición total por nube | 4% de la superficie (fijo) | ~100% a lo largo de las épocas |

> El modelo no crece ni cambia su arquitectura: recibe exactamente la misma entrada `(3, n_points)`. Lo que cambia es **qué puntos llegan**, no cuántos. La mayor diversidad actúa como regularizador: fuerza a la red a aprender features que funcionan para cualquier región de la superficie, en vez de memorizar la disposición de 1024 puntos fijos.

### ¿Quién ejecuta el augment?

El `__getitem__` del Dataset, pero **lo ejecutan los workers del DataLoader**, no el proceso principal:

```
main process (GPU)
    │
    ├─ pide batch N al DataLoader
    │
    ▼
DataLoader (cola de batches pre-fetcheados)
    │
    ├─ worker 0  ──┬─ __getitem__ → lee .ply del disco → normalize → augment → sample → tensor
    ├─ worker 1  ──┤
    ├─ worker 2  ──┤   (cada worker procesa su porción del batch en paralelo)
    └─ worker 3  ──┘
```

Cada worker, en paralelo:
1. Abre el `.ply` del disco (`open3d.io.read_point_cloud`)
2. Normaliza los ~25k pts
3. Aplica augment (rotación, escalado, jitter, dropout)
4. Samplea a `n_points`
5. Devuelve el tensor `(3, n_points)`

El proceso principal solo recibe tensores listos y los manda a la GPU. La I/O y el augment corren en paralelo, ocultos detrás del entrenamiento del batch anterior.

---

## 6. Costo del lazy loading

| Aspecto | Impacto |
|---------|---------|
| I/O de disco | 171k lecturas de `.ply` por época (57k × 3) |
| Latencia por batch | ~3 lecturas de 295 KB → insignificante en SSD |
| Workers necesarios | 4-6 (vs 2 actuales) para cubrir latencia de I/O |
| Tiempo de arranque | <1 s (vs 30-60 s actual) |
| Variabilidad entre épocas | **Mayor**: el augment opera sobre 25k pts en vez de `n_points` pre-filtrados, dropout+sample eligen puntos distintos cada vez |

---

## 7. Archivos a modificar

| Archivo | Cambio |
|---------|--------|
| `src/train.py` | Reemplazar `build_all_point_clouds()` por `discover_point_clouds()` (solo paths). Eliminar import de `sample_point_cloud` y `numpy`. |
| `src/data/dataset.py` | `PointCloudItem` = `(folder, path)` sin numpy. `__init__` no carga datos. `__getitem__` lee `.ply` del disco al vuelo. |
| `src/pipeline/trainer.py` | Cambiar desempaquetado de 3 a 2 elementos. Subir `num_workers` de 2 a 4. |
