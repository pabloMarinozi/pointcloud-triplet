# Fix: `fps_baya` y `--save-sampled`

## Problema original

### 1. `sample_n` ignoraba la estructura de bayas

`dataset.py:26` trataba `fps` y `fps_baya` como idénticos: ambos aplicaban FPS global sobre toda la nube.  
Esto significa que el modelo **nunca usó `_fps_from_bayas` durante el entrenamiento**.  
El muestreo no garantizaba que cada región de 500 puntos (cada «baya») estuviera representada proporcionalmente.

### 2. `_save_sampled_clouds` no normalizaba

Los PLYs guardados tenían coordenadas crudas del sensor (ej: Z ~ 32-39).  
Al abrirlos en un visor 3D (CloudCompare, MeshLab), la cámara apunta a `(0,0,0)` y la nube está a decenas de metros → invisible sin zoom manual.

### 3. Algoritmo distinto entre guardado y entrenamiento

`_save_sampled_clouds` usaba `sample_point_cloud` (con `_fps_from_bayas`), mientras que el dataset usaba `sample_n` (con FPS global).  
El PLY guardado no reflejaba lo que el modelo realmente veía durante el entrenamiento.

---

## Cambios realizados

### `src/data/dataset.py` — `sample_n` ahora respeta bayas

```python
# fps_baya → delega a _fps_from_bayas SIN permutacion global
# (la permutacion ocurre DENTRO de cada baya en io.py)
if n >= n_points and sampling == "fps_baya":
    from src.data.io import _fps_from_bayas
    return _fps_from_bayas(points, n_points)

# fps → FPS con permutacion global (comportamiento original)
if sampling == "fps" and n >= n_points:
    idx = np.random.permutation(n)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[idx])
    return np.asarray(pcd.farthest_point_down_sample(n_points).points, dtype=np.float32)

# random → np.random.choice (sin cambios)
```

Esto afecta directamente al dataset en modo lazy: cada `__getitem__` ahora muestrea respetando la partición en bayas de 500 puntos.

### `src/data/io.py` — `_fps_from_bayas` con permutación intra-baya

```python
def _fps_from_bayas(pts, n_points, baya_size=500):
    n_bayas = n // baya_size
    base = n_points // n_bayas
    remainder = n_points % n_bayas

    for i in range(n_bayas):
        take = base + 1 if i < remainder else base
        if take <= 0:
            continue
        baya = pts[i*500 : (i+1)*500]               # bloque contiguo original
        if len(baya) == 0:
            continue
        baya = baya[np.random.permutation(len(baya))]  # ← shuffle INTRA-baya
        # FPS sobre baya permutada
        ...
```

**Sin permutación global**: los bloques de 500 puntos conservan el orden del archivo → cada baya representa una región espacial coherente.

**Con permutación intra-baya**: dentro de cada región, el FPS arranca de un punto aleatorio distinto en cada epoch → variabilidad sin romper el agrupamiento.

### `src/data/io.py` — `_fps_from_bayas_split` para inspección visual

```python
def _fps_from_bayas_split(pts, n_points, baya_size=500):
    # Misma lógica que _fps_from_bayas pero devuelve lista de bayas individuales
    # (sin mergearlas). Útil para guardar cada baya como PLY separado.
```

### `src/data/io.py` — `sample_point_cloud` diferencias por modo

| Modo | Comportamiento | Motivo |
|------|---------------|--------|
| `random` | `np.random.choice` sobre coordenadas crudas | Comportamiento original (main) |
| `fps` | FPS con permutación global sobre coordenadas crudas | Comportamiento original (main) |
| `fps_baya` | Devuelve la nube completa sin samplear | El FPS baya-aware necesita la nube completa normalizada; se difiere a `sample_n` |

### `src/pipeline/trainer.py` — `_save_sampled_clouds` mejoras

```python
# Guarda en carpeta por run
sampled_dir = os.path.join("experiments", "sampling", timestamp)

# fps_baya: guarda merged + cada baya individual
if is_baya:
    merged = _fps_from_bayas(cloud_np, self.n_points)
    out_name = f"{timestamp}+{stem}_merged.ply"

    bayas = _fps_from_bayas_split(cloud_np, self.n_points)
    for i, baya in enumerate(bayas):
        out_name = f"{timestamp}+{stem}_baya{i:03d}.ply"

# Otros modos: sample_n normal
else:
    cloud_np = sample_n(cloud_np, self.n_points, self.sampling)
    out_name = f"{timestamp}+{base_name}"
```

### `experiments/run_traditional.py` — sufijo `_ss` en nombre de run

```python
def build_run_name(cfg):
    ...
    if cfg.get("save_sampled"):
        name += "_ss"
    return name
```

Ej: `np512_w8_ep1_bs16_lr3e-4_m0.5_fps_baya_lazy_ss`

---

## Estrategia de permutación por modo

| Modo | Permutación | Dónde | Efecto |
|------|-------------|-------|--------|
| `random` | inherente (`np.random.choice`) | `sample_n`, `sample_point_cloud` | N puntos aleatorios |
| `fps` | **global** | `sample_n:34`, `sample_point_cloud:119` | Shuffle total → FPS arranca de punto aleatorio |
| `fps_baya` | **intra-baya** | `io.py:51,90` | Shuffle dentro de cada bloque de 500 → FPS por región con arranque aleatorio |

---

## Flujo completo del pipeline

### Lazy mode

```
train.py
│
├─ discover_point_clouds(data_dir)
│      Solo lista paths. No se lee ni normaliza ni samplea nada.
│
├─ PipelineClass.__init__(all_point_clouds)
│     ├─ _save_sampled_clouds(all_point_clouds, timestamp)  ← si --save-sampled
│     │     leer nube completa → normalize_unit_sphere → sample_n(fps_baya)
│     │       └→ _fps_from_bayas (merged) + _fps_from_bayas_split (individual)
│     │     escribe experiments/sampling/{run_name}/
│     │
│     ├─ split train/val/test (solo paths)
│     └─ crear datasets (solo guardan paths)
│
└─ pipeline.train()
      └─ dataset.__getitem__(idx)
            ├─ _load_from_disk(path)
            │     leer nube completa → normalize_unit_sphere
            └─ sample_n(norm_pts, 512, fps_baya)
                  └→ _fps_from_bayas (con intra-baya permutation)
```

### Eager mode

```
train.py
│
├─ discover_point_clouds(data_dir)
│     Lee todas las nubes Y las samplea en memoria.
│
├─ PipelineClass.__init__(all_point_clouds)
│     ├─ _save_sampled_clouds(...)                         ← si --save-sampled
│     │     Para eager, sample_point_cloud ya sampleó en memoria.
│     │     _save_sampled lee del disco limpio y aplica sample_n.
│     │
│     └─ TripletPointCloudDataset:
│           sample_point_cloud(file, n_points, sampling)   ← samplea en init
│             └→ fps:      FPS en crudo con permutacion global
│             └→ fps_baya: nube completa (se difiere)
│             └→ random:   np.random.choice
│           normalize_unit_sphere(pts)                     ← normaliza en init
│
└─ pipeline.train()
      └─ dataset.__getitem__(idx)
            ├─ train: augment(pts, n_points, sampling)
            │     rotación + escala + ruido + dropout (hasta 10%)
            │     └→ sample_n(pts_dropped, n_points, sampling)
            │
            └─ eval: sample_n(pts, n_points, sampling)
```

---

## Orden de ejecución: ¿cuándo se normaliza y cuándo se samplea?

| Operación | ¿Antes o después de qué? | Motivo |
|-----------|------------------------|--------|
| **Leer** | Antes de todo | Sin datos no hay nada que procesar |
| **Normalizar** | Después de leer, antes de samplear | La normalización usa la nube **completa** para calcular media y escala reales |
| **Samplear** | Después de normalizar | Sobre la nube ya centrada y escalada, FPS selecciona puntos con coordenadas comparables entre nubes |

### Caso concreto: nube de 23500 pts, n_points = 512, fps_baya

```
Paso 1 — Leer
  pts.shape = (23500, 3)     Z ∈ [32, 39]

Paso 2 — Normalizar  
  media = (-3.84, -10.22, 35.10)
  escala = 4.12
  pts.shape = (23500, 3)     Z ∈ [-0.4, 0.6]

Paso 3 — Samplear (_fps_from_bayas)
  n_bayas = 23500 // 500 = 47
  base = 512 // 47 = 10, remainder = 42

  42 bayas × 11 pts + 5 bayas × 10 pts = 512 pts total
  Cada baya: shuffle(500 pts) → FPS(take) → aporta su cuota
```

---

## Por qué el entrenamiento mejora

### Cada baya recibe su proporción justa de puntos

| Nube de 23500 pts / n_points=512 | |
|---|---|
| 42 bayas | 11 pts c/u |
| 5 bayas | 10 pts c/u |
| **Total** | **512** |

Cada región del objeto está garantizada en el muestreo.  
Con FPS global, regiones enteras podían quedar sin representación.

### Variabilidad sin romper estructura

La permutación intra-baya asegura que el FPS arranque de un punto distinto en cada epoch, pero siempre dentro de la misma región espacial. Esto da diversidad al entrenamiento sin sacrificar la cobertura regional.

### Consistencia entrenamiento ↔ guardado

El PLY guardado con `--save-sampled` ahora es **idéntico** a lo que el modelo recibe en cada `__getitem__`:

```
flujo lazy actual:
  disco → leer nube completa → normalize_unit_sphere → sample_n(fps_baya)
                                                          └→ _fps_from_bayas
```

### Nubes visibles

Los PLYs generados están centrados en `(0,0,0)` con radio ~1. Se ven inmediatamente al abrirlos en cualquier visor 3D.

### Estructura de archivos guardados

```
experiments/sampling/
└── np512_w8_ep1_bs16_lr3e-4_m0.5_fps_baya_lazy_ss/
    ├── {run_name}+cloud_001_merged.ply       ← 512 pts mergeados
    ├── {run_name}+cloud_001_baya000.ply       ← baya 0 individual
    ├── {run_name}+cloud_001_baya001.ply       ← baya 1 individual
    └── ...
```
