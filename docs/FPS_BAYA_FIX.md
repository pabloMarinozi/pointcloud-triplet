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
# Antes: fps y fps_baya usaban el mismo FPS global
if sampling in ("fps", "fps_baya") and n >= n_points:
    ...

# Ahora: fps_baya delega a _fps_from_bayas
if n >= n_points and sampling == "fps_baya":
    from src.data.io import _fps_from_bayas
    idx = np.random.permutation(n)
    return _fps_from_bayas(points[idx], n_points)
```

Esto afecta directamente al dataset en modo lazy: cada `__getitem__` ahora muestrea respetando la partición en bayas de 500 puntos.

### `src/data/io.py` — fixes de borde

- **`_fps_from_bayas`**: el último chunk parcial ya no se trunca silenciosamente cuando la nube no es múltiplo exacto de 500.
- **`sample_point_cloud` (fps)**: cuando la nube tiene menos de `n_points`, ahora padea con reemplazo (antes devolvía menos puntos).

### `src/pipeline/trainer.py` — `_save_sampled_clouds` normaliza

```python
# Antes: guardaba coordenadas crudas
cloud_np = sample_point_cloud(file_path, self.n_points, self.sampling)

# Ahora: normaliza y usa sample_n (mismo algoritmo que el dataset)
pcd = o3d.io.read_point_cloud(file_path)
full_pts = np.asarray(pcd.points, dtype=np.float32)
cloud_np = normalize_unit_sphere(full_pts)
cloud_np = sample_n(cloud_np, self.n_points, self.sampling)
```

---

## Orden de ejecución: ¿cuándo se normaliza y cuándo se samplea?

### Línea de tiempo del pipeline (modo lazy)

```
train.py / run_mlflow.py
│
├─ 1. discover_point_clouds(data_dir)
│     Solo lista paths. No se lee ni normaliza ni samplea nada.
│     all_point_clouds = [(carpeta, path), ...]
│
├─ 2. PipelineClass.__init__(all_point_clouds, ...)
│     │
│     ├─ 2a. _save_sampled_clouds(all_point_clouds, timestamp)   ← si --save-sampled
│     │      Por cada archivo:
│     │        leer nube completa (23500 pts)     ← acá se lee del disco
│     │        normalize_unit_sphere(full_pts)     ← acá se normaliza
│     │        sample_n(norm_pts, 512, fps_baya)   ← acá se samplea (baya-aware)
│     │          └→ _fps_from_bayas
│     │        escribir .ply normalizado
│     │
│     ├─ 2b. split train/val/test  (solo paths, sin datos)
│     │
│     └─ 2c. crear datasets (solo guardan paths, no cargan nada)
│
└─ 3. pipeline.train()
      │
      └─ por cada epoch y batch:
           dataset.__getitem__(idx)
             │
             ├─ _load_from_disk(path)              ← acá se lee del disco
             │     leer nube completa (23500 pts)
             │     normalize_unit_sphere(full_pts)  ← acá se normaliza
             │
             └─ sample_n(norm_pts, 512, fps_baya)  ← acá se samplea (baya-aware)
                   └→ _fps_from_bayas
```

### ¿Por qué en ese orden y no en otro?

| Operación | ¿Antes o después de qué? | Motivo |
|-----------|------------------------|--------|
| **Leer** | Antes de todo | Sin datos no hay nada que procesar |
| **Normalizar** | Después de leer, antes de samplear | La normalización usa la nube **completa** para calcular media y escala reales. Si normalizás después de samplear, usás una versión recortada que no representa la forma verdadera |
| **Samplear** | Después de normalizar | Sobre la nube ya centrada y escalada, FPS selecciona los mismos puntos que sin normalizar, pero las coordenadas finales son comparables entre nubes |

### Caso concreto: nube de 23500 pts, n_points = 512, fps_baya

```
Paso 1 — Leer
  pts.shape = (23500, 3)     Z ∈ [32, 39]

Paso 2 — Normalizar  
  media = (-3.84, -10.22, 35.10)
  escala = 4.12
  pts.shape = (23500, 3)     Z ∈ [-0.4, 0.6]    ← misma forma, otro sistema de coordenadas

Paso 3 — Samplear (_fps_from_bayas)
  Reparte 512 puntos entre 47 bayas:
    42 bayas × 11 pts + 5 bayas × 10 pts
  pts.shape = (512, 3)       Z ∈ [-0.4, 0.6]    ← lista para el modelo o para guardar
```

---

## Por qué el entrenamiento mejora

### Cada baya recibe su proporción justa de puntos

Para una nube de 23500 puntos (47 bayas × 500), `n_points = 512`:

| Baya | Puntos |
|------|--------|
| 42 bayas | 11 pts c/u |
| 5 bayas | 10 pts c/u |
| **Total** | **512** |

Cada región del objeto está garantizada en el muestreo.  
Con FPS global, regiones enteras podían quedar sin representación porque el algoritmo no conoce la partición.

### Consistencia entrenamiento ↔ guardado

El PLY guardado con `--save-sampled` ahora es **idéntico** a lo que el modelo recibe en cada `__getitem__`:

```
flujo lazy actual:
  disco → leer nube completa → normalize_unit_sphere → sample_n(fps_baya)
                                                          └→ _fps_from_bayas
```

### Nubes visibles

Los PLYs generados están centrados en `(0,0,0)` con radio ~1.  
Se ven inmediatamente al abrirlos en cualquier visor 3D.
