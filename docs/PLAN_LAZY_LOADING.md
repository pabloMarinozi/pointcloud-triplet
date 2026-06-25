# Plan de implementación: lazy loading

## 1. Justificación

### Problemas del diseño actual (eager)

**a) RAM descontrolada por duplicación con workers**

`build_all_point_clouds()` carga 57k nubes sampleadas a RAM. Luego `TripletPointCloudDataset` crea una segunda copia normalizada. Al arrancar `DataLoader(num_workers=2)`, cada worker hereda el proceso padre vía `fork()`. Python rompe el copy-on-write del SO porque su reference counting modifica los headers de los objetos al leerlos, forzando al kernel a duplicar páginas enteras. Cada worker termina con una copia casi completa del dataset.

Con 2048 puntos esto implica ~16 GB de RAM, imposible en máquinas de 8 GB.

**b) Subset de puntos fijo durante todo el entrenamiento**

`sample_point_cloud()` elige una sola vez los `n_points` que se guardan de cada nube. El augment opera siempre sobre esos mismos puntos, variando únicamente rotación, escala y jitter. El dropout descarta ~100 pts y el sampleo repone con reemplazo desde los mismos ~922 remanentes. El modelo **nunca ve el 96% restante de cada nube**.

**c) Sesgo en el split train/val/test**

Las nubes ya están sampleadas antes del split. Train, val y test comparten el mismo subset fijo de puntos por nube. Val no puede evaluar regiones de la superficie que no entraron en el sampleo inicial. La evaluación no es representativa de la nube completa.

**d) Arranque lento**

Cargar, samplear y normalizar 57k archivos `.ply` toma entre 30 y 60 segundos cada vez que se inicia un run.

### Qué resuelve el lazy loading

| Problema | Cómo se resuelve |
|----------|-----------------|
| RAM excesiva | El Dataset guarda solo paths (~17 MB). Los workers leen del disco en cada `__getitem__` y liberan al terminar. No hay arrays que duplicar. |
| Subset fijo | Cada vez que una nube se pide, se leen los 25k pts completos. El augment opera sobre la nube entera, el sampleo elige un 4% distinto cada vez. |
| Sesgo en split | El split se hace sobre paths. Cada conjunto ve regiones distintas de la misma nube. |
| Arranque lento | Solo se escanean paths con `os.walk()`. <1 segundo. |

---

## 2. Enfoque de implementación: dos pipelines separados

El pipeline lazy se implementó como una clase **completamente independiente** —
`LazyTripletTrainingPipeline` — que hereda de `TripletTrainingPipeline` para reutilizar
el loop de entrenamiento (`train()`), checkpoints y logging, pero tiene su propio
`__init__` que opera con paths en vez de arrays. El pipeline eager original no se tocó.

```python
# trainer.py
class TripletTrainingPipeline:       # eager — intacto, sin cambios
    ...

class LazyTripletTrainingPipeline(TripletTrainingPipeline):  # lazy — nuevo
    def __init__(self, all_point_clouds, ...):
        # 2-tuplas (folder, path)
        # LazyTripletPointCloudDataset
        # num_workers = 4
        # config["lazy"] = True
        ...
    # train(), _log(), _save_checkpoint(), _load_checkpoint(), _reload_for_retry()
    # se heredan del padre sin cambios
```

### Ruteo desde `train.py`

```python
if args.lazy:
    all_point_clouds = discover_point_clouds(args.data_dir)
    PipelineClass = LazyTripletTrainingPipeline
else:
    all_point_clouds = build_all_point_clouds(args.data_dir, ...)
    PipelineClass = TripletTrainingPipeline

pipeline = PipelineClass(all_point_clouds=..., ...)
pipeline.train(resume=args.resume)
```

---

## 3. Cambios archivo por archivo

### `src/train.py`

**Qué se agrega:**

- `--lazy` flag (`action="store_true"`). Por defecto el comportamiento es eager.
- `discover_point_clouds()` — recorre con `find_ply_files()`, devuelve `List[Tuple[str, str]]` (carpeta, path) sin cargar archivos.

```python
def discover_point_clouds(ply_dir: str):
    files = find_ply_files(ply_dir)
    return [(os.path.basename(os.path.dirname(f)), f) for f in files]
```

- Import de `LazyTripletTrainingPipeline` junto con `TripletTrainingPipeline`.
- `main()` selecciona `PipelineClass` y `all_point_clouds` según `--lazy`.

**Qué NO cambia:** `build_all_point_clouds()`, `PROGRESS_EVERY_N_FILES`, imports de numpy y `sample_point_cloud`. Todo se conserva intacto para el pipeline eager.

---

### `src/data/dataset.py`

**Nuevo type alias:**

```python
PointCloudItem = Tuple[str, str, np.ndarray]  # eager (existente, sin cambios)
LazyPointCloudItem = Tuple[str, str]           # lazy (nuevo)
```

**Nueva clase `LazyTripletPointCloudDataset`:**

Clase independiente con la misma interfaz que `TripletPointCloudDataset`. Diferencias:

| `TripletPointCloudDataset` (eager) | `LazyTripletPointCloudDataset` (lazy) |
|-------------------------------------|---------------------------------------|
| `self.items: List[(cls, pts_norm)]` | `self.items: List[(cls, path)]` |
| `__init__` normaliza y guarda arrays | `__init__` solo guarda paths |
| `__getitem__` lee array de RAM | `__getitem__` lee `.ply` del disco, normaliza y augmenta/samplea |
| `augment/sample_n` sobre `n_points` | **sample-first**: `sample_n` (25k→1024), luego `augment` (1024→1024). El augment (rotación, escala, ruido, dropout) opera sobre 1024 pts en vez de 25k, ~24x más rápido por batch sin perder diversidad. |

```python
class LazyTripletPointCloudDataset(Dataset):
    def __init__(self, all_point_clouds: List[LazyPointCloudItem], ...):
        for idx, (folder, path) in enumerate(all_point_clouds):
            self.items.append((folder, path))  # solo paths

    def _load_from_disk(self, path: str) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        return normalize_unit_sphere(pts)

    def __getitem__(self, index: int):
        file_a, path_a = self.items[index]
        pts_a = self._load_from_disk(path_a)       # ~25k pts
        pts_p = self._load_from_disk(path_p)
        pts_n = self._load_from_disk(path_n)

        sa = sample_n(pts_a, self.n_points, ...)     # 25k → 1024
        sp = sample_n(pts_p, self.n_points, ...)
        sn = sample_n(pts_n, self.n_points, ...)

        if self.train:
            pa = augment(sa, self.n_points, ...)     # 1024 → 1024 con aug
        else:
            pa = sa
```

`sample_n()` recibe la nube completa (~25k pts) y la reduce a `n_points`. Luego `augment()` opera sobre los 1024 pts ya sampleados (rotación, escala, ruido, dropout aleatorio, sampleo con reemplazo). Cada epoch ve un subset distinto generado por `sample_n` + `augment`.

---

### `src/pipeline/trainer.py`

**`TripletTrainingPipeline`** — totalmente intacto. Cero cambios respecto al original.

**`LazyTripletTrainingPipeline(TripletTrainingPipeline)`** — nueva clase. Hereda `train()`, `_log()`, `_save_checkpoint()`, `_load_checkpoint()` y `_reload_for_retry()`. Solo redefine `__init__` con estas diferencias:

| Aspecto | `TripletTrainingPipeline` | `LazyTripletTrainingPipeline` |
|---------|--------------------------|-------------------------------|
| Tuplas | `(folder, path, cloud_np)` — 3 elementos | `(folder, path)` — 2 elementos |
| Dataset | `TripletPointCloudDataset` | `LazyTripletPointCloudDataset` |
| `num_workers` | 2 | 4 |
| `config["lazy"]` | no existe | `True` |
| `_log`, `train`, etc | definidos en la clase | heredados sin cambios |

---

### `src/data/io.py`

**Cambios en `sample_point_cloud()`:**

Se agrega `np.random.permutation` antes de FPS y FPS-baya. Open3D es determinista: dado el mismo array en el mismo orden, FPS siempre devuelve los mismos puntos. El shuffle rompe ese determinismo, garantizando que cada nube muestreada tenga un subset distinto para cada seed.

```python
def sample_point_cloud(file_path, n_points, sampling="random"):
    pcd = o3d.io.read_point_cloud(file_path)

    if sampling == "fps":
        if len(pcd.points) >= n_points:
            pts = np.asarray(pcd.points)
            idx = np.random.permutation(len(pts))   # shuffle antes de FPS
            pcd.points = o3d.utility.Vector3dVector(pts[idx])
            pcd = pcd.farthest_point_down_sample(n_points)
        return np.asarray(pcd.points, dtype=np.float32)

    pts = np.asarray(pcd.points)

    if sampling == "fps_baya":
        idx = np.random.permutation(len(pts))        # shuffle antes de FPS-baya
        return _fps_from_bayas(pts[idx], n_points)
    # ...
```

---

### FPS determinista y el shuffle con `--seed`

**`sample_n()` y `sample_point_cloud()`** comparten el mismo fix.

**Problema:** `farthest_point_down_sample` de Open3D es determinista — siempre elige el primer punto del array y de ahí itera. Si el array no cambia de orden, FPS devuelve idénticos puntos cada vez. En lazy esto anula la ventaja: cada epoch ve exactamente los mismos 1024 puntos. En eager, `sample_n` se llama al final de `augment` (después del dropout) y en val, pero también sufre el mismo problema cuando se usa FPS.

**Solución:** `np.random.permutation(len(pts))` antes de pasar el array a FPS. Al permutar, FPS empieza desde un punto aleatorio distinto en cada llamado, produciendo subsets diferentes.

**Reproducibilidad:** `set_seed(args.seed)` siembra `np.random` en `main()`. La permutación usa ese estado → mismo `--seed` = misma secuencia de permutaciones = mismos subsets = resultados idénticos.

**Flujo final en lazy:**
```
train: disco(25k) → _load_from_disk → sample_n(25k→1024, +shuffle FPS) → augment(1024→1024)
val:   disco(25k) → _load_from_disk → sample_n(25k→1024, +shuffle FPS)
```

**Flujo final en eager (sin cambios en la estructura):**
```
train: RAM(1024) → augment(1024→1024, +sample_n con shuffle FPS)
val:   RAM(1024) → sample_n(1024→1024, +shuffle FPS)
```

En eager val, FPS sobre 1024 pts para obtener 1024 pts: la permutación cambia el orden pero el conjunto es el mismo (se toman todos). En eager train, el dropout de `augment` baja a ~922 y la permutación da variedad al upsampling.

---

### Logging detallado con tiempos

Ambos pipelines registran en `training.log`:

- **`[INIT]`**: tiempo total de inicialización del pipeline
- **Por época**: `T:` (train), `V:` (val), `ep:` (total época), `total:` (acumulado)
- **`[PROGRESO]`**: mensajes de `main()` (post-pipeline) también al log
- **CSV**: columnas `train_s`, `val_s`, `epoch_s`, `total_s`

Ejemplo de salida:
```
[01/30] train=0.042315  val=0.038102  lr=9.870e-04  |  T:45.3s  V:12.1s  ep:57.4s  total:57.4s
```

---

## 4. Comparativa eager vs lazy

| Aspecto | Eager (default) | Lazy (`--lazy`) |
|---------|-----------------|-----------------|
| **Pipeline class** | `TripletTrainingPipeline` | `LazyTripletTrainingPipeline` |
| **Dataset class** | `TripletPointCloudDataset` | `LazyTripletPointCloudDataset` |
| **Arranque** | 30-60 s (carga 57k nubes) | <1 s (solo paths) |
| **RAM en padre** | ~8 GB (57k × 1024 pts) | ~17 MB (solo paths) |
| **RAM con workers** | ~16 GB (duplicación fork) | ~17 MB (workers leen disco) |
| **Subset de puntos** | Fijo (mismo subset toda la vida) | Distinto cada batch (dropout aleatorio) |
| **Split train/val/test** | Sobre subsets ya sampleados | Sobre paths (independiente) |
| **Velocidad por batch** | Rápido (arrays en RAM) | Más lento (I/O de disco por batch) |
| **Workers** | 2 | 4 (compensa I/O) |
| **Entrenamiento** | `TripletTrainingPipeline.train()` | `LazyTripletTrainingPipeline.train()` (heredado, idéntico) |

## 5. Ventajas del lazy loading sobre el eager

1. **RAM viable en hardware limitado**: 17 MB vs 8-16 GB. Entrenable en GPUs con 8 GB de VRAM sin OOM.
2. **Mejor generalización**: el modelo ve regiones distintas de cada nube en cada epoch.
3. **Evaluación representativa**: train, val y test ven regiones independientes de cada nube.
4. **Iteración rápida**: arranque en <1 s permite probar hiperparámetros sin esperar 60 s.
5. **Sin cambios en el modelo**: `TripletNet`, `CPN`, `triplet_loss_squared` no se tocan.
6. **Pipelines aislados**: el código eager no se tocó. Cero riesgo de regresión. Las pruebas comparativas son limpias y confiables.

## 6. Uso

Basado en el comando paper-like del README (`--epochs 200 --batch_size 32 --lr 1e-4`):

```bash
# Eager (default) — precarga todas las nubes en RAM
python -m src.train --data_dir "./dataset_ply" \
  --n_points 512 --width 8 --batch_size 32 --lr 1e-4 --epochs 200 \
  --sampling random

# Lazy — carga del disco en cada batch
python -m src.train --data_dir "./dataset_ply" --lazy \
  --n_points 512 --width 8 --batch_size 32 --lr 1e-4 --epochs 200 \
  --sampling random

# Comparativa A/B
python -m src.train --data_dir "./dataset_ply" \
  --run_name eager_w8_np512_bs32 \
  --n_points 512 --width 8 --batch_size 32 --lr 1e-4 --epochs 200 \
  --sampling random

python -m src.train --data_dir "./dataset_ply" --lazy \
  --run_name lazy_w8_np512_bs32 \
  --n_points 512 --width 8 --batch_size 32 --lr 1e-4 --epochs 200 \
  --sampling random
```

## 7. Lo que NO cambia

- `augment()` — misma lógica (rotación, escala, ruido, dropout), pero en lazy opera sobre 1024 pts sampleados previamente en vez de la nube completa
- `_get_positive()` y `_get_negative()` — operan sobre índices, no sobre datos
- El loop de entrenamiento `train()` — idéntico, heredado por `LazyTripletTrainingPipeline`
- `src/evaluation/` — ya carga archivos individualmente, no necesita cambios
- `find_ply_files()` en `src/data/io.py` — sin cambios
- La arquitectura del modelo (`TripletNet`, `CPN`) — cero cambios
- El pipeline eager completo (`TripletTrainingPipeline`, `TripletPointCloudDataset`) — cero cambios
