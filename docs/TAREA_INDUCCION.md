# Tarea de inducción – Pointcloud Triplet

Bienvenido/a al repo. Esta tarea sirve para que te familiarices con el pipeline
reproduciendo un resultado real y, de paso, dejes el código un poco mejor que como
lo encontraste. Tiene dos partes: **B** (reproducir) y **A** (mejorar reproducibilidad).

Leé primero el `README.md`, `docs/RESULTADOS.md` (resultados del proyecto) y
`docs/EXPERIMENTO_ACTUAL.md` (dónde está cada cosa en el código).

---

## Contexto en una línea

Metric learning sobre nubes de puntos 3D (TripletNet / PointNet) para identidad por
geometría en racimos de uva. Se entrena con triplet loss y se clasifica comparando el
embedding de cada muestra contra prototipos de referencia. El mejor modelo es
`w8_np512_m0.5_lr3e-4_bs16_seed42` a 1000 épocas.

---

## Setup

1. Clonar el repo y crear el entorno (Python 3.10):
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # Linux/Mac: source .venv/bin/activate
   ```
2. Instalar dependencias (ver nota de PyTorch/CUDA en `requirements.txt`):
   ```bash
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121  # con GPU NVIDIA
   pip install -r requirements.txt
   ```
3. Vas a recibir **por separado** (no están en git):
   - El **dataset** de `.ply` (se te indicará la ruta; respetá la estructura de carpetas).
   - La **carpeta del run** `runs/w8_np512_m0.5_lr3e-4_bs16_seed42/` (modelo + splits).

   Colocá la carpeta del run dentro de `runs/` en la raíz del repo.

---

## Parte B – Reproducir el resultado del mejor modelo (ep1000)

**Objetivo:** correr la evaluación del mejor modelo y obtener métricas equivalentes a
las publicadas. **No tienen que dar idénticas** (ver Parte A para entender por qué);
alcanza con que queden dentro de ~1 punto porcentual.

1. Corré la evaluación en val y test:
   ```bash
   python -m src.eval --data_dir <ruta_al_dataset> --run w8_np512_m0.5_lr3e-4_bs16_seed42 --split both
   ```
2. Mirá la mejor combinación (al final del log, "MEJOR RUN GLOBAL", y los bloques por
   estrategia). El mejor resultado esperado es **`centroid_all` + `L1 Distance`**:

   | Split | acc | top5 | MRR | mean_rank |
   |-------|-----|------|-----|-----------|
   | val   | ~34.1% | ~81% | ~0.54 | ~3.7 |
   | test  | ~34.0% | ~80% | ~0.53 | ~3.8 |

3. **Entregable B:** un documento corto (`docs/repro_<tu_nombre>.md` o en el PR) con:
   - el comando exacto que usaste y en qué hardware (GPU/CPU),
   - la tabla de tus resultados val/test para `centroid_all + L1`,
   - una observación: ¿coincidieron con los esperados? ¿hubo diferencias? ¿de cuánto?

> Si corrés la evaluación **dos veces**, vas a ver que los números cambian un poco entre
> corridas. Eso no es un error tuyo: es el problema que vas a arreglar en la Parte A.

---

## Parte A – Hacer la evaluación reproducible (tu primer PR)

**Diagnóstico:** el muestreo de puntos (`sample_n` en `src/data/dataset.py` y
`sample_point_cloud` en `src/data/io.py`) usa `np.random.choice` **sin semilla**, y
`src/eval.py` nunca fija una. Por eso cada corrida submuestrea las nubes distinto y las
métricas bailan entre ejecuciones.

**Tarea:**

1. Agregá un argumento `--seed` a `src/eval.py` (default 42) y llamá a
   `set_seed(args.seed)` (ya existe en `src/utils/seed.py`) al inicio de `main()`,
   **antes** de cualquier muestreo o generación de embeddings.
2. Verificá que ahora **dos corridas seguidas con el mismo `--seed` dan métricas idénticas**.
3. Dejá una nota breve en `docs/RESULTADOS.md` (o en el README) aclarando que la eval es
   determinista a partir de tal seed.

**Entregable A:** un Pull Request con el cambio, que incluya en la descripción:
- el antes/después (dos corridas distintas antes; dos corridas iguales después),
- qué archivos tocaste y por qué.

---

## Reglas de trabajo (importante)

- Trabajá en una **rama** propia (`git checkout -b induccion/<tu_nombre>`), nunca sobre `main`.
- Entregá vía **Pull Request**; no pushees directo a `main`.
- **Nunca** commitees `runs/`, modelos (`*.pt`), datasets, ni logs de consola: ya están en
  `.gitignore` y deben quedarse afuera.
- Si tenés dudas sobre el alcance, preguntá antes de hacer cambios grandes.

Cualquier cosa que no esté clara en el repo, anotala: también es parte de la tarea
detectar qué documentación falta.
