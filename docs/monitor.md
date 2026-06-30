# Monitor de sistema durante entrenamiento

## Motivación

Entrenar con `n_points` altos (1024, 2048) consume mucha RAM — ver [`MEMORIA_RAM.md`](MEMORIA_RAM.md) para el análisis detallado. El monitor registra en tiempo real CPU, RAM y GPU para detectar cuellos de botella, picos de memoria o subutilización de GPU durante el entrenamiento.

## Componentes

### `src/utils/monitor.py` — `SystemMonitor`

Clase que corre en un **thread daemon** independiente. Cada `interval` segundos (default 5 s) muestrea y escribe una fila a un CSV.

**Métricas registradas:**

| Columna | Fuente | Descripción |
|---------|--------|-------------|
| `timestamp` | `time.strftime` | Hora del sampleo (HH:MM:SS) |
| `elapsed_s` | `time.perf_counter` | Segundos desde que arrancó el monitor |
| `cpu_percent` | `psutil.cpu_percent(interval=None)` | % de CPU de todo el sistema |
| `ram_used_gb` | `psutil.virtual_memory().used` | RAM usada en GB |
| `ram_percent` | `psutil.virtual_memory().percent` | % de RAM usada |
| `gpu{i}_util_percent` | `pynvml.nvmlDeviceGetUtilizationRates` | % de utilización de GPU `i` |
| `gpu{i}_mem_used_gb` | `pynvml.nvmlDeviceGetMemoryInfo` | VRAM usada en GB por GPU `i` |
| `gpu{i}_mem_percent` | `pynvml.nvmlDeviceGetMemoryInfo` | % de VRAM usada por GPU `i` |

**Ciclo de vida:**

```
start()                stop()
   │                      │
   ├─ write_header()      ├─ stop_event.set()
   ├─ sample()            ├─ thread.join()
   └─ thread.start()      └─ sample() (última muestra)
```

- `start()`: escribe header CSV, toma primera muestra, lanza el thread.
- `stop()`: señal de parada, espera al thread, toma muestra final.

Si `pynvml` no está disponible o no hay GPUs NVIDIA, omite las columnas de GPU sin tirar error.

### Integración en `src/pipeline/trainer.py`

El pipeline llama a `_start_monitor()` antes del loop de épocas y a `monitor.stop()` al terminar:

```python
# trainer.py : train()
monitor = self._start_monitor()       # arranca antes de las épocas

for epoch in range(start_epoch, ...):
    ...

if monitor is not None:
    monitor.stop()                    # detiene al finalizar
```

**Salida:** el CSV se guarda como `system_metrics.csv` dentro del directorio del experimento (`runs/<timestamp>/system_metrics.csv`), junto con `metrics.csv`, `config.json` y `training.log`.

### Dependencia nueva

`psutil==7.1.3` agregado a `requirements.txt`. `pynvml` es opcional (viene con el driver NVIDIA, no se lista como dependencia).

## Uso

El monitor se activa automáticamente en cada run de entrenamiento. No requiere flags adicionales. Si `psutil` no está instalado, loguea un warning y continúa sin monitoreo.

Para analizar los datos:

```python
import pandas as pd

df = pd.read_csv("runs/<run>/system_metrics.csv")
print(df.describe())
df.plot(x="elapsed_s", y=["cpu_percent", "ram_percent"])
```
