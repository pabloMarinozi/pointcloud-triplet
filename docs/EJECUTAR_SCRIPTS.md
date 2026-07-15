# Ejecutar todos los entrenamientos

El archivo `scripts/run_all.py` ejecuta secuencialmente y en orden alfabético
todos los lanzadores `run_*.py` ubicados en `experiments/scripts/`.

Desde la raíz del proyecto y con el entorno virtual activado, ejecutar:

```bash
python scripts/run_all.py
```

La salida se muestra en la terminal y también se guarda en:

```text
experiments/logs/<fecha>/run_all_<hora>.log
```

Este archivo contiene el proceso completo, desde el primer entrenamiento hasta
el último. Además, cada lanzador genera su propio log individual:

```text
experiments/logs/<fecha>/run_<nombre_del_entrenamiento>.log
```

Si un entrenamiento falla, `run_all.py` registra el error y continúa
obligatoriamente con el siguiente. Una vez intentados todos los entrenamientos,
muestra un resumen de los fallidos y devuelve código `1` si hubo al menos un
error. Todos los logs generados se conservan.
