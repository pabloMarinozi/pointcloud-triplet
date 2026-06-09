# Guía de conducta con el repo

Cómo trabajamos en este proyecto. La idea no es burocracia: son las pocas reglas que
mantienen el repo sano ahora que somos más de una persona. Si algo acá no se entiende o
te parece mejorable, decilo — esta guía también se itera.

---

## 1. Antes de tocar nada

- Leé `README.md`, `docs/EXPERIMENTO_ACTUAL.md` (dónde está cada cosa) y
  `docs/RESULTADOS.md` (qué se obtuvo hasta ahora).
- Levantá el entorno siguiendo `requirements.txt` (Python 3.10; ver la nota de PyTorch/CUDA).
- Antes de un cambio grande, asegurate de entender la parte del código que vas a tocar.
  Si no la entendés, preguntá o pedí que te la expliquen — no cambies a ciegas.

---

## 2. Git: ramas y Pull Requests

- **Nunca** trabajes ni pushees directo sobre `main`.
- Una rama por tarea, con nombre descriptivo: `feature/...`, `fix/...`, `induccion/...`.
- Todo entra por **Pull Request**, revisado antes de mergear.
- Commits chicos y con mensaje claro (qué y por qué, no "cambios varios").
- Antes de abrir el PR, **leé tu propio `git diff` completo.** Sos responsable de cada
  línea que proponés, la hayas escrito vos o un asistente.

---

## 3. Qué NUNCA se commitea

Ya está cubierto por `.gitignore`, pero la regla es tuya igual:

- `runs/` y cualquier checkpoint o peso (`*.pt`, `*.pth`, `*.npz`).
- El **dataset** (`.ply`) ni nada derivado pesado.
- Logs de consola (`*.txt` de train/eval), PDFs, papers.
- Credenciales, tokens, rutas privadas o datos sensibles.

Si dudás si algo va al repo, asumí que no y preguntá.

---

## 4. Finales de línea y diffs limpios

- El repo usa LF (ver `.gitattributes`). No cambies esa config.
- Si abrís un archivo y tu editor lo reescribe entero, el `git diff` te va a mostrar el
  archivo completo como "modificado" aunque no hayas cambiado nada real: eso es ruido de
  CRLF. **No lo commitees.** Revisá siempre el diff: un cambio real se ve como pocas
  líneas, no como el archivo entero.
- No reformatees archivos que no estás tocando "de paso". Cada PR cambia lo que tiene que
  cambiar y nada más.

---

## 5. Código

- Cambios chicos y acotados; un PR = una cosa.
- No rompas lo que ya anda: si tocás una función compartida, fijate quién la usa.
- Seguí el estilo que ya hay en el repo (nombres, estructura `src/...`, type hints).
- Documentá lo no obvio con un docstring o un comentario corto.
- Si encontrás algo que está mal o sin documentar, anotalo (issue, TODO, o avisá). Detectar
  deuda también suma.

---

## 6. Datos, modelos y reproducibilidad

- El **dataset y los `runs/` no están en el repo**; se entregan aparte. El `--data_dir` es
  una ruta externa. No los busques dentro del repo.
- La **semilla del proyecto es 42** (está en `config.json` y en el nombre de cada run). El
  entrenamiento y el split son deterministas con esa semilla.
- **La evaluación hoy NO es determinista** (el muestreo de puntos no fija semilla), así que
  las métricas bailan un poco entre corridas. Es un problema conocido, no un bug tuyo.
- Para reproducir un resultado: las métricas no tienen que dar idénticas, alcanza con que
  queden dentro de ~1 punto porcentual de lo publicado.
- Para entrenar/evaluar en tiempos razonables hace falta GPU NVIDIA (CUDA 12.1).

---

## 7. Si usás un asistente de IA (Claude u otro)

- Es una herramienta para acelerar, no para tercerizar la responsabilidad. **El código del
  PR es tuyo**, lo entendés y lo defendés vos.
- Dale contexto: que lea primero los docs del repo (sección 1) antes de proponer cambios.
- Usalo para entender el código y para lo repetitivo (docstrings, descripción del PR,
  resumir un diff), no solo para generar.
- **No commitees nada que no entiendas** ni que no hayas revisado línea por línea.
- Pedile que verifique, no que afirme: por ejemplo, "corré la eval dos veces y comparame",
  mejor que creerle que "quedó reproducible".
- Cuidado con que reescriba archivos enteros (ver sección 4) o intente versionar `runs/`,
  modelos o datos.

---

## 8. Comunicación

- Ante una duda de alcance, preguntá **antes** de hacer un cambio grande; es más barato que
  rehacer un PR.
- Si te trabás más de un rato, pedí ayuda: no es perder tiempo, es ahorrarlo.
- Lo que te costó entender del repo probablemente le falte a la documentación. Proponé
  mejorarla.
