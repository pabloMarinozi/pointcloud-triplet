# Guía de experimentos — corrida definitiva (paper)

## Estado actual (ya hecho, NO repetir)
- Grid **random** 200ep completo — mejor: **w8 / lr3e-4 / random = test acc 0.192**
- Campeón random extendido a **1000ep**
- **w8 / lr3e-4 / fps global 200ep = test acc 0.219**  ← **mejor actual**

## Regla de oro
Por cada experimento: **entrenar → evaluar → anotar el `test acc` (best del `evaluation_report.json`)**.
Solo se extiende a 1000ep lo que **supere 0.219** en test a 200 epochs.

## Plantilla de comandos
Reemplazar `<W> <LR> <S> <E> <NAME>` según cada tabla. Va en una sola línea (corre igual en PowerShell o Git Bash).

**Entrenar:**
```
python -m src.train --data_dir "D:/thresh105_qr120_umbral008_acomodado" --runs_dir runs --n_points 512 --batch_size 16 --margin 0.5 --seed 42 --width <W> --lr <LR> --sampling <S> --epochs <E> --run_name <NAME>
```

**Evaluar:**
```
python -m src.eval --data_dir "D:/thresh105_qr120_umbral008_acomodado" --runs_dir runs --run <NAME> --split both --export_csv --seed 42
```

> Nombres estilo viejo (sin epochs): por eso se lanzan directo con `--run_name`, no con `run_traditional.py`.

---

## FASE 1 — Exploración (200 epochs). Correr EN ESTE ORDEN.

| # | run_name (`<NAME>`) | W | LR | sampling |
|---|---|---|---|---|
| 1 | `w8_np512_m0.5_lr3e-4_bs16_seed42_fpsbaya` | 8 | 3e-4 | fps_baya |
| 2 | `w8_np512_m0.5_lr1e-3_bs16_seed42_fps` | 8 | 1e-3 | fps |
| 3 | `w8_np512_m0.5_lr5e-4_bs16_seed42_fps` | 8 | 5e-4 | fps |
| 4 | `w8_np512_m0.5_lr1e-4_bs16_seed42_fps` | 8 | 1e-4 | fps |
| 5 | `w16_np512_m0.5_lr3e-4_bs16_seed42_fps` | 16 | 3e-4 | fps |
| 6 | `w32_np512_m0.5_lr3e-4_bs16_seed42_fps` | 32 | 3e-4 | fps |

- **#1** cierra la comparación de muestreos en la config campeón (random 0.192 / fps 0.219 / **baya-fixed ?**).
- **#2–#4** sweep de lr sobre fps global (lr 3e-4 ya está hecho = 0.219).
- **#5–#6** chequean que w8 siga ganando **con fps** (no asumir que la ventaja de width transfiere entre muestreos).

### GATE fps_baya (evaluar DESPUÉS del #1)
- Si **#1 da test acc ≥ 0.19** (matchea o supera random) → fps_baya tiene futuro: correr también su sweep de lr (#7–#9).
- Si **#1 da < 0.18** → saltear #7–#9; fps_baya queda descartado.

| # | run_name (solo si el gate da OK) | W | LR | sampling |
|---|---|---|---|---|
| 7 | `w8_np512_m0.5_lr1e-3_bs16_seed42_fpsbaya` | 8 | 1e-3 | fps_baya |
| 8 | `w8_np512_m0.5_lr5e-4_bs16_seed42_fpsbaya` | 8 | 5e-4 | fps_baya |
| 9 | `w8_np512_m0.5_lr1e-4_bs16_seed42_fpsbaya` | 8 | 1e-4 | fps_baya |

---

## GATE DE DECISIÓN
Juntar el `test acc` (best) de TODA la Fase 1 + lo ya hecho. Elegir el **top 1–2 configs**.
> El ranking a 200ep es buen predictor pero no perfecto del de 1000ep — por eso se estira el **top 1–2**, no solo el #1.

---

## FASE 2 — Explotación (extender a 1000 epochs con `--resume`)
Para cada ganadora del gate: **reutilizar las 200 epochs ya entrenadas** con `--resume`, extendiendo a 1000. Mismo run_name (continúa desde `checkpoint_last.pt`).

```
python -m src.train --data_dir "D:/thresh105_qr120_umbral008_acomodado" --runs_dir runs --n_points 512 --batch_size 16 --margin 0.5 --seed 42 --width <W> --lr <LR> --sampling <S> --epochs 1000 --resume --run_name <NAME>
```
Luego evaluar igual que siempre.

**Campeón random → convergencia** ("hasta que deje de mejorar"): resumir con cap alto + early stopping.
```
python -m src.train --data_dir "D:/thresh105_qr120_umbral008_acomodado" --runs_dir runs --n_points 512 --batch_size 16 --margin 0.5 --seed 42 --width 8 --lr 3e-4 --sampling random --epochs 2000 --resume --early_stopping_patience 100 --run_name w8_np512_m0.5_lr3e-4_bs16_seed42
```

### ⚠️ Bug conocido del scheduler al resumir (documentado)
`CosineAnnealingLR` guarda `T_max` dentro de su `state_dict`, y `_load_checkpoint` hace `self.scheduler.load_state_dict(...)` (`trainer.py:362`). Eso **restaura el `T_max` viejo y pisa el `--epochs` nuevo**: al resumir 200→1000, el coseno NO se estira a 1000; mantiene período 200 y el lr hace **warm-restarts** (sube y baja, período 2·T_max = 400 epochs) en vez de un decaimiento único.

- **No es necesariamente malo:** así se construyó el campeón (200→400→1000), y esos restarts tipo **SGDR** probablemente ayudaron a bajar de 0.13 a 0.07. Si querés reproducir ese comportamiento, dejá el resume tal cual.
- **Si querés un coseno único** hasta el total nuevo (schedule "de libro"), el fix es: después de `_load_checkpoint`, forzar `self.scheduler.T_max = epochs` (o excluir `T_max` del `state_dict` que se restaura). Así reutilizás los pesos entrenados **y** el lr decae limpio hasta el nuevo total.
- **Decisión para el paper:** elegir UNA de las dos y usarla consistentemente en todas las corridas de Fase 2, para que sean comparables entre sí.

---

## FASE 3 — Completitud (baja prioridad / GPU en la nube)
Grid restante para la tabla del paper. Solo si sobra GPU. Todas 200ep.

| run_name | W | LR | sampling |
|---|---|---|---|
| `w16_np512_m0.5_lr1e-3_bs16_seed42_fps` | 16 | 1e-3 | fps |
| `w32_np512_m0.5_lr1e-3_bs16_seed42_fps` | 32 | 1e-3 | fps |
| `w16_np512_m0.5_lr3e-4_bs16_seed42_fpsbaya` | 16 | 3e-4 | fps_baya |
| `w16_np512_m0.5_lr1e-3_bs16_seed42_fpsbaya` | 16 | 1e-3 | fps_baya |
| `w32_np512_m0.5_lr3e-4_bs16_seed42_fpsbaya` | 32 | 3e-4 | fps_baya |
| `w32_np512_m0.5_lr1e-3_bs16_seed42_fpsbaya` | 32 | 1e-3 | fps_baya |

> Casi seguro confirman lo ya visto (widths grandes y lr1e-3 rinden peor). Ideal para correr en una GPU alquilada mientras la local hace Fase 1/2.

---

## Notas
- Recordá anotar el `test acc` best de cada `runs/<NAME>/ep<E>/evaluation_report.json` en una tabla común para el gate de decisión.
- Una sola GPU = un entrenamiento a la vez (secuencial).
- El track open-set de franco corre en paralelo: validar el split disjunto por identidad + eval open-set sobre los checkpoints existentes (forward pass, casi no usa GPU). El entrenamiento open-set se deja para la config final elegida.
