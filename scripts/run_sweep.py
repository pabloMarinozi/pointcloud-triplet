"""
Script para ejecutar un sweep de hiperparámetros de manera secuencial.

Cada experimento:
1. Genera un run_name determinístico basado en los hiperparámetros
2. Verifica si el run ya existe y está completo (skip si está completo)
3. Ejecuta el entrenamiento
4. Ejecuta la evaluación automáticamente
5. Registra resultados en sweep_summary.csv

El script puede reanudarse: si se corta, al relanzarlo continúa desde donde quedó.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch


@dataclass
class ExperimentConfig:
    """Configuración de un experimento individual."""

    n_points: int
    width: int
    margin: float
    lr: float
    batch_size: int
    epochs: int
    seed: int = 42
    clip_norm: float = 1.0
    val_size: float = 0.2

    def to_run_name(self) -> str:
        """Genera un nombre determinístico para el run basado en los hiperparámetros."""
        # Formato: w{width}_np{n_points}_m{margin}_lr{lr}_bs{batch_size}_seed{seed}
        lr_str = f"{self.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+").replace(".0e", "e")
        return f"w{self.width}_np{self.n_points}_m{self.margin}_lr{lr_str}_bs{self.batch_size}_seed{self.seed}"

    def to_dict(self) -> Dict:
        """Convierte la configuración a diccionario para logging."""
        return {
            "n_points": self.n_points,
            "width": self.width,
            "margin": self.margin,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "clip_norm": self.clip_norm,
            "val_size": self.val_size,
        }


def define_experiments() -> List[ExperimentConfig]:
    """
    Define la lista de experimentos a ejecutar.
    Modifica esta función para cambiar los hiperparámetros del sweep.
    """
    experiments = []

    # Screening inicial: 50 épocas
    for n_points in [512, 1024, 2048, 4096]:
        for width in [8, 16, 32, 64]:
            for lr in [1e-3, 3e-4]:
                experiments.append(
                    ExperimentConfig(
                        n_points=n_points,
                        width=width,
                        margin=0.5,
                        lr=lr,
                        batch_size=16,
                        epochs=50,
                        seed=42,
                    )
                )

    # TODO: Después del screening, agregar aquí los mejores con epochs=200
    # Ejemplo:
    # experiments.append(
    #     ExperimentConfig(
    #         n_points=2048,
    #         width=64,
    #         margin=0.5,
    #         lr=3e-4,
    #         batch_size=16,
    #         epochs=200,
    #         seed=42,
    #     )
    # )

    return experiments


def is_run_complete(runs_dir: str, run_name: str) -> bool:
    """
    Verifica si un run está completo.
    Un run se considera completo si tiene:
    - model_best.pt
    - reference_embeddings_train.npz
    - config.json
    """
    exp_dir = os.path.join(runs_dir, run_name)
    if not os.path.isdir(exp_dir):
        return False

    required_files = [
        "model_best.pt",
        "reference_embeddings_train.npz",
        "config.json",
    ]

    for fname in required_files:
        if not os.path.isfile(os.path.join(exp_dir, fname)):
            return False

    return True


def has_checkpoint(runs_dir: str, run_name: str) -> bool:
    """
    Verifica si existe un checkpoint intermedio (checkpoint_last.pt).
    Esto indica que el entrenamiento puede reanudarse.
    """
    checkpoint_path = os.path.join(runs_dir, run_name, "checkpoint_last.pt")
    return os.path.isfile(checkpoint_path)


def get_checkpoint_epoch(runs_dir: str, run_name: str) -> Optional[int]:
    """
    Extrae la época del checkpoint si existe.
    Retorna None si no hay checkpoint o no se puede leer.
    """
    checkpoint_path = os.path.join(runs_dir, run_name, "checkpoint_last.pt")
    if not os.path.isfile(checkpoint_path):
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return int(checkpoint.get("epoch", 0))
    except Exception:
        return None


def get_best_val_loss(runs_dir: str, run_name: str) -> Optional[float]:
    """Extrae el mejor val_loss del metrics.csv."""
    metrics_path = os.path.join(runs_dir, run_name, "metrics.csv")
    if not os.path.isfile(metrics_path):
        return None

    try:
        df = pd.read_csv(metrics_path)
        if "val_loss" in df.columns and len(df) > 0:
            return float(df["val_loss"].min())
    except Exception:
        pass

    return None


def get_best_eval_accuracy(runs_dir: str, run_name: str) -> Optional[float]:
    """
    Extrae la mejor accuracy de evaluación.
    Busca en los CSVs de evaluación y retorna la mejor accuracy encontrada.
    """
    eval_dir = os.path.join(runs_dir, run_name, "evaluation")
    if not os.path.isdir(eval_dir):
        return None

    best_acc = None

    # Buscar archivos CSV de evaluación
    for fname in os.listdir(eval_dir):
        if not fname.endswith(".csv") or "validation_predictions" not in fname:
            continue

        csv_path = os.path.join(eval_dir, fname)
        try:
            df = pd.read_csv(csv_path)
            if "correct" in df.columns and len(df) > 0:
                acc = float(df["correct"].mean())
                if best_acc is None or acc > best_acc:
                    best_acc = acc
        except Exception:
            continue

    return best_acc


def monitor_training_progress(runs_dir: str, run_name: str, total_epochs: int, stop_event: threading.Event):
    """
    Monitorea el progreso del entrenamiento leyendo metrics.csv periódicamente.
    Muestra un resumen del progreso cada 1 minuto.
    """
    metrics_path = os.path.join(runs_dir, run_name, "metrics.csv")
    last_epoch = 0
    
    while not stop_event.is_set():
        time.sleep(60)  # Verificar cada 1 minuto
        
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path)
                if len(df) > 0 and "epoch" in df.columns:
                    current_epoch = int(df["epoch"].max())
                    if current_epoch > last_epoch:
                        last_epoch = current_epoch
                        progress_pct = (current_epoch / total_epochs) * 100
                        bar_length = 30
                        filled = int(bar_length * current_epoch / total_epochs)
                        bar = "=" * filled + "-" * (bar_length - filled)
                        
                        # Obtener última métrica si está disponible
                        last_row = df.iloc[-1]
                        val_loss = last_row.get("val_loss", "N/A")
                        train_loss = last_row.get("train_loss", "N/A")
                        
                        print(
                            f"\r[PROGRESO] Epoca {current_epoch}/{total_epochs} ({progress_pct:.1f}%) "
                            f"[{bar}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
                            end="",
                            flush=True,
                        )
            except Exception:
                pass
    
    # Limpiar la línea de progreso
    print("\r" + " " * 100 + "\r", end="", flush=True)


def run_training(config: ExperimentConfig, data_dir: str, runs_dir: str, run_name: str, resume: bool = False) -> bool:
    """
    Ejecuta el entrenamiento como subprocess. Retorna True si fue exitoso.
    Muestra progreso en tiempo real.
    
    Args:
        resume: Si True, pasa --resume para reanudar desde checkpoint.
    """
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--data_dir",
        data_dir,
        "--runs_dir",
        runs_dir,
        "--run_name",
        run_name,
        "--n_points",
        str(config.n_points),
        "--width",
        str(config.width),
        "--batch_size",
        str(config.batch_size),
        "--epochs",
        str(config.epochs),
        "--lr",
        str(config.lr),
        "--margin",
        str(config.margin),
        "--seed",
        str(config.seed),
        "--clip_norm",
        str(config.clip_norm),
        "--val_size",
        str(config.val_size),
    ]

    if resume:
        cmd.append("--resume")

    print(f"\n{'='*80}")
    print(f"ENTRENANDO: {run_name}")
    print(f"{'='*80}")
    print(f"Config: epochs={config.epochs}, n_points={config.n_points}, width={config.width}, lr={config.lr}")
    print(f"Comando: {' '.join(cmd)}")
    print(f"\nProgreso del entrenamiento:")
    print("-" * 80)

    # Iniciar monitoreo de progreso en un thread separado
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_training_progress,
        args=(runs_dir, run_name, config.epochs, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    try:
        # Ejecutar el proceso y capturar salida en tiempo real
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Mostrar salida en tiempo real
        for line in process.stdout:
            # Filtrar líneas relevantes para mostrar
            line = line.rstrip()
            if line and ("epoch" in line.lower() or "best" in line.lower() or "error" in line.lower() or "warning" in line.lower()):
                print(f"  {line}")

        # Esperar a que termine el proceso
        return_code = process.wait()
        
        # Detener el monitor
        stop_event.set()
        monitor_thread.join(timeout=1)
        
        print("-" * 80)
        
        if return_code == 0:
            print(f"[OK] Entrenamiento completado: {run_name}")
            return True
        else:
            print(f"[FAIL] Entrenamiento falló con código {return_code}: {run_name}")
            return False
            
    except subprocess.CalledProcessError as e:
        stop_event.set()
        print(f"[FAIL] ERROR en entrenamiento de {run_name}: {e}")
        return False
    except Exception as e:
        stop_event.set()
        print(f"[FAIL] ERROR inesperado en entrenamiento de {run_name}: {e}")
        return False


def run_evaluation(data_dir: str, runs_dir: str, run_name: str) -> bool:
    """Ejecuta la evaluación como subprocess. Retorna True si fue exitoso."""
    cmd = [
        sys.executable,
        "-m",
        "src.eval",
        "--data_dir",
        data_dir,
        "--runs_dir",
        runs_dir,
        "--run",
        run_name,
        "--export_csv",
    ]

    print(f"\n{'='*80}")
    print(f"EVALUANDO: {run_name}")
    print(f"{'='*80}")
    print(f"Comando: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR en evaluación de {run_name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR inesperado en evaluación de {run_name}: {e}")
        return False


def update_sweep_summary(
    runs_dir: str,
    experiments: List[ExperimentConfig],
    summary_path: str,
):
    """
    Genera o actualiza el sweep_summary.csv con información de todos los experimentos.
    """
    rows = []

    for config in experiments:
        run_name = config.to_run_name()
        exp_dir = os.path.join(runs_dir, run_name)
        row = {
            "run_name": run_name,
            "n_points": config.n_points,
            "width": config.width,
            "margin": config.margin,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "seed": config.seed,
            "status": "pending",
            "checkpoint_epoch": None,
            "best_val_loss": None,
            "best_eval_accuracy": None,
            "model_path": None,
            "config_path": None,
            "eval_dir": None,
        }

        if os.path.isdir(exp_dir):
            if is_run_complete(runs_dir, run_name):
                row["status"] = "complete"
            elif has_checkpoint(runs_dir, run_name):
                row["status"] = "incomplete (resumable)"
                checkpoint_epoch = get_checkpoint_epoch(runs_dir, run_name)
                if checkpoint_epoch is not None:
                    row["checkpoint_epoch"] = checkpoint_epoch
            else:
                row["status"] = "incomplete"

            best_val = get_best_val_loss(runs_dir, run_name)
            if best_val is not None:
                row["best_val_loss"] = best_val

            best_acc = get_best_eval_accuracy(runs_dir, run_name)
            if best_acc is not None:
                row["best_eval_accuracy"] = best_acc

            model_path = os.path.join(exp_dir, "model_best.pt")
            if os.path.isfile(model_path):
                row["model_path"] = os.path.abspath(model_path)

            config_path = os.path.join(exp_dir, "config.json")
            if os.path.isfile(config_path):
                row["config_path"] = os.path.abspath(config_path)

            eval_dir = os.path.join(exp_dir, "evaluation")
            if os.path.isdir(eval_dir):
                row["eval_dir"] = os.path.abspath(eval_dir)

        rows.append(row)

    # Guardar CSV
    df = pd.DataFrame(rows)
    df = df.sort_values(["best_eval_accuracy", "best_val_loss"], ascending=[False, True], na_position="last")
    df.to_csv(summary_path, index=False, float_format="%.6f")

    print(f"\n{'='*80}")
    print(f"RESUMEN ACTUALIZADO: {summary_path}")
    print(f"{'='*80}")
    print(f"Total experimentos: {len(rows)}")
    print(f"Completos: {sum(1 for r in rows if r['status'] == 'complete')}")
    print(f"Incompletos: {sum(1 for r in rows if r['status'] == 'incomplete')}")
    print(f"Pendientes: {sum(1 for r in rows if r['status'] == 'pending')}")

    # Mostrar top 5 por accuracy
    df_complete = df[df["status"] == "complete"].copy()
    if len(df_complete) > 0:
        print("\nTop 5 por accuracy:")
        top5 = df_complete.head(5)
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            acc = row["best_eval_accuracy"] if pd.notna(row["best_eval_accuracy"]) else "N/A"
            val_loss = row["best_val_loss"] if pd.notna(row["best_val_loss"]) else "N/A"
            print(f"  {idx}. {row['run_name']} | acc={acc} | val_loss={val_loss}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ejecuta un sweep de hiperparámetros.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directorio raíz con .ply (recursivo).")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Carpeta donde guardar experimentos.")
    parser.add_argument("--skip_complete", action="store_true", help="Skip runs que ya están completos.")
    parser.add_argument("--skip_eval", action="store_true", help="No ejecutar evaluación automáticamente.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    runs_dir = args.runs_dir
    skip_complete = args.skip_complete
    skip_eval = args.skip_eval

    # Crear runs_dir si no existe
    os.makedirs(runs_dir, exist_ok=True)

    # Definir experimentos
    experiments = define_experiments()
    print(f"\n{'='*80}")
    print(f"SWEEP DE HIPERPARÁMETROS")
    print(f"{'='*80}")
    print(f"Total experimentos: {len(experiments)}")
    print(f"Data dir: {data_dir}")
    print(f"Runs dir: {runs_dir}")
    print(f"Skip completos: {skip_complete}")
    print(f"Skip evaluación: {skip_eval}")

    # Path del resumen
    summary_path = os.path.join(runs_dir, "sweep_summary.csv")

    # Ejecutar experimentos secuencialmente
    completed = 0
    skipped = 0
    failed = 0

    for idx, config in enumerate(experiments, 1):
        run_name = config.to_run_name()

        print(f"\n{'='*80}")
        print(f"EXPERIMENTO {idx}/{len(experiments)}: {run_name}")
        print(f"{'='*80}")

        # Verificar si está completo
        if is_run_complete(runs_dir, run_name):
            if skip_complete:
                print(f"[OK] Run completo, saltando: {run_name}")
                skipped += 1
                update_sweep_summary(runs_dir, experiments, summary_path)
                continue
            else:
                print(f"[WARN] Run completo, pero skip_complete=False. Reentrenando...")

        # Verificar si hay checkpoint para reanudar
        should_resume = has_checkpoint(runs_dir, run_name) and not is_run_complete(runs_dir, run_name)
        if should_resume:
            checkpoint_epoch = get_checkpoint_epoch(runs_dir, run_name)
            if checkpoint_epoch is not None:
                print(f"[RESUME] Detectado checkpoint en época {checkpoint_epoch}. Reanudando entrenamiento...")
            else:
                print(f"[RESUME] Detectado checkpoint. Reanudando entrenamiento...")

        # Ejecutar entrenamiento
        train_success = run_training(config, data_dir, runs_dir, run_name, resume=should_resume)

        if not train_success:
            print(f"[FAIL] FALLO entrenamiento: {run_name}")
            failed += 1
            update_sweep_summary(runs_dir, experiments, summary_path)
            continue

        # Verificar que el entrenamiento produjo los archivos necesarios
        if not is_run_complete(runs_dir, run_name):
            print(f"[WARN] Entrenamiento terminó pero el run está incompleto: {run_name}")
            failed += 1
            update_sweep_summary(runs_dir, experiments, summary_path)
            continue

        # Ejecutar evaluación
        if not skip_eval:
            eval_success = run_evaluation(data_dir, runs_dir, run_name)
            if not eval_success:
                print(f"[WARN] Evaluación falló para {run_name}, pero continuando...")

        completed += 1
        print(f"[OK] Completado: {run_name}")

        # Actualizar resumen después de cada experimento
        update_sweep_summary(runs_dir, experiments, summary_path)

    # Resumen final
    print(f"\n{'='*80}")
    print(f"SWEEP FINALIZADO")
    print(f"{'='*80}")
    print(f"Completados: {completed}")
    print(f"Saltados: {skipped}")
    print(f"Fallidos: {failed}")
    print(f"Total: {len(experiments)}")
    print(f"\nResumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
