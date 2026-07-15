"""Ejecuta secuencialmente todos los lanzadores de entrenamiento Python."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def write_line(log_file, message: str) -> None:
    """Muestra una línea y la agrega al log general."""
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    training_scripts_dir = project_root / "experiments" / "scripts"
    training_scripts = sorted(training_scripts_dir.glob("run_*.py"))

    if not training_scripts:
        print(f"No se encontraron entrenamientos en {training_scripts_dir}.")
        return 1

    now = datetime.now()
    log_dir = project_root / "experiments" / "logs" / now.strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_all_{now.strftime('%H-%M-%S')}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    failures: list[tuple[str, int]] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        write_line(log_file, f"Inicio: {now.isoformat(timespec='seconds')}")
        write_line(log_file, f"Entrenamientos: {len(training_scripts)}")

        for index, script in enumerate(training_scripts, start=1):
            write_line(
                log_file,
                f"\n[{index}/{len(training_scripts)}] Ejecutando {script.name}...",
            )

            process = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()
            if return_code != 0:
                failures.append((script.name, return_code))
                write_line(
                    log_file,
                    f"[ERROR] {script.name} terminó con código {return_code}.",
                )
                write_line(log_file, "Continuando con el siguiente entrenamiento...")
                continue

            write_line(log_file, f"[OK] {script.name}")

        if failures:
            write_line(log_file, "\nEntrenamientos con errores:")
            for script_name, return_code in failures:
                write_line(log_file, f"- {script_name}: código {return_code}")
        else:
            write_line(log_file, "\nTodos los entrenamientos terminaron correctamente.")
        write_line(log_file, f"Log general: {log_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
