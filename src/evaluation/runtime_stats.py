"""Utilidades pequeñas para medir recursos sin dependencias externas."""
from __future__ import annotations

import resource
import sys


def peak_process_memory_mb() -> float:
    """Devuelve el máximo RSS del proceso en MiB."""
    maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return maximum_rss / (1024 ** 2)
    return maximum_rss / 1024.0
