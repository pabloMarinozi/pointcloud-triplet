from __future__ import annotations

import csv
import os
import threading
import time
from typing import Optional

import psutil

_GPU_AVAILABLE = False
try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_DEVICE_COUNT = pynvml.nvmlDeviceGetCount()
    if _GPU_DEVICE_COUNT > 0:
        _GPU_AVAILABLE = True
except Exception:
    _GPU_DEVICE_COUNT = 0


class SystemMonitor:
    def __init__(self, csv_path: str, interval: float = 2.0):
        self._csv_path = csv_path
        self._interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._t0: float = 0.0
        self._header_written = False

    def _write_header(self) -> None:
        header = ["timestamp", "elapsed_s", "cpu_percent", "ram_used_gb", "ram_percent"]
        if _GPU_AVAILABLE:
            for i in range(_GPU_DEVICE_COUNT):
                header.append(f"gpu{i}_util_percent")
                header.append(f"gpu{i}_mem_used_gb")
                header.append(f"gpu{i}_mem_percent")
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        self._header_written = True

    def _sample(self) -> None:
        elapsed = time.perf_counter() - self._t0
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024**3)
        ram_percent = mem.percent

        row = [
            time.strftime("%H:%M:%S"),
            round(elapsed, 1),
            cpu,
            round(ram_used_gb, 2),
            round(ram_percent, 1),
        ]

        if _GPU_AVAILABLE:
            for i in range(_GPU_DEVICE_COUNT):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                row.append(util.gpu)
                row.append(round(mem_info.used / (1024**3), 2))
                row.append(round(mem_info.used / mem_info.total * 100, 1))

        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._write_header()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 2)
        self._sample()
        self._thread = None

    def _shutdown_nvml(self) -> None:
        if _GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

