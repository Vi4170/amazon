"""Laptop protection: cap polars threads and abort the process if RSS exceeds ER_MAX_GB (default 9)."""
import os
import threading
import time

os.environ.setdefault("POLARS_MAX_THREADS", "6")

import psutil  # noqa: E402

PEAK = [0.0]


def _watch(limit_gb: float) -> None:
    p = psutil.Process(os.getpid())
    while True:
        gb = p.memory_info().rss / 1e9
        PEAK[0] = max(PEAK[0], gb)
        if gb > limit_gb:
            print(f"[guard] RSS {gb:.1f} GB > {limit_gb} GB limit - aborting to protect the laptop", flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=_watch, args=(float(os.environ.get("ER_MAX_GB", "9")),), daemon=True).start()
