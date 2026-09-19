"""Stres testi — eşzamanlı arama ve dayanıklılık."""

from __future__ import annotations

import threading
import time
from typing import Any

from core.logger import setup_logger
from core.search_engine import SearchEngine

logger = setup_logger(__name__)


def run_stress_test(
    settings,
    *,
    search_count: int = 1000,
    query_terms: list[str] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """1000 eşzamanlı arama simülasyonu (UI thread dışında çalıştırın)."""
    terms = query_terms or ["lv", "floral", "geometric", "gucci", "kahverengi"]
    engine = SearchEngine(settings)
    errors: list[str] = []
    lock = threading.Lock()
    completed = 0
    latencies: list[float] = []

    def _one_search(i: int) -> None:
        nonlocal completed
        term = terms[i % len(terms)]
        t0 = time.perf_counter()
        try:
            engine.search_by_text(term, limit=20)
        except Exception as exc:
            with lock:
                errors.append(str(exc))
        else:
            with lock:
                latencies.append(time.perf_counter() - t0)
        with lock:
            completed += 1
            if progress_callback and completed % 50 == 0:
                progress_callback({"done": completed, "total": search_count})

    workers = min(32, max(4, search_count // 25))
    batch = search_count // workers
    threads: list[threading.Thread] = []
    idx = 0
    for _ in range(workers):
        count = batch if idx + batch <= search_count else search_count - idx
        if count <= 0:
            break

        def _worker(start: int, n: int) -> None:
            for j in range(n):
                _one_search(start + j)

        t = threading.Thread(target=_worker, args=(idx, count), daemon=True)
        threads.append(t)
        idx += count
        t.start()

    for t in threads:
        t.join(timeout=120)

    avg_ms = round(1000 * sum(latencies) / len(latencies), 1) if latencies else 0
    result = {
        "search_count": search_count,
        "completed": completed,
        "errors": len(errors),
        "error_samples": errors[:5],
        "avg_latency_ms": avg_ms,
        "crashed": False,
        "ok": len(errors) == 0 and completed >= search_count,
    }
    logger.info("Stress test: %s", {k: v for k, v in result.items() if k != "error_samples"})
    return result
