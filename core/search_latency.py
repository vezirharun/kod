"""Görsel arama latency profiler — ranking davranışını değiştirmez."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator


def classify_storage(path: str) -> str:
    """local | nas | removable | missing | unknown — Windows drive type."""
    raw = str(path or "").strip()
    if not raw:
        return "unknown"
    if not os.path.exists(raw):
        return "missing"
    if raw.startswith("\\\\") or raw.startswith("//"):
        return "nas"
    drive = os.path.splitdrive(raw)[0]
    if not drive:
        return "local"
    try:
        import ctypes

        kind = int(ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\"))
    except Exception:
        return "local"
    if kind == 4:
        return "nas"
    if kind == 2:
        return "removable"
    return "local"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_ms(values: list[float]) -> dict[str, float]:
    xs = [float(v) for v in values]
    if not xs:
        return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0, "n": 0}
    return {
        "n": len(xs),
        "min": round(min(xs), 3),
        "max": round(max(xs), 3),
        "avg": round(sum(xs) / len(xs), 3),
        "p50": round(percentile(xs, 50), 3),
        "p95": round(percentile(xs, 95), 3),
        "p99": round(percentile(xs, 99), 3),
    }


def classify_budget(ms: float) -> str:
    if ms < 200:
        return "MUKEMMEL"
    if ms < 500:
        return "COK_IYI"
    if ms < 1000:
        return "IYI"
    if ms < 2000:
        return "KABUL_EDILEBILIR"
    if ms < 5000:
        return "YAVAS"
    return "KRITIK"


class NullLatency:
    enabled = False

    def reset(self) -> None:
        return None

    def mark_start(self) -> None:
        return None

    @contextmanager
    def span(self, _name: str) -> Iterator[None]:
        yield

    def add(self, _name: str, _ms: float) -> None:
        return None

    def set(self, _key: str, _value: Any) -> None:
        return None

    def incr(self, _key: str, _n: int = 1) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {}


@dataclass
class SearchLatencyProfiler:
    """Stage timers + counts. Enable only during benchmark runs."""

    enabled: bool = True
    t0: float = 0.0
    spans_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.t0 = perf_counter()
        self.spans_ms = {}
        self.counts = {}

    def mark_start(self) -> None:
        if not self.t0:
            self.t0 = perf_counter()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t = perf_counter()
        try:
            yield
        finally:
            self.spans_ms[name] = self.spans_ms.get(name, 0.0) + (perf_counter() - t) * 1000.0

    def add(self, name: str, ms: float) -> None:
        self.spans_ms[name] = self.spans_ms.get(name, 0.0) + float(ms)

    def set(self, key: str, value: Any) -> None:
        self.counts[key] = value

    def incr(self, key: str, n: int = 1) -> None:
        self.counts[key] = int(self.counts.get(key) or 0) + int(n)

    def elapsed_ms(self) -> float:
        if not self.t0:
            return 0.0
        return (perf_counter() - self.t0) * 1000.0

    def snapshot(self) -> dict[str, Any]:
        spans = {k: round(v, 3) for k, v in self.spans_ms.items()}
        total = round(self.elapsed_ms(), 3)
        ranked = sorted(spans.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "total_ms": total,
            "budget": classify_budget(total),
            "spans_ms": spans,
            "counts": dict(self.counts),
            "top_bottlenecks": [
                {"stage": k, "ms": v, "pct": round(100.0 * v / total, 1) if total else 0.0}
                for k, v in ranked[:10]
            ],
        }
