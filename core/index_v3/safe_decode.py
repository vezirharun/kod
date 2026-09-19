"""Index Engine V3 — riskli decode için hard-timeout (legacy Indexer state yok)."""

from __future__ import annotations

import multiprocessing as mp
import queue as py_queue
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, TypeVar

from core.network_index_throttle import is_network_path
from core.utils import normalize_path

T = TypeVar("T")

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_RETRIES = 2

# Tek paylaşımlı pool: dosya başına ThreadPoolExecutor + shutdown(wait=False)
# terk edilen native decode thread'lerini biriktiriyordu (100 → 1300+).
_POOL_LOCK = threading.Lock()
_TIMEOUT_POOL: ThreadPoolExecutor | None = None
_ABANDONED = 0
_ABANDON_RESET_AT = 8
_POOL_WORKERS = 2


def is_risky_path(path: str) -> bool:
    p = normalize_path(path)
    ext = Path(p).suffix.lower()
    if ext in (".tif", ".tiff"):
        return True
    try:
        return bool(is_network_path(p))
    except Exception:
        return p.startswith("\\\\") or p.startswith("//")


def _get_timeout_pool() -> ThreadPoolExecutor:
    global _TIMEOUT_POOL
    with _POOL_LOCK:
        if _TIMEOUT_POOL is None:
            _TIMEOUT_POOL = ThreadPoolExecutor(
                max_workers=_POOL_WORKERS,
                thread_name_prefix="v3-timeout",
            )
        return _TIMEOUT_POOL


def _reset_timeout_pool(*, reason: str = "") -> None:
    """Takılı native decode worker'larını sınırlı bırak; yeni pool aç."""
    global _TIMEOUT_POOL, _ABANDONED
    with _POOL_LOCK:
        old = _TIMEOUT_POOL
        _TIMEOUT_POOL = ThreadPoolExecutor(
            max_workers=_POOL_WORKERS,
            thread_name_prefix="v3-timeout",
        )
        _ABANDONED = 0
    if old is not None:
        try:
            old.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _close_mp_queue(q: Any) -> None:
    try:
        q.close()
    except Exception:
        pass
    try:
        q.join_thread()
    except Exception:
        pass


def run_with_timeout(
    fn: Callable[[], T],
    *,
    timeout_sec: float = _DEFAULT_TIMEOUT,
    retries: int = _DEFAULT_RETRIES,
    use_process: bool = False,
    retry_on_timeout: bool = False,
) -> T:
    """Hard timeout. Process izolasyonu riskli TIFF/UNC için.

    Hard timeout (FuturesTimeout / TimeoutError) default olarak tekrarlanmaz:
    aynı dosyada N×timeout worker kilidi (~3×45s) üretiyordu. Geçici hatalar
    için retries hâlâ geçerli; timeout retry yalnız retry_on_timeout=True iken.
    """
    global _ABANDONED
    last_exc: BaseException | None = None
    attempts = max(1, int(retries) + 1)
    for _ in range(attempts):
        try:
            if use_process:
                return _run_in_process(fn, timeout_sec=timeout_sec)
            pool = _get_timeout_pool()
            fut = pool.submit(fn)
            try:
                return fut.result(timeout=max(0.5, float(timeout_sec)))
            except FuturesTimeout as exc:
                with _POOL_LOCK:
                    _ABANDONED += 1
                    abandoned = _ABANDONED
                if abandoned >= _ABANDON_RESET_AT:
                    _reset_timeout_pool(reason="timeout_abandon_cap")
                last_exc = TimeoutError(f"v3_decode_timeout:{timeout_sec}s")
                last_exc.__cause__ = exc
                if not retry_on_timeout:
                    break
                continue
        except TimeoutError as exc:
            last_exc = exc
            if not retry_on_timeout:
                break
        except Exception as exc:
            last_exc = exc
            # non-timeout: one shot fail for permanent errors
            if "timeout" not in str(exc).lower():
                raise
    assert last_exc is not None
    raise last_exc


def _process_entry(q: Any, fn: Callable[[], T]) -> None:
    try:
        q.put(("ok", fn()))
    except BaseException as exc:
        q.put(("err", repr(exc)))


def _run_in_process(fn: Callable[[], T], *, timeout_sec: float) -> T:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_process_entry, args=(q, fn))
    try:
        proc.start()
        proc.join(max(0.5, float(timeout_sec)))
        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
            raise TimeoutError(f"v3_decode_timeout:{timeout_sec}s")
        # Queue.empty() is not reliable across processes on Windows. A successful
        # child can exit before the feeder thread makes the item visible, which
        # falsely turns a valid decode into a retry/permanent failure.
        try:
            kind, payload = q.get(timeout=2.0)
        except py_queue.Empty as exc:
            raise RuntimeError(
                f"v3_decode_subprocess_exit:{proc.exitcode}"
            ) from exc
        if kind == "ok":
            return payload
        raise RuntimeError(str(payload))
    finally:
        _close_mp_queue(q)
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
        except Exception:
            pass
        try:
            proc.close()
        except Exception:
            pass
