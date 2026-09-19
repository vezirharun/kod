"""NAS / UNC kaynaklarında eşzamanlı dosya okuma sınırı ve IO sayaçları."""

from __future__ import annotations

import os
import threading
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from core.logger import setup_logger

logger = setup_logger(__name__)

_lock = threading.Lock()
_semaphores: dict[int, threading.Semaphore] = {}
_log_lock = threading.Lock()
_MAX_OPEN_LOG = 20
_UNSET = object()
_tls = threading.local()

T = TypeVar("T")


def bind_deep_from_cache(value: bool) -> None:
    """Pass-scoped guard; dual pipeline'da shared stats yerine thread-local kullan."""
    _tls.deep_from_cache = bool(value)


def unbind_deep_from_cache() -> None:
    if hasattr(_tls, "deep_from_cache"):
        delattr(_tls, "deep_from_cache")


@contextmanager
def deep_from_cache_scope(value: bool) -> Iterator[None]:
    prev = getattr(_tls, "deep_from_cache", _UNSET)
    _tls.deep_from_cache = bool(value)
    try:
        yield
    finally:
        if prev is _UNSET:
            unbind_deep_from_cache()
        else:
            _tls.deep_from_cache = prev


def resolve_deep_from_cache(
    stats: dict | None = None,
    deep_from_cache: bool | None = None,
) -> bool | None:
    """Öncelik: explicit arg > thread-local > stats (geriye uyumluluk)."""
    if deep_from_cache is not None:
        return bool(deep_from_cache)
    tls_val = getattr(_tls, "deep_from_cache", _UNSET)
    if tls_val is not _UNSET:
        return bool(tls_val)
    if stats is not None and "_deep_from_cache" in stats:
        return bool(stats.get("_deep_from_cache"))
    return None

NETWORK_METADATA_KEY = "network_metadata_check_count"
NETWORK_ORIGINAL_KEY = "network_original_open_count"
NETWORK_PREVIEW_KEY = "network_preview_open_count"
NETWORK_TOTAL_KEY = "network_read_count"
LOCAL_CACHE_KEY = "local_cache_read_count"
DB_METADATA_KEY = "db_metadata_read_count"
ORIGINAL_FILE_OPEN_KEY = "original_file_open_count"
NETWORK_OPEN_LOG_KEY = "network_open_log"


def is_network_path(path: str | None) -> bool:
    """UNC / SMB. ``\\\\?\\C:\\...`` yerel extended path'tir, ağ değildir."""
    if not path:
        return False
    from core.utils import strip_extended_path_prefix

    norm = strip_extended_path_prefix(str(path)).replace("/", "\\")
    if not (norm.startswith("\\\\") or norm.startswith("//")):
        return False
    body = norm.lstrip("\\/")
    # ``C:\...`` bu noktaya düşmemeli; yine de drive harfini ağ sayma.
    if len(body) >= 2 and body[1] == ":":
        return False
    return True


def bump_io_stat(stats: dict | None, key: str, amount: int = 1) -> None:
    if stats is None:
        return
    stats[key] = int(stats.get(key, 0)) + amount
    if key in (NETWORK_METADATA_KEY, NETWORK_ORIGINAL_KEY, NETWORK_PREVIEW_KEY):
        stats[NETWORK_TOTAL_KEY] = (
            int(stats.get(NETWORK_METADATA_KEY, 0))
            + int(stats.get(NETWORK_ORIGINAL_KEY, 0))
            + int(stats.get(NETWORK_PREVIEW_KEY, 0))
        )


def _pass_counter_key(stats: dict | None, base_key: str) -> str:
    pass_name = str((stats or {}).get("index_pass") or "session")
    return f"{pass_name}_{base_key}"


def _caller_chain(skip: int = 3, depth: int = 5) -> str:
    frames = traceback.extract_stack()[:-skip][-depth:]
    return " <- ".join(f"{f.filename.rsplit(os.sep, 1)[-1]}:{f.name}:{f.lineno}" for f in frames)


def _record_network_open(
    stats: dict | None,
    *,
    path: str,
    reason: str,
    kind: str,
    deep_from_cache: bool | None = None,
) -> None:
    if stats is None:
        return
    entry: dict[str, Any] = {
        "kind": kind,
        "path": path,
        "reason": reason or "unspecified",
        "index_pass": str(stats.get("index_pass") or "session"),
        "deep_from_cache": deep_from_cache,
        "caller": _caller_chain(),
    }
    with _log_lock:
        log = list(stats.get(NETWORK_OPEN_LOG_KEY) or [])
        if len(log) < _MAX_OPEN_LOG:
            log.append(entry)
            stats[NETWORK_OPEN_LOG_KEY] = log
    logger.warning(
        "[NETWORK %s OPEN] pass=%s deep_from_cache=%s reason=%s path=%s caller=%s",
        kind.upper(),
        entry["index_pass"],
        deep_from_cache,
        entry["reason"],
        path,
        entry["caller"],
    )


def track_local_cache_read(stats: dict | None, amount: int = 1) -> None:
    bump_io_stat(stats, LOCAL_CACHE_KEY, amount)


def track_db_metadata_read(stats: dict | None, amount: int = 1) -> None:
    bump_io_stat(stats, DB_METADATA_KEY, amount)


def _semaphore_for(limit: int) -> threading.Semaphore:
    limit = max(1, int(limit or 1))
    with _lock:
        sem = _semaphores.get(limit)
        if sem is None:
            sem = threading.Semaphore(limit)
            _semaphores[limit] = sem
        return sem


def _network_limit(settings) -> int:
    try:
        from core.work_profile import effective_network_workers

        return max(1, int(effective_network_workers(settings)))
    except Exception:
        return max(1, int(getattr(settings, "max_network_workers", 2) or 2))


@contextmanager
def _network_throttle(settings) -> Iterator[None]:
    sem = _semaphore_for(_network_limit(settings))
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def _bump_network_open_stats(
    stats: dict | None,
    *,
    base_key: str,
    path: str,
    reason: str,
    kind: str,
    deep_from_cache: bool | None = None,
) -> None:
    bump_io_stat(stats, base_key)
    bump_io_stat(stats, _pass_counter_key(stats, base_key))
    if base_key == NETWORK_ORIGINAL_KEY:
        bump_io_stat(stats, ORIGINAL_FILE_OPEN_KEY)
        bump_io_stat(stats, _pass_counter_key(stats, ORIGINAL_FILE_OPEN_KEY))
        # Geriye uyumluluk: oturum genelinde orijinal dosya açıldı sayacı
        bump_io_stat(stats, "original_reopened")
        bump_io_stat(stats, _pass_counter_key(stats, "original_reopened"))
    _record_network_open(
        stats,
        path=path,
        reason=reason,
        kind=kind,
        deep_from_cache=deep_from_cache,
    )


@contextmanager
def network_original_open_slot(
    settings,
    path: str,
    stats: dict | None = None,
    *,
    reason: str = "",
    deep_from_cache: bool | None = None,
) -> Iterator[None]:
    """Orijinal dosya açma (metadata, thumbnail, hash) — ağ yollarında throttle."""
    deep_from_cache = resolve_deep_from_cache(stats, deep_from_cache)
    if not is_network_path(path):
        yield
        return
    if deep_from_cache:
        logger.error(
            "Deep pass ağ orijinal açma ihlali: reason=%s path=%s caller=%s",
            reason,
            path,
            _caller_chain(),
        )
        bump_io_stat(stats, "deep_original_open_violation_count")
        bump_io_stat(stats, _pass_counter_key(stats, "deep_original_open_violation_count"))
        _record_network_open(
            stats,
            path=path,
            reason=f"VIOLATION:{reason}",
            kind="original",
            deep_from_cache=True,
        )
        # Heavy/deep pass NAS orijinaline dokunmamalı. Bu noktaya geldiysek bu bir bug.
        raise RuntimeError(
            f"Deep pass attempted to open network original: reason={reason} path={path}"
        )
    with _network_throttle(settings):
        _bump_network_open_stats(
            stats,
            base_key=NETWORK_ORIGINAL_KEY,
            path=path,
            reason=reason,
            kind="original",
            deep_from_cache=deep_from_cache,
        )
        yield


@contextmanager
def network_preview_open_slot(
    settings,
    path: str,
    stats: dict | None = None,
    *,
    reason: str = "",
    deep_from_cache: bool | None = None,
) -> Iterator[None]:
    """Önizleme üretimi / render — ağ yollarında throttle."""
    deep_from_cache = resolve_deep_from_cache(stats, deep_from_cache)
    if not is_network_path(path):
        yield
        return
    with _network_throttle(settings):
        _bump_network_open_stats(
            stats,
            base_key=NETWORK_PREVIEW_KEY,
            path=path,
            reason=reason,
            kind="preview",
            deep_from_cache=deep_from_cache,
        )
        yield


@contextmanager
def network_read_slot(
    settings,
    path: str,
    stats: dict | None = None,
    *,
    reason: str = "",
    deep_from_cache: bool | None = None,
) -> Iterator[None]:
    """Geriye uyumluluk — orijinal açma sayacına yönlendirir."""
    with network_original_open_slot(
        settings,
        path,
        stats,
        reason=reason,
        deep_from_cache=deep_from_cache,
    ):
        yield


def network_metadata_check(
    settings,
    path: str,
    stats: dict | None,
    fn: Callable[[], T],
    *,
    throttle: bool = True,
    reason: str = "",
) -> T:
    """exists/stat/getmtime gibi metadata kontrolleri — ağ yollarında sayılır."""
    if not is_network_path(path):
        return fn()
    bump_io_stat(stats, NETWORK_METADATA_KEY)
    bump_io_stat(stats, _pass_counter_key(stats, NETWORK_METADATA_KEY))
    _record_network_open(
        stats,
        path=path,
        reason=reason or "metadata_check",
        kind="metadata",
        deep_from_cache=resolve_deep_from_cache(stats),
    )
    if throttle:
        with _network_throttle(settings):
            return fn()
    return fn()


def tracked_isfile(
    settings,
    path: str,
    stats: dict | None = None,
    *,
    allow_network: bool = True,
    reason: str = "isfile",
) -> bool:
    if not path:
        return False
    if is_network_path(path):
        if not allow_network:
            return False
        return bool(
            network_metadata_check(
                settings, path, stats, lambda: os.path.isfile(path), reason=reason
            )
        )
    exists = os.path.isfile(path)
    if exists:
        track_local_cache_read(stats)
    return exists


def tracked_getmtime(
    settings,
    path: str,
    stats: dict | None = None,
    *,
    allow_network: bool = True,
    default: float = 0.0,
    reason: str = "getmtime",
) -> float:
    if not path:
        return default
    if is_network_path(path):
        if not allow_network:
            return default

        def _read() -> float:
            try:
                return float(os.path.getmtime(path))
            except OSError:
                return default

        return float(
            network_metadata_check(settings, path, stats, _read, reason=reason)
        )
    try:
        mtime = float(os.path.getmtime(path))
        track_local_cache_read(stats)
        return mtime
    except OSError:
        return default
