"""Çalışma profili — PC'yi kilitlemeyen arka plan motoru (P1)."""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from typing import Any

from core.logger import setup_logger

logger = setup_logger(__name__)

# Windows process/thread priority classes
_IDLE = 0x00000040
_BELOW_NORMAL = 0x00004000
_NORMAL = 0x00000020
_THREAD_PRIORITY_LOWEST = -2
_THREAD_PRIORITY_BELOW_NORMAL = -1
_THREAD_PRIORITY_IDLE = -15

PROFILE_BACKGROUND = "background"
PROFILE_BALANCED = "balanced"
PROFILE_MAX_SPEED = "max_speed"

CREATIVE_PROCESS_NAMES = frozenset(
    {
        "photoshop.exe",
        "illustrator.exe",
        "coreldr.exe",
        "coreldraw.exe",
        "indesign.exe",
    }
)


@dataclass(frozen=True)
class WorkProfileLimits:
    key: str
    label: str
    index_workers: int
    network_workers: int
    preview_workers: int
    ai_workers: int
    process_priority: str  # below_normal | idle | normal
    ocr_thread_priority: str  # idle | lowest | below_normal
    db_batch_size: int
    chunk_size: int


PROFILES: dict[str, WorkProfileLimits] = {
    PROFILE_BACKGROUND: WorkProfileLimits(
        key=PROFILE_BACKGROUND,
        label="Arka Plan (önerilen)",
        index_workers=2,
        network_workers=2,
        preview_workers=2,
        ai_workers=2,
        process_priority="below_normal",
        ocr_thread_priority="idle",
        db_batch_size=200,
        chunk_size=24,
    ),
    PROFILE_BALANCED: WorkProfileLimits(
        key=PROFILE_BALANCED,
        label="Dengeli",
        index_workers=4,
        network_workers=2,
        preview_workers=4,
        ai_workers=2,
        process_priority="below_normal",
        ocr_thread_priority="lowest",
        db_batch_size=300,
        chunk_size=40,
    ),
    PROFILE_MAX_SPEED: WorkProfileLimits(
        key=PROFILE_MAX_SPEED,
        label="Maksimum Hız",
        index_workers=8,
        network_workers=4,
        preview_workers=6,
        ai_workers=4,
        process_priority="normal",
        ocr_thread_priority="below_normal",
        db_batch_size=500,
        chunk_size=64,
    ),
}

_lock = threading.Lock()
_adaptive_cap: int | None = None
_last_creative: tuple[str, ...] = ()


def normalize_profile(key: str | None) -> str:
    raw = (key or PROFILE_BACKGROUND).strip().lower()
    if raw in ("background", "arka_plan", "arka plan", "low"):
        return PROFILE_BACKGROUND
    if raw in ("balanced", "dengeli", "medium"):
        return PROFILE_BALANCED
    if raw in ("max_speed", "maximum", "max", "hiz", "hız"):
        return PROFILE_MAX_SPEED
    return PROFILE_BACKGROUND if raw not in PROFILES else raw


def get_profile(settings: Any = None) -> WorkProfileLimits:
    key = PROFILE_BACKGROUND
    if settings is not None:
        key = normalize_profile(getattr(settings, "work_profile", None))
    return PROFILES[key]


def detect_creative_apps() -> list[str]:
    found: list[str] = []
    try:
        import psutil

        for p in psutil.process_iter(["name"]):
            name = (p.info.get("name") or "").lower()
            if name in CREATIVE_PROCESS_NAMES:
                found.append(name)
    except Exception:
        pass
    return sorted(set(found))


def system_pressure() -> dict[str, float]:
    out = {"cpu": 0.0, "ram": 0.0, "handles": 0.0}
    try:
        import psutil

        out["cpu"] = float(psutil.cpu_percent(interval=None))
        out["ram"] = float(psutil.virtual_memory().percent)
        try:
            out["handles"] = float(psutil.Process(os.getpid()).num_handles())
        except (AttributeError, psutil.Error):
            pass
    except Exception:
        pass
    return out


def refresh_adaptive_cap(settings: Any = None) -> int:
    """
    Photoshop vb. açıkken veya sistem baskısında worker tavanını düşür.
    Örn: 24 → 8 → 4 → 2
    """
    global _adaptive_cap, _last_creative
    profile = get_profile(settings)
    base = max(1, int(profile.index_workers))
    creative = detect_creative_apps()
    pressure = system_pressure()
    cap = base

    if creative:
        # Creative app → agresif düşüş
        if base >= 8:
            cap = 4
        elif base >= 4:
            cap = 2
        else:
            cap = 1
    if pressure["cpu"] >= 85 or pressure["ram"] >= 90:
        cap = min(cap, 2)
    if pressure["handles"] >= 8000:
        cap = min(cap, 2)
    if pressure["handles"] >= 12000:
        cap = 1

    with _lock:
        prev = _adaptive_cap
        _adaptive_cap = cap
        if creative != list(_last_creative) or prev != cap:
            _last_creative = tuple(creative)
            logger.warning(
                "[WORK PROFILE] profile=%s adaptive_cap=%s creative=%s "
                "cpu=%.0f ram=%.0f handles=%.0f",
                profile.key,
                cap,
                list(creative),
                pressure["cpu"],
                pressure["ram"],
                pressure["handles"],
            )
    return cap


def get_adaptive_cap(settings: Any = None) -> int:
    with _lock:
        if _adaptive_cap is None:
            pass
        else:
            return max(1, int(_adaptive_cap))
    return refresh_adaptive_cap(settings)


def effective_network_workers(settings: Any) -> int:
    profile = get_profile(settings)
    user = max(1, int(getattr(settings, "max_network_workers", 2) or 2))
    cap = get_adaptive_cap(settings)
    return max(1, min(user, profile.network_workers, cap))


def apply_process_priority(settings: Any = None) -> None:
    """Windows process önceliğini profil göre ayarla."""
    profile = get_profile(settings)
    if os.name != "nt":
        try:
            os.nice(10 if profile.process_priority != "normal" else 0)
        except Exception:
            pass
        return
    mapping = {
        "idle": _IDLE,
        "below_normal": _BELOW_NORMAL,
        "normal": _NORMAL,
    }
    cls = mapping.get(profile.process_priority, _BELOW_NORMAL)
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.kernel32.SetPriorityClass(handle, cls)
        if ok:
            logger.info(
                "Process priority set to %s (%s)",
                profile.process_priority,
                profile.key,
            )
        else:
            logger.debug("SetPriorityClass failed err=%s", ctypes.GetLastError())
    except Exception as exc:
        logger.debug("apply_process_priority: %s", exc)


def apply_current_thread_priority(kind: str = "worker") -> None:
    """OCR/embedding/thumbnail thread'lerini LOW'a çek."""
    if os.name != "nt":
        return
    level = {
        "ocr": _THREAD_PRIORITY_IDLE,
        "embedding": _THREAD_PRIORITY_LOWEST,
        "thumbnail": _THREAD_PRIORITY_LOWEST,
        "worker": _THREAD_PRIORITY_BELOW_NORMAL,
    }.get(kind, _THREAD_PRIORITY_BELOW_NORMAL)
    try:
        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, level)
    except Exception:
        pass


def release_batch_memory() -> None:
    """Batch sonunda thumbnail/OCR/embedding tamponlarını serbest bırak."""
    import gc

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
