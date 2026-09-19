"""CPU dostu arka plan index politikası + çalışma profili."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from core.index_modes import IndexMode
from core.network_index_throttle import is_network_path
from core.settings import AppSettings
from core.work_profile import (
    apply_process_priority,
    effective_network_workers,
    get_profile,
    refresh_adaptive_cap,
)


@dataclass
class IndexThrottleState:
    search_active: bool = False
    background_low_priority: bool = True


_state = IndexThrottleState()
_lock = threading.Lock()


def set_search_active(active: bool) -> None:
    with _lock:
        _state.search_active = active


def is_search_active() -> bool:
    with _lock:
        return bool(_state.search_active)


def set_background_low_priority(enabled: bool) -> None:
    with _lock:
        _state.background_low_priority = enabled


def effective_worker_count(
    settings: AppSettings,
    *,
    source_type: str = "local_pc",
    ai_enabled: bool = False,
    light_pass: bool = True,
    source_path: str = "",
) -> int:
    """Profil + arama önceliği + adaptif tavan ile worker sayısı."""
    apply_process_priority(settings)
    profile = get_profile(settings)
    mode = getattr(settings, "index_mode", "standard") or "standard"

    preview_workers = max(
        1,
        min(
            int(getattr(settings, "preview_worker_count", None) or settings.index_worker_count or 4),
            profile.preview_workers,
        ),
    )
    ai_workers = max(
        1,
        min(
            int(
                getattr(settings, "ai_worker_count", None)
                or getattr(settings, "heavy_index_worker_count", 1)
                or 1
            ),
            profile.ai_workers,
        ),
    )
    network_cap = effective_network_workers(settings)
    base = max(
        1,
        min(
            int(getattr(settings, "index_worker_count", None) or settings.worker_count or 1),
            profile.index_workers,
        ),
    )

    with _lock:
        search_on = _state.search_active
        low_pri = _state.background_low_priority

    if mode == IndexMode.NIGHT_COMPLETE.value or (ai_enabled and not light_pass):
        workers = ai_workers
    elif mode == IndexMode.FAST_ARCHIVE.value or light_pass:
        workers = preview_workers
    else:
        workers = base

    is_network = source_type in ("server_share", "nas_backup") or is_network_path(source_path)
    if is_network:
        workers = min(workers, network_cap)

    adaptive = refresh_adaptive_cap(settings)
    workers = min(workers, adaptive, profile.index_workers)

    if search_on and low_pri:
        return 1
    return max(1, workers)


def status_label(settings: AppSettings) -> str:
    with _lock:
        parts = []
        profile = get_profile(settings)
        parts.append(f"Mod: {profile.label}")
        if _state.background_low_priority:
            parts.append("düşük öncelik")
        if _state.search_active:
            parts.append("Arama öncelikli")
        if settings.ai_embedding_enabled:
            parts.append(
                "AI index: kapalı (hızlı)"
                if settings.index_skip_ai
                else "AI index: açık (yavaş)"
            )
        return " | ".join(parts) if parts else "CPU dostu mod"
