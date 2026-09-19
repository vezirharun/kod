"""Kalıcı index oturumu — kapanış sonrası doğru modla devam."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.index_modes import IndexMode

META_KEY = "index_session_v1"

MODE_LABELS: dict[str, str] = {
    IndexMode.FAST_ARCHIVE.value: "Hızlı Index",
    IndexMode.NIGHT_COMPLETE.value: "Genel AI",
    IndexMode.BACKFILL.value: "Eksikleri Tamamla",
    IndexMode.COMPLETE.value: "Hızlı + Genel AI",
    IndexMode.STANDARD.value: "Standart",
    "patch": "PATCH",
    "ocr": "OCR",
    "post_ga": "PATCH + OCR",
    "general_ai": "Genel AI",
}

ACTIVE_STATUSES = frozenset({"running", "paused", "interrupted"})


@dataclass
class IndexSessionState:
    index_mode: str = IndexMode.STANDARD.value
    scan_mode: str = "quick"
    status: str = "idle"  # idle | running | paused | interrupted | completed
    started_at: str = ""
    updated_at: str = ""
    light_done: int = 0
    light_total: int = 0
    heavy_done: int = 0
    heavy_total: int = 0
    jobs_done: int = 0
    jobs_total: int = 0
    search_ready: int = 0
    total_files: int = 0
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def display_mode(self) -> str:
        return MODE_LABELS.get(self.index_mode, self.index_mode)

    def is_resumable(self) -> bool:
        return self.status in ACTIVE_STATUSES and bool(self.index_mode)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mode_display_label(index_mode: str) -> str:
    return MODE_LABELS.get(str(index_mode or ""), str(index_mode or "—"))


def save_session(db: Any, state: IndexSessionState) -> None:
    state.updated_at = _now()
    if not state.started_at:
        state.started_at = state.updated_at
    db.set_meta(META_KEY, json.dumps(asdict(state), ensure_ascii=False))


def load_session(db: Any) -> IndexSessionState | None:
    raw = db.get_meta(META_KEY, "")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    known = {f.name for f in IndexSessionState.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known}
    try:
        return IndexSessionState(**filtered)
    except TypeError:
        return None


def clear_session(db: Any) -> None:
    db.set_meta(META_KEY, "")


def begin_session(
    db: Any,
    *,
    index_mode: str,
    scan_mode: str = "quick",
    totals: dict[str, int] | None = None,
) -> IndexSessionState:
    """Resume metadata only — light_done/heavy_done cache UI SSOT değildir."""
    totals = totals or {}
    state = IndexSessionState(
        index_mode=str(index_mode or IndexMode.STANDARD.value),
        scan_mode=str(scan_mode or "quick"),
        status="running",
        started_at=_now(),
        light_done=int(totals.get("light_done", 0) or 0),
        light_total=int(totals.get("light_total", totals.get("total_files", 0)) or 0),
        heavy_done=int(totals.get("heavy_done", 0) or 0),
        heavy_total=int(totals.get("heavy_total", totals.get("total_files", 0)) or 0),
        search_ready=int(totals.get("search_ready", 0) or 0),
        total_files=int(totals.get("total_files", 0) or 0),
        jobs_done=0,
        jobs_total=0,
    )
    save_session(db, state)
    return state


def touch_session(
    db: Any,
    *,
    status: str | None = None,
    jobs_done: int | None = None,
    jobs_total: int | None = None,
    light_done: int | None = None,
    heavy_done: int | None = None,
    search_ready: int | None = None,
    total_files: int | None = None,
    note: str | None = None,
) -> IndexSessionState | None:
    state = load_session(db) or IndexSessionState()
    if status is not None:
        state.status = status
    if jobs_done is not None:
        state.jobs_done = int(jobs_done)
    if jobs_total is not None:
        state.jobs_total = int(jobs_total)
    if light_done is not None:
        state.light_done = int(light_done)
    if heavy_done is not None:
        state.heavy_done = int(heavy_done)
    if search_ready is not None:
        state.search_ready = int(search_ready)
    if total_files is not None:
        state.total_files = int(total_files)
        if state.light_total <= 0:
            state.light_total = state.total_files
        if state.heavy_total <= 0:
            state.heavy_total = state.total_files
    if note is not None:
        state.note = note
    save_session(db, state)
    return state


def mark_interrupted(db: Any) -> IndexSessionState | None:
    state = load_session(db)
    if not state or state.status in ("idle", "completed"):
        return state
    state.status = "interrupted"
    save_session(db, state)
    return state


def mark_completed(db: Any) -> None:
    state = load_session(db)
    if not state:
        clear_session(db)
        return
    state.status = "completed"
    save_session(db, state)


def snapshot_progress_totals(db: Any) -> dict[str, int]:
    """Resume dialog — SSOT count_lanes (UI ana sayaçları bunu kullanmaz)."""
    from core.index_ssot import count_lanes

    lanes = count_lanes(db)
    total = int(lanes.get("total") or 0)
    with db.connect() as conn:
        emb = conn.execute(
            """
            SELECT COUNT(*) FROM features fe
            JOIN files f ON f.id = fe.file_id
            WHERE fe.dino_embedding IS NOT NULL AND length(fe.dino_embedding) > 32
              AND fe.clip_embedding IS NOT NULL AND length(fe.clip_embedding) > 32
              AND COALESCE(f.thumbnail_path,'') != ''
              AND COALESCE(f.feature_preview_path,'') != ''
            """
        ).fetchone()[0]
    return {
        "total_files": total,
        "light_done": int(lanes.get("light_done") or 0),
        "light_total": total,
        "heavy_done": int(lanes.get("heavy_done") or 0),
        "heavy_total": total,
        "search_ready": int(emb or 0),
    }
