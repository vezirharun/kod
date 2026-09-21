"""Arka plan iş parçacıkları — UI donmasın."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from core.capability_check import AI_FALLBACK_MSG
from core.indexer import Indexer
from core.logger import setup_logger
from core.scan_scheduler import ScanScheduler
from core.search_engine import SearchEngine
from core.search_models import SearchQuery, SearchResponse, SearchStats
from core.settings import AppSettings
from core.sources import ScanMode, SourceManager
from ui.designer_labels import stage_label

logger = setup_logger(__name__)

_THREAD_WAIT_MS = 3000

PROGRESSIVE_STAGES: tuple[tuple[str, float], ...] = (
    ("Exact", 0.2),
    ("Pattern Family", 0.5),
    ("Semantic", 1.0),
    ("Texture", 2.0),
    ("Deep Search", 4.0),
)


def _is_progressive_exact(result) -> bool:
    dbg = getattr(result, "debug", None) or {}
    if getattr(result, "is_self_match", False) or dbg.get("protected_exact"):
        return True
    if str(dbg.get("result_layer") or "") == "same_files":
        return True
    try:
        if float(dbg.get("exact_score", dbg.get("exact_search_score", 0)) or 0) >= 0.70:
            return True
        if float((getattr(result, "breakdown", None) or {}).get("phash", 0) or 0) >= 0.82:
            return True
    except (TypeError, ValueError):
        return False
    return False


def _progressive_stage_for_elapsed(elapsed: float) -> str:
    stage = PROGRESSIVE_STAGES[0][0]
    for name, min_t in PROGRESSIVE_STAGES:
        if elapsed >= min_t:
            stage = name
    return stage


def _filter_progressive_results(results: list, stage: str) -> list:
    pool = list(results or [])
    exact = [r for r in pool if _is_progressive_exact(r)]
    if stage == "Exact":
        return exact[:80]
    if stage == "Pattern Family":
        rest = [
            r
            for r in pool
            if r not in exact
            and (
                getattr(r, "same_pattern_family", False)
                or getattr(r, "same_animal_family", False)
                or float((getattr(r, "debug", None) or {}).get("pattern_family_score", 0) or 0)
                >= 0.42
            )
        ][:120]
        return (exact + rest)[:120]
    if stage == "Semantic":
        rest = [
            r
            for r in pool
            if r not in exact
            and (
                float((getattr(r, "debug", None) or {}).get("semantic_score", 0) or 0) >= 0.28
                or float((getattr(r, "debug", None) or {}).get("dna_score", 0) or 0) >= 0.28
            )
        ][:160]
        return (exact + rest)[:160]
    if stage == "Texture":
        rest = [
            r
            for r in pool
            if r not in exact
            and (
                float((getattr(r, "debug", None) or {}).get("texture_score", 0) or 0) >= 0.22
                or float((getattr(r, "breakdown", None) or {}).get("patch", 0) or 0) >= 0.28
            )
        ][:200]
        return (exact + rest)[:200]
    rest = [r for r in pool if r not in exact]
    return (exact + rest)[:240]


def _wait_progressive(t_start: float, target: float, stop_check) -> bool:
    while time.perf_counter() - t_start < target:
        if stop_check():
            return False
        time.sleep(0.02)
    return True


class _WorkerBase(QThread):
    """Ortak durdurma ve güvenli bekleme."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def is_stop_requested(self) -> bool:
        return self._stop_requested

    def wait_until_finished(self, timeout_ms: int = _THREAD_WAIT_MS) -> bool:
        if not self.isRunning():
            return True
        if not self.wait(timeout_ms):
            logger.warning(
                "%s kapanmadı (timeout %d ms)", self.__class__.__name__, timeout_ms
            )
            return False
        return True

    def arm_delete_later_on_finished(self) -> None:
        """Connect QThread.finished → deleteLater once (idempotent)."""
        if getattr(self, "_delete_later_armed", False):
            return
        self._delete_later_armed = True
        self.finished.connect(self.deleteLater)


class IndexWorker(_WorkerBase):
    """Phase 3: Index Engine V3 — legacy Indexer/ScanScheduler KULLANMAZ."""

    progress = Signal(dict)
    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        customer_filter: str = "",
        scan_mode: str = ScanMode.QUICK.value,
        source_id: int = 0,
        run_due_only: bool = False,
        index_mode: str = "standard",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("IndexWorker")
        self.settings = settings
        self.customer_filter = customer_filter
        self.scan_mode = scan_mode
        self.source_id = source_id
        self.run_due_only = run_due_only
        self.index_mode = index_mode
        self._engine = None
        self._live_scope_ids: list[int] | None = None

    def include_source_id(self, source_id: int) -> None:
        sid = int(source_id or 0)
        if sid <= 0:
            return
        ids = self._live_scope_ids
        if ids is not None and sid not in ids:
            ids.append(sid)
        eng = self._engine
        if eng is not None:
            try:
                eng.include_source_id(sid)
            except Exception:
                pass

    def drop_source_id(self, source_id: int) -> None:
        sid = int(source_id or 0)
        if sid <= 0:
            return
        ids = self._live_scope_ids
        if ids is not None:
            try:
                ids.remove(sid)
            except ValueError:
                pass
        eng = self._engine
        if eng is not None:
            try:
                eng.drop_source_id(sid)
            except Exception:
                pass

    def run(self) -> None:
        try:
            self._stop_requested = False
            try:
                from core.work_profile import apply_process_priority

                apply_process_priority(self.settings)
            except Exception:
                pass

            from core.db import Database
            from core.index_v3 import IndexEngineV3, Mode
            from core.index_v3.scope import orphan_source_report, resolve_index_scope
            from core.index_v3.ui_bridge import map_ui_mode_to_v3, v3_status_dict
            from core.sources import SourceManager

            v3_mode = Mode(map_ui_mode_to_v3(self.index_mode))
            self.progress.emit(
                {
                    "stage": "index_starting",
                    "active_phase": "V3",
                    "current_file": "Index Engine V3 hazırlanıyor…",
                    "index_mode": self.index_mode,
                    "pipeline_task": "V3",
                    "engine": "index_v3",
                }
            )

            db = Database(self.settings.db_path)
            sm = SourceManager(self.settings)
            archive_selected = [
                int(x)
                for x in (self.settings.selected_source_ids or [])
                if int(x) > 0
            ]
            walk_selected = (
                [int(self.source_id)] if self.source_id else list(archive_selected)
            )
            resolved = resolve_index_scope(db, walk_selected)
            ssot_resolved = resolve_index_scope(
                db, archive_selected or walk_selected
            )
            scope_ids = list(ssot_resolved.source_ids)
            self._live_scope_ids = scope_ids
            try:
                from core.index_v3.ui_bridge import count_v3_ssot

                scoped_total = int(count_v3_ssot(db, scope_ids).get("total") or 0)
            except Exception:
                scoped_total = -1
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "INDEX_START: selected_source_ids=%s resolved_source_ids=%s "
                "scoped_total=%s mode=%s index_mode=%s",
                walk_selected,
                scope_ids,
                scoped_total,
                resolved.mode,
                self.index_mode,
            )
            print(
                f"INDEX_START: selected_source_ids={walk_selected} "
                f"resolved_source_ids={scope_ids} scoped_total={scoped_total}",
                flush=True,
            )

            # Registry roots + orphan stubs (root_path boş → walk yok, DB enqueue)
            by_id = {
                int(s["id"]): s
                for s in sm.list_sources(active_only=False)
                if int(s.get("id") or 0) > 0
            }
            sources: list[dict] = []
            for sid in scope_ids:
                row = by_id.get(sid)
                if not row:
                    # Orphan source_id arşiv/kuyruk kapsamına girmez
                    continue
                sources.append(
                    {
                        "id": sid,
                        "root_path": str(row.get("root_path") or ""),
                        "orphan": False,
                    }
                )

            self.progress.emit(
                {
                    "stage": "index_ready",
                    "active_phase": v3_mode.value,
                    "current_file": (
                        f"Kapsam {len(scope_ids)} source_id · "
                        f"{resolved.source_count_display} kayıtlı"
                        + (
                            f" · {resolved.orphan_file_total:,} orphan dosya"
                            if resolved.orphan_file_total
                            else ""
                        )
                    ),
                    "index_mode": self.index_mode,
                    "engine": "index_v3",
                    "orphan_sources_report": orphan_source_report(db)
                    if resolved.orphan_source_ids
                    else [],
                }
            )

            self._engine = IndexEngineV3(
                db,
                settings=self.settings,
                use_real_extractors=True,
            )

            last_emit = [0.0]
            last_disc_ssot = [0.0]
            current_info: dict = {"phase": "starting"}
            session_pool_start: dict = {}
            try:
                from core.index_v3.ui_bridge import count_v3_ssot as _count_ssot

                session_pool_start = dict(_count_ssot(db, scope_ids))
            except Exception:
                session_pool_start = {}

            def _emit_live_claim(info: dict) -> None:
                """Dosya/aşama — count_v3_ssot yok. Sayaç UI 1 Hz StatusWorker."""
                import time as _time
                from pathlib import Path as _P

                nonlocal current_info
                incoming = dict(info)
                old_started = current_info.get("claim_started")
                same_claim = (
                    current_info.get("phase") == "claimed"
                    and incoming.get("phase") == "claimed"
                    and int(current_info.get("file_id") or 0)
                    == int(incoming.get("file_id") or 0)
                    and str(current_info.get("artifact") or "")
                    == str(incoming.get("artifact") or "")
                )
                current_info = incoming
                if incoming.get("phase") == "claimed":
                    current_info["claim_started"] = (
                        float(old_started)
                        if same_claim and old_started is not None
                        else _time.monotonic()
                    )
                fname = str(
                    current_info.get("filename")
                    or _P(str(current_info.get("path") or "")).name
                    or ""
                )
                art = str(
                    current_info.get("stage")
                    or current_info.get("artifact")
                    or ""
                )
                processing = (
                    1 if current_info.get("phase") == "claimed" else 0
                )
                elapsed = 0.0
                started = current_info.get("claim_started")
                if started is not None and processing:
                    elapsed = max(0.0, _time.monotonic() - float(started))
                wlabel = str(current_info.get("worker") or "")
                if fname and processing:
                    msg = (
                        f"{fname} · {art or '…'}"
                        + (f" · {wlabel}" if wlabel else "")
                        + f" · {elapsed:.1f}s"
                    )
                else:
                    msg = f"V3 {v3_mode.value}"
                src_name = ""
                try:
                    sid = int(current_info.get("source_id") or 0)
                    if sid and sid in by_id:
                        src_name = str(by_id[sid].get("name") or "")
                except Exception:
                    src_name = ""
                q = str(current_info.get("queue") or "").lower()
                is_light = q in ("light", "preview") or art in (
                    "thumbnail",
                    "preview",
                )
                self.progress.emit(
                    {
                        "stage": "v3_progress",
                        "engine": "index_v3",
                        "index_mode": self.index_mode,
                        "ssot_refresh": False,
                        "active_phase": v3_mode.value,
                        "pipeline_task": f"V3 {art}" if art else f"V3 {v3_mode.value}",
                        "current_file": msg,
                        "current_filename": fname,
                        "current_source": src_name,
                        "current_path": str(current_info.get("path") or ""),
                        "current_stage": art,
                        "current_worker": wlabel or "v3-heavy",
                        "current_elapsed_sec": elapsed,
                        "current_file_info": dict(current_info),
                        "processing": processing,
                        "light_processing": processing if is_light else 0,
                        "heavy_processing": 0 if is_light else processing,
                    }
                )

            def _emit_ssot(
                stage: str = "v3_progress",
                note: str = "",
                *,
                file_info: dict | None = None,
                force: bool = False,
            ) -> None:
                if self._stop_requested:
                    return
                from core.index_v3.live_contract import (
                    build_live_progress,
                    worker_progress_should_compute_ssot,
                )

                # Drain/aşama callback SSOT çekmez (force=True dahil).
                if not worker_progress_should_compute_ssot(stage):
                    return
                import time as _time
                from pathlib import Path as _P

                nonlocal current_info
                if file_info is not None:
                    incoming = dict(file_info)
                    old_started = current_info.get("claim_started")
                    same_claim = (
                        current_info.get("phase") == "claimed"
                        and incoming.get("phase") == "claimed"
                        and int(current_info.get("file_id") or 0)
                        == int(incoming.get("file_id") or 0)
                        and str(current_info.get("artifact") or "")
                        == str(incoming.get("artifact") or "")
                    )
                    current_info = incoming
                    if incoming.get("phase") == "claimed":
                        current_info["claim_started"] = (
                            float(old_started)
                            if same_claim and old_started is not None
                            else _time.monotonic()
                        )
                now = _time.monotonic()
                if (
                    stage == "v3_progress"
                    and not force
                    and (now - last_emit[0]) < 1.0
                ):
                    return
                last_emit[0] = now
                try:
                    sel = [
                        int(x)
                        for x in (self.settings.selected_source_ids or [])
                        if int(x) > 0
                    ]
                    live = list(resolve_index_scope(db, sel).source_ids)
                    for sid in live:
                        if sid not in scope_ids:
                            scope_ids.append(sid)
                        eng = self._engine
                        if eng is not None and not int(self.source_id or 0):
                            eng.include_source_id(sid)
                except Exception:
                    pass
                claimed_n = (
                    int(self._engine.store.count_claimed() or 0)
                    if self._engine
                    else 0
                )
                claimed_fresh_n = (
                    int(
                        self._engine.store.count_claimed_fresh(
                            source_ids=scope_ids
                        )
                        or 0
                    )
                    if self._engine
                    else 0
                )
                pending_n = (
                    int(
                        self._engine.store.count_pending(source_ids=scope_ids)
                        or 0
                    )
                    if self._engine
                    else 0
                )
                pending_light = 0
                pending_heavy = 0
                pending_heavy_files = 0
                retry_light = 0
                retry_heavy = 0
                failed_permanent = 0
                failed_permanent_files = 0
                retry_file_names: list[str] = []
                failed_file_names: list[str] = []
                if self._engine:
                    from core.index_v3.types import QueueKind as _QK

                    pending_light = int(
                        self._engine.store.count_pending(
                            _QK.LIGHT, source_ids=scope_ids
                        )
                        or 0
                    ) + int(
                        self._engine.store.count_pending(
                            _QK.PREVIEW, source_ids=scope_ids
                        )
                        or 0
                    )
                    pending_heavy = int(
                        self._engine.store.count_pending(
                            _QK.HEAVY, source_ids=scope_ids
                        )
                        or 0
                    ) + int(
                        self._engine.store.count_pending(
                            _QK.REPAIR, source_ids=scope_ids
                        )
                        or 0
                    )
                    pending_heavy_files = int(
                        self._engine.store.count_pending_files(
                            queues=(_QK.HEAVY, _QK.REPAIR),
                            source_ids=scope_ids,
                        )
                        or 0
                    )
                    retry_light = int(
                        self._engine.store.count_pending_retry(
                            _QK.LIGHT, source_ids=scope_ids
                        )
                        or 0
                    ) + int(
                        self._engine.store.count_pending_retry(
                            _QK.PREVIEW, source_ids=scope_ids
                        )
                        or 0
                    )
                    retry_heavy = int(
                        self._engine.store.count_pending_retry(
                            _QK.HEAVY, source_ids=scope_ids
                        )
                        or 0
                    ) + int(
                        self._engine.store.count_pending_retry(
                            _QK.REPAIR, source_ids=scope_ids
                        )
                        or 0
                    )
                    failed_permanent = int(
                        self._engine.store.count_failed_permanent(
                            source_ids=scope_ids
                        )
                        or 0
                    )
                    failed_permanent_files = int(
                        self._engine.store.count_failed_permanent_files(
                            source_ids=scope_ids
                        )
                        or 0
                    )
                    from pathlib import Path as _FilePath
                    retry_file_names = [
                        (_FilePath(str(x.get("path") or "")).name or str(x.get("file_id") or ""))
                        for x in self._engine.store.list_pending_retry_files(
                            source_ids=scope_ids, limit=25
                        )
                    ]
                    failed_file_names = [
                        (_FilePath(str(x.get("path") or "")).name or str(x.get("file_id") or ""))
                        for x in self._engine.store.list_failed_permanent_files(
                            source_ids=scope_ids, limit=25
                        )
                    ]
                # İşleniyor = aktif claim (phase) VEYA JobStore state=claimed.
                # claimed_fresh gecikmesinde çalışan iş 0 görünmesin.
                from core.index_v3.types import QueueKind as _QK2

                claimed_light_n = (
                    int(
                        self._engine.store.count_claimed(
                            source_ids=scope_ids,
                            queues=(_QK2.LIGHT, _QK2.PREVIEW),
                        )
                        or 0
                    )
                    if self._engine
                    else 0
                )
                claimed_heavy_n = (
                    int(
                        self._engine.store.count_claimed(
                            source_ids=scope_ids,
                            queues=(_QK2.HEAVY, _QK2.REPAIR),
                        )
                        or 0
                    )
                    if self._engine
                    else 0
                )
                phase_claimed = current_info.get("phase") == "claimed"
                art_live = str(
                    current_info.get("artifact")
                    or current_info.get("stage")
                    or ""
                ).lower()
                q_live = str(current_info.get("queue") or "").lower()
                live_is_light = q_live in ("light", "preview") or art_live in (
                    "thumbnail",
                    "preview",
                )
                light_proc_n = claimed_light_n
                heavy_proc_n = claimed_heavy_n
                if phase_claimed:
                    if live_is_light:
                        light_proc_n = max(light_proc_n, 1)
                    else:
                        heavy_proc_n = max(heavy_proc_n, 1)
                processing_n = max(light_proc_n, heavy_proc_n)
                jobs_light_1m = (
                    int(
                        self._engine.store.count_completed_recent(
                            60.0,
                            source_ids=scope_ids,
                            queues=(_QK2.LIGHT, _QK2.PREVIEW),
                        )
                        or 0
                    )
                    if self._engine
                    else 0
                )
                jobs_heavy_1m = (
                    int(
                        self._engine.store.count_completed_recent(
                            60.0,
                            source_ids=scope_ids,
                            queues=(_QK2.HEAVY, _QK2.REPAIR),
                        )
                        or 0
                    )
                    if self._engine
                    else 0
                )
                st = v3_status_dict(
                    db,
                    scope_ids,
                    processing=processing_n,
                    claimed=processing_n,
                    light_processing=light_proc_n,
                    heavy_processing=heavy_proc_n,
                    pending_jobs=pending_n,
                    failed_permanent_files=failed_permanent_files,
                    scope=resolved,
                    current_file=current_info,
                )
                st["pending_light_jobs"] = pending_light
                st["pending_heavy_jobs"] = pending_heavy
                st["pending_heavy_files"] = pending_heavy_files
                st["claimed_jobs"] = claimed_n
                st["claimed_fresh_jobs"] = claimed_fresh_n
                st["claimed_light_jobs"] = claimed_light_n
                st["claimed_heavy_jobs"] = claimed_heavy_n
                st["retry_light_jobs"] = retry_light
                st["retry_heavy_jobs"] = retry_heavy
                st["jobs_completed_light_1m"] = jobs_light_1m
                st["jobs_completed_heavy_1m"] = jobs_heavy_1m
                st["light_job_speed_pm"] = float(jobs_light_1m)
                st["heavy_job_speed_pm"] = float(jobs_heavy_1m)
                # UI'de hata sayısı job değil farklı dosya sayısıdır.
                st["failed_permanent_jobs"] = failed_permanent_files
                st["failed_permanent_files"] = failed_permanent_files
                st["retry_file_names"] = retry_file_names
                st["failed_file_names"] = failed_file_names
                live = build_live_progress(
                    st,
                    session_start=session_pool_start,
                    claim=current_info,
                    now=now,
                )
                preview = int(live["fast"]["completed"])
                total = int(live["total"])
                ai = int(live["general_ai"]["completed"])
                if v3_mode.value == "fast":
                    done, goal = preview, total
                    remaining = int(live["fast"].get("file_queue", live["fast"]["remaining"]))
                    pending = int(live["fast"].get("job_queue", live["fast"]["executable"]))
                elif v3_mode.value == "general_ai":
                    done, goal = ai, total
                    remaining = int(
                        live["general_ai"].get("file_queue", live["general_ai"]["remaining"])
                    )
                    pending = int(
                        live["general_ai"].get("job_queue", live["general_ai"]["executable"])
                    )
                else:
                    done, goal = ai, total
                    remaining = int(live["fast"].get("file_queue", live["fast"]["remaining"]))
                    pending = pending_n
                fname = str(live.get("current_filename") or "").strip()
                art = str(live.get("current_stage") or "")
                elapsed = float(live.get("elapsed_sec") or 0)
                stall = bool(live.get("stall"))
                if fname and live.get("processing"):
                    wlabel = str(live.get("current_worker") or "")
                    msg = (
                        f"{fname} · {art or '…'}"
                        + (f" · {wlabel}" if wlabel else "")
                        + f" · {elapsed:.1f}s"
                        + (" · STALL" if stall else "")
                    )
                else:
                    msg = note or (
                        f"V3 {v3_mode.value}: {done:,}/{goal:,} · "
                        f"kuyruk {remaining:,} · job {pending:,}"
                    )
                src_name = ""
                try:
                    sid = int(current_info.get("source_id") or 0)
                    if sid and sid in by_id:
                        src_name = str(by_id[sid].get("name") or "")
                except Exception:
                    src_name = ""
                self.progress.emit(
                    {
                        **st,
                        "stage": stage,
                        "index_mode": self.index_mode,
                        "engine": "index_v3",
                        "processed": done,
                        "active_phase": v3_mode.value,
                        "pipeline_task": (
                            f"V3 {art}" if art else f"V3 {v3_mode.value}"
                        ),
                        "current_file": msg,
                        "current_filename": fname,
                        "current_source": src_name,
                        "current_path": str(live.get("current_path") or ""),
                        "current_stage": art,
                        "current_worker": str(
                            live.get("current_worker")
                            or current_info.get("worker")
                            or "v3-main"
                        ),
                        "current_elapsed_sec": elapsed,
                        "processing": int(live.get("processing") or 0),
                        "claimed_jobs": claimed_n,
                        "claimed_fresh_jobs": claimed_fresh_n,
                        "stall_warning": stall,
                        "remaining_display": remaining,
                        "executable_display": pending,
                        "session_plus": dict(live.get("session_plus") or {}),
                        "live_stages": dict(live.get("stages") or {}),
                        "live_fast": dict(live.get("fast") or {}),
                        "live_general_ai": dict(live.get("general_ai") or {}),
                    }
                )

            def _on_worker_progress(info: dict | None = None) -> None:
                # Engine drain (None): claim ezme, SSOT yok.
                # Aşama: yalnız canlı dosya; count_v3_ssot UI 1 Hz timer'da.
                if info is None:
                    return
                if str(info.get("phase") or "") == "discovery":
                    import time as _time

                    payload = {
                        "stage": "source_discover",
                        "phase": "discovery",
                        "engine": "index_v3",
                        "source_id": int(
                            info.get("source_id") or self.source_id or 0
                        ),
                        "found": int(info.get("found") or 0),
                        "inserted": int(info.get("inserted") or 0),
                        "jobs_enqueued": int(info.get("jobs_enqueued") or 0),
                        "queued": int(info.get("queued") or 0),
                        "indexed": int(info.get("indexed") or 0),
                    }
                    now = _time.monotonic()
                    if last_disc_ssot[0] == 0.0 or (now - last_disc_ssot[0]) >= 0.25:
                        last_disc_ssot[0] = now
                        try:
                            from core.index_v3.ui_bridge import count_v3_ssot as _cssot

                            pools = _cssot(db, list(scope_ids))
                            payload["total"] = int(pools.get("total") or 0)
                            payload["light_done"] = int(
                                pools.get("light_complete") or 0
                            )
                        except Exception:
                            pass
                    self.progress.emit(payload)
                    return
                _emit_live_claim(info)

            _emit_ssot(
                "index_ready",
                "Kapsam hazır — claim bekleniyor…",
                force=True,
            )
            if self._stop_requested:
                self.finished_ok.emit(
                    {"stopped": True, "index_mode": self.index_mode, "engine": "index_v3"}
                )
                return

            mode = Mode.REPAIR if self.run_due_only else v3_mode
            from core.index_v3.types import POST_GA_MODES

            walk_disk = False
            if mode not in POST_GA_MODES:
                from core.index_v3.discovery import should_walk_on_index_start

                for src in sources:
                    if src.get("orphan"):
                        continue
                    sid = int(src["id"])
                    with db.connect() as conn:
                        n_existing = int(
                            conn.execute(
                                """
                                SELECT COUNT(*) FROM files
                                WHERE source_id=? AND status NOT IN ('excluded_internal','missing')
                                """,
                                (sid,),
                            ).fetchone()[0]
                            or 0
                        )
                    if should_walk_on_index_start(
                        explicit_source_id=int(self.source_id or 0),
                        existing_file_count=n_existing,
                        root_path=str(src.get("root_path") or ""),
                        post_ga=False,
                    ):
                        walk_disk = True
                        break

            report = self._engine.run(
                mode=mode,
                sources=sources,
                walk_disk=walk_disk,
                progress_callback=_on_worker_progress,
            )
            current_info = {"phase": "idle"}
            _emit_ssot("index_done", force=True)
            stats = {
                "engine": "index_v3",
                "index_mode": self.index_mode,
                "v3_mode": mode.value,
                "discovery": report.discovery,
                "worker": report.worker,
                "stopped": bool(self._stop_requested),
            }
            worker_errors = list((report.worker or {}).get("worker_errors") or [])
            if worker_errors:
                self.progress.emit(
                    {
                        "stage": "worker_error",
                        "active_phase": v3_mode.value,
                        "engine": "index_v3",
                        "worker_errors": worker_errors,
                        "error": "; ".join(worker_errors),
                        "processing": 0,
                    }
                )
                self.error.emit(stats["worker_errors"][-1])
                return
            self.finished_ok.emit(stats)
        except Exception as exc:
            import logging as _logging

            _logging.getLogger(__name__).exception(
                "IndexWorker failed mode=%s: %s", self.index_mode, exc
            )
            if self._stop_requested:
                self.finished_ok.emit(
                    {
                        "stopped": True,
                        "index_mode": self.index_mode,
                        "engine": "index_v3",
                        "error": str(exc),
                    }
                )
            else:
                self.error.emit(str(exc))

    def request_stop(self) -> None:
        super().request_stop()
        if self._engine is not None:
            self._engine.request_stop()

    def stop(self) -> None:
        self.request_stop()

    def pause(self) -> None:
        if self._engine is not None:
            self._engine.request_pause()

    def resume(self) -> None:
        if self._engine is not None:
            self._engine.request_resume()


class QuickIndexWorker(_WorkerBase):
    """Phase 3: klasör index — Index Engine V3 COMPLETE (legacy Indexer yok)."""

    progress = Signal(dict)
    fast_done = Signal(dict)
    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(self, settings: AppSettings, folder_path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("QuickIndexWorker")
        self.settings = settings
        self.folder_path = folder_path
        self._engine = None

    def run(self) -> None:
        try:
            from pathlib import Path

            from core.db import Database
            from core.index_freeze import INDEX_FROZEN, process_search_active
            from core.index_v3 import IndexEngineV3, Mode
            from core.index_v3.ui_bridge import v3_status_dict
            from core.sources import SourceManager
            from core.utils import normalize_path

            if INDEX_FROZEN and process_search_active():
                self.finished_ok.emit(
                    {"skipped": True, "reason": "search_session", "engine": "index_v3"}
                )
                return

            folder = normalize_path(self.folder_path)
            db = Database(self.settings.db_path)
            sm = SourceManager(self.settings)
            match = None
            for s in sm.list_sources(active_only=False):
                if normalize_path(str(s.get("root_path") or "")) == folder:
                    match = s
                    break
            if match is None:
                sid = sm.add_source(
                    name=Path(folder).name or "quick",
                    root_path=folder,
                    is_active=True,
                )
                match = {"id": int(sid), "root_path": folder}
            sources = [
                {"id": int(match["id"]), "root_path": str(match.get("root_path") or folder)}
            ]
            self._engine = IndexEngineV3(
                db, settings=self.settings, use_real_extractors=True
            )
            # Fast pass
            r1 = self._engine.run(mode=Mode.FAST, sources=sources)
            st = v3_status_dict(db, [int(match["id"])])
            self.fast_done.emit(
                {"processed": int(st.get("light_done") or 0), "engine": "index_v3", **r1.worker}
            )
            if self._stop_requested:
                self.finished_ok.emit({"stopped": True, "engine": "index_v3"})
                return
            r2 = self._engine.run(
                mode=Mode.GENERAL_AI, sources=sources, walk_disk=False
            )
            self.finished_ok.emit(
                {"engine": "index_v3", "fast": r1.worker, "heavy": r2.worker}
            )
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))

    def request_stop(self) -> None:
        super().request_stop()
        if self._engine is not None:
            self._engine.request_stop()


class SearchWorker(_WorkerBase):
    finished_ok = Signal(object)
    partial = Signal(object)
    error = Signal(str)
    progress = Signal(int, int)

    def __init__(self, settings: AppSettings, query: SearchQuery, parent=None):
        super().__init__(parent)
        self.setObjectName("SearchWorker")
        self.settings = settings
        self.query = query

    def _is_visual_query(self) -> bool:
        return self.query.mode != "text" and bool(self.query.image_path)

    def _multi_image_paths(self) -> list[str]:
        extra = [str(p).strip() for p in (self.query.image_paths or []) if str(p).strip()]
        if len(extra) > 1:
            return extra
        return []

    def _use_progressive_search(self) -> bool:
        if self._multi_image_paths():
            return False
        return self._is_visual_query() and getattr(
            self.settings, "search_progressive", True
        )

    def _run_multi_image_search(self, engine) -> None:
        from core.multi_image_search import run_multi_image_queries

        paths = self._multi_image_paths()
        resp = run_multi_image_queries(
            engine,
            paths,
            base_query=self.query,
            progress=lambda i, n, _p: self.progress.emit(int(i), int(n)),
            stopped=lambda: self._stop_requested,
            on_partial=lambda r: self.partial.emit(r),
        )
        if not self._stop_requested:
            self.finished_ok.emit(resp)

    def _clone_query(self, *, fast_only: bool) -> SearchQuery:
        return SearchQuery(
            mode=self.query.mode,
            image_path=self.query.image_path,
            text=self.query.text,
            crop_rect=self.query.crop_rect,
            use_crop=self.query.use_crop,
            customer=self.query.customer,
            threshold=self.query.threshold,
            fast_only=fast_only,
        )

    def _run_preflight(self, *, query_only: bool) -> None:
        if not getattr(self.settings, "auto_preflight_index_on_search", True):
            return
        if not self.query.image_path:
            return
        from core.on_demand_scan import run_search_preflight

        run_search_preflight(
            self.settings, self.query.image_path, query_only=query_only
        )

    def run(self) -> None:
        if self._stop_requested:
            return
        from core.index_freeze import in_search_session, search_session

        if not in_search_session():
            with search_session():
                self.run()
            return
        try:
            from core.search_cache import SEARCH_RESPONSE_CACHE, search_cache_key

            cache_key = search_cache_key(self.settings, self.query)
            cached = SEARCH_RESPONSE_CACHE.get(cache_key)
            cached_response = cached
            if cached_response is None and self.query.mode == "text":
                try:
                    from core.search_memory import get_search_memory

                    cached_response = get_search_memory(
                        self.settings.db_path
                    ).get(cache_key)
                    if cached_response is not None:
                        cached = cached_response
                except Exception:
                    pass
            # Metin aramalarında önbellek "sonuç" değil "ilk ekran" olarak kullanılır.
            # Kullanıcı eski sonuçları hemen görür; worker arka planda güncel
            # indeks üzerinde aynı sorguyu yeniden çalıştırıp yeni eklenenleri
            # cache'e alır. Böylece "çiçek → gül → leopard" gibi tekrar eden
            # sorgularda UI beklemez ama indeks büyümeye devam eder.
            if cached_response is not None and self.query.mode == "text":
                if not self._stop_requested:
                    meta = dict(cached_response.meta or {})
                    meta.update({
                        "streaming": True,
                        "progressive_stage": "Önbellek",
                        "progressive_stage_key": "cache",
                        "stale_while_revalidate": True,
                        "background_refresh": True,
                        "search_cache_hit": True,
                    })
                    cached_response.meta = meta
                    self.partial.emit(cached_response)
            elif cached_response is not None:
                if not self._stop_requested:
                    self.finished_ok.emit(cached_response)
                return

            progressive = self._use_progressive_search()
            # Model yükleme asla SearchEngine ctor'da olmasın — UI GIL donması yapıyor.
            # Önce hızlı sonuçlar; AI sonra ensure_ai_loaded ile yüklenir.
            load_ai = False
            engine = SearchEngine(self.settings, load_ai=load_ai)
            ai_fallback = ""
            fast_resp = None

            if self._multi_image_paths():
                if self.settings.ai_embedding_enabled:
                    if engine.ai_index_ready():
                        engine.ensure_ai_loaded()
                self._run_multi_image_search(engine)
                return

            if progressive:
                t_start = time.perf_counter()
                fast_query = self._clone_query(fast_only=True)
                from core.search_display import visual_search_floor

                floor = visual_search_floor(self.settings)
                last_emit = 0.0

                def emit_progressive(
                    pool: list,
                    *,
                    stage: str,
                    scanned: int = 0,
                    total: int = 0,
                    fast_hash: bool = True,
                ) -> None:
                    nonlocal last_emit
                    if self._stop_requested or not pool:
                        return
                    ranked = sorted(pool, key=lambda r: float(r.score), reverse=True)
                    picked = _filter_progressive_results(ranked, stage) or ranked[:20]
                    visible = [
                        r
                        for r in picked
                        if float(r.score) >= floor
                        or r.is_self_match
                        or r.debug.get("protected_exact")
                        or getattr(r, "same_pattern_family", False)
                    ]
                    if not visible:
                        visible = picked[:20]
                    now = time.perf_counter()
                    exact_n = sum(1 for r in ranked if _is_progressive_exact(r))
                    if (
                        now - last_emit < 0.15
                        and stage not in ("Exact", "Deep Search")
                        and exact_n == 0
                    ):
                        return
                    last_emit = now
                    logger.info(
                        "progressive stage=%s exact=%s shown=%s pool=%s",
                        stage,
                        exact_n,
                        len(visible),
                        len(ranked),
                    )
                    relevant_n = sum(
                        1
                        for r in ranked
                        if float(r.score) >= floor
                        or r.is_self_match
                        or r.debug.get("protected_exact")
                        or getattr(r, "same_pattern_family", False)
                    )
                    self.partial.emit(
                        SearchResponse(
                            all_results=ranked[:80],
                            results=visible[:40],
                            stats=SearchStats(
                                total_indexed=total or len(pool),
                                selected_source_files=total or len(pool),
                                candidates_evaluated=scanned or len(pool),
                                above_threshold=relevant_n,
                                displayed=min(len(visible), 40),
                                remaining=max(0, (total or len(pool)) - (scanned or len(pool))),
                                prefilter_candidates=total or len(pool),
                                prefilter_used=True,
                            ),
                            meta={
                                "mode": fast_query.mode,
                                "text": fast_query.text,
                                "threshold": fast_query.threshold,
                                "fast_hash_only": fast_hash,
                                "streaming": True,
                                "progressive_stage": stage_label(stage),
                                "progressive_stage_key": stage,
                                "progressive_elapsed_ms": int(
                                    (time.perf_counter() - t_start) * 1000
                                ),
                                "search_mode": self.settings.search_mode,
                                "relevant_count": relevant_n,
                                "candidate_count": scanned or len(pool),
                                "retrieve_k": total or scanned or len(pool),
                                "lock_ranking": False,
                            },
                        )
                    )

                def emit_stream(scored, scanned: int, total: int) -> None:
                    stage = _progressive_stage_for_elapsed(time.perf_counter() - t_start)
                    emit_progressive(
                        list(scored),
                        stage=stage,
                        scanned=scanned,
                        total=total,
                        fast_hash=stage in ("Exact", "Pattern Family"),
                    )

                fast_resp = engine.execute_search(
                    fast_query,
                    result_callback=emit_stream,
                )
                if self._stop_requested:
                    return
                emit_progressive(
                    list(fast_resp.all_results or []),
                    stage="Exact",
                    total=int(fast_resp.stats.total_indexed or 0),
                    fast_hash=True,
                )
                if self._stop_requested:
                    return
                if _wait_progressive(t_start, 0.5, lambda: self._stop_requested):
                    emit_progressive(
                        list(fast_resp.all_results or []),
                        stage="Pattern Family",
                        total=int(fast_resp.stats.total_indexed or 0),
                        fast_hash=True,
                    )
                if self._stop_requested:
                    return
                if self.settings.ai_embedding_enabled:
                    if engine.ai_index_ready():
                        if not engine.ensure_ai_loaded():
                            ai_fallback = (
                                getattr(engine, "_ai_load_error", "") or AI_FALLBACK_MSG
                            )
                    else:
                        ai_fallback = (
                            "AI indeksi hazırlanıyor; arama hash + doku ile tamamlandı.\n"
                            f"(FAISS dino={engine.faiss.dino_count}, "
                            f"clip={engine.faiss.clip_count})"
                        )
                full_query = self._clone_query(fast_only=False)

                def full_stream(scored, scanned: int, total: int) -> None:
                    elapsed = time.perf_counter() - t_start
                    stage = _progressive_stage_for_elapsed(elapsed)
                    if elapsed < 1.0 and stage == "Deep Search":
                        stage = "Semantic"
                    emit_progressive(
                        list(scored),
                        stage=stage,
                        scanned=scanned,
                        total=total,
                        fast_hash=False,
                    )

                resp = engine.execute_search(
                    full_query,
                    result_callback=full_stream,
                )
                from core.live_scoring import stable_live_order

                resp.all_results = stable_live_order(
                    list(fast_resp.all_results or []),
                    list(resp.all_results or []),
                    tolerance=0.03,
                    lock_ranking=False,
                )
                resp.results = stable_live_order(
                    list(fast_resp.results or []),
                    list(resp.results or []),
                    tolerance=0.03,
                    lock_ranking=False,
                )
                resp.meta = {**(resp.meta or {}), "stable_order_applied": True}
                if not self._stop_requested:
                    self._run_preflight(query_only=True)
            else:
                if self._is_visual_query():
                    self._run_preflight(query_only=False)
                if self.settings.ai_embedding_enabled:
                    if engine.ai_index_ready():
                        if not engine.ensure_ai_loaded():
                            ai_fallback = (
                                getattr(engine, "_ai_load_error", "") or AI_FALLBACK_MSG
                            )
                    else:
                        ai_fallback = (
                            "AI indeksi hazırlanıyor; arama hash + doku ile tamamlandı.\n"
                            f"(FAISS dino={engine.faiss.dino_count}, "
                            f"clip={engine.faiss.clip_count})"
                        )
                resp = engine.execute_search(self.query)

            if ai_fallback:
                resp.meta = dict(resp.meta or {})
                resp.meta["ai_fallback_warning"] = ai_fallback

            SEARCH_RESPONSE_CACHE.put(cache_key, resp)
            try:
                from core.search_memory import get_search_memory

                if self.query.mode == "text":
                    get_search_memory(self.settings.db_path).put(cache_key, resp)
            except Exception:
                pass
            if cached_response is not None and self.query.mode == "text":
                resp.meta = dict(resp.meta or {})
                resp.meta.update({
                    "stale_while_revalidate": True,
                    "background_refresh": True,
                    "search_cache_hit": False,
                    "cache_refreshed": True,
                })

            if not self._stop_requested:
                self.finished_ok.emit(resp)
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))


class SourceFileCountWorker(_WorkerBase):
    """Kaynak kökünde indexlenebilir dosya sayısı — indexer/queue yok."""

    progress = Signal(int)
    finished_ok = Signal(int)
    error = Signal(str)

    def __init__(self, root_path: str, source_id: int = 0, parent=None):
        super().__init__(parent)
        self.setObjectName("SourceFileCountWorker")
        self.root_path = str(root_path or "")
        self.source_id = int(source_id or 0)

    def run(self) -> None:
        if self._stop_requested:
            return
        try:
            from core.index_v3.discovery import count_indexable_files

            total = count_indexable_files(
                self.root_path,
                on_progress=lambda n: self.progress.emit(int(n)),
                every=25,
                should_stop=lambda: bool(self._stop_requested),
            )
            if not self._stop_requested:
                self.finished_ok.emit(int(total))
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))


class BackgroundTask(_WorkerBase):
    """Run one DB/index mutation without blocking the Qt event loop."""

    finished_ok = Signal(object)
    error = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.setObjectName("BackgroundTask")
        self.function = function

    def run(self) -> None:
        if self._stop_requested:
            return
        try:
            result = self.function()
            if not self._stop_requested:
                self.finished_ok.emit(result)
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))


class PurgeMissingWorker(_WorkerBase):
    """Diskte silinmiş (missing) index kayıtlarını DB'den kaldırır."""

    progress = Signal(dict)
    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        source_id: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("PurgeMissingWorker")
        self.settings = settings
        self.source_id = source_id or None

    def run(self) -> None:
        if self._stop_requested:
            return
        try:
            mgr = SourceManager(self.settings)
            sid = self.source_id
            result = mgr.purge_missing_index(
                source_id=sid,
                progress_callback=lambda data: self.progress.emit(data),
            )
            if not self._stop_requested:
                self.finished_ok.emit(result)
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))


class StatusWorker(_WorkerBase):
    finished_ok = Signal(dict)

    def __init__(self, settings: AppSettings, parent=None, *, lanes_only: bool = False):
        super().__init__(parent)
        self.setObjectName("StatusWorker")
        self.settings = settings
        self.lanes_only = bool(lanes_only)

    def run(self) -> None:
        if self._stop_requested:
            return
        from core.app_status import get_lightweight_status, refresh_lane_counts

        if not self._stop_requested:
            if self.lanes_only:
                self.finished_ok.emit(refresh_lane_counts(self.settings))
            else:
                self.finished_ok.emit(get_lightweight_status(self.settings))


class CacheReconciliationWorker(_WorkerBase):
    """V3 arka plan full scan — gap + silinen; index drain'den bağımsız."""

    progress = Signal(dict)
    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        parent=None,
        *,
        once: bool = False,
    ):
        super().__init__(parent)
        self.setObjectName("CacheReconciliationWorker")
        self.settings = settings
        self.once = bool(once)
        self._engine = None
        self._paused_requested = False

    def request_stop(self) -> None:
        super().request_stop()
        if self._engine is not None:
            self._engine.request_stop()

    def pause(self) -> None:
        self._paused_requested = True
        if self._engine is not None:
            self._engine.pause()

    def resume(self) -> None:
        self._paused_requested = False
        if self._engine is not None:
            self._engine.resume()

    def run(self) -> None:
        try:
            from core.db import Database
            from core.index_v3.background_scan import BackgroundIndexScan
            from core.index_v3.queues import JobStore
            from core.index_v3.scope import resolve_index_scope
            from core.sources import SourceManager

            db = Database(self.settings.db_path)
            db_path = Path(str(self.settings.db_path))
            store = JobStore(db_path.with_name(db_path.stem + ".v3jobs.db"))
            interval_min = max(
                0,
                int(
                    getattr(self.settings, "background_full_scan_interval_min", 30)
                    or 0
                ),
            )
            gap_sec = float(interval_min * 60) if interval_min > 0 else 3600.0
            self._engine = BackgroundIndexScan(
                db,
                store,
                settings=self.settings,
                gap_interval_sec=gap_sec,
                walk_interval_sec=max(gap_sec, 900.0),
                chunk_size=80,
                chunk_pause_sec=0.05,
            )
            if self._stop_requested:
                self._engine.request_stop()
            if self._paused_requested:
                self._engine.pause()

            def _sources():
                sm = SourceManager(self.settings)
                selected = [
                    int(x)
                    for x in (self.settings.selected_source_ids or [])
                    if int(x) > 0
                ]
                resolved = resolve_index_scope(db, selected)
                by_id = {
                    int(s["id"]): s
                    for s in sm.list_sources(active_only=True)
                    if int(s.get("id") or 0) > 0
                }
                out = []
                for sid in resolved.source_ids:
                    row = by_id.get(int(sid))
                    if row:
                        out.append(
                            {
                                "id": int(sid),
                                "root_path": str(row.get("root_path") or ""),
                            }
                        )
                return out

            purge = bool(getattr(self.settings, "purge_missing_after_scan", True))
            if self.once:
                result = self._engine.run_once(
                    source_provider=_sources,
                    progress_callback=lambda data: self.progress.emit(data),
                    force_walk=True,
                    purge_missing=purge,
                )
            else:
                if not bool(
                    getattr(self.settings, "background_full_scan_enabled", True)
                ):
                    self.finished_ok.emit({"skipped": True, "reason": "disabled"})
                    return
                if interval_min <= 0:
                    self.finished_ok.emit({"skipped": True, "reason": "interval_0"})
                    return
                result = self._engine.run_until_stopped(
                    source_provider=_sources,
                    progress_callback=lambda data: self.progress.emit(data),
                    purge_missing=purge,
                )
            self.finished_ok.emit(result)
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))


class PurgeSourceWorker(_WorkerBase):
    """Kaynak index temizliği — UI thread'i bloklamaz."""

    progress = Signal(dict)
    finished_ok = Signal(dict)
    error = Signal(str)

    def __init__(
        self, settings: AppSettings, source_id: int, source_name: str = "", parent=None
    ):
        super().__init__(parent)
        self.setObjectName("PurgeSourceWorker")
        self.settings = settings
        self.source_id = source_id
        self.source_name = source_name

    def run(self) -> None:
        if self._stop_requested:
            return
        try:
            mgr = SourceManager(self.settings)
            result = mgr.purge_source_index(
                self.source_id,
                self.settings.cache_dir,
                progress_callback=lambda data: self.progress.emit(
                    {
                        **data,
                        "source_id": self.source_id,
                        "source_name": self.source_name,
                    }
                ),
            )
            if not self._stop_requested:
                self.finished_ok.emit(result)
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))
