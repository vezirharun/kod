"""Index Engine V3 — orchestrator (isolated from legacy Indexer)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.index_v3.discovery import discover_source, enqueue_existing_gaps
from core.index_v3.progress import ProgressSnapshot, count_progress, session_delta
from core.index_v3.queues import JobStore
from core.index_v3.scheduler import heavy_artifact_lanes
from core.index_v3.types import (
    GA_SIDE_ARTIFACTS,
    HEAVY_ARTIFACTS,
    POST_GA_ARTIFACTS,
    POST_GA_MODES,
    Artifact,
    Mode,
    QueueKind,
)
from core.index_v3.worker import ArtifactProcessor, Worker, WorkerStats
from core.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EngineReport:
    mode: str
    discovery: dict[str, Any] = field(default_factory=dict)
    worker: dict[str, Any] = field(default_factory=dict)
    progress_before: ProgressSnapshot | None = None
    progress_after: ProgressSnapshot | None = None
    session: dict[str, int] = field(default_factory=dict)


class IndexEngineV3:
    """Artifact-first engine. Does not use legacy Indexer / dual pipeline."""

    def __init__(
        self,
        db: Any,
        *,
        job_db_path: str | Path | None = None,
        processor: ArtifactProcessor | None = None,
        settings: Any | None = None,
        use_real_extractors: bool = True,
    ) -> None:
        self.db = db
        if job_db_path is None:
            base = Path(str(getattr(db, "db_path", "index_v3.db")))
            job_db_path = base.with_name(base.stem + ".v3jobs.db")
        self.store = JobStore(job_db_path)
        self.settings = settings
        if processor is not None:
            self.processor = processor
        elif settings is not None and use_real_extractors:
            from core.index_v3.real_processor import RealArtifactProcessor

            self.processor = ArtifactProcessor(
                process_fn=RealArtifactProcessor(settings).process
            )
            logger.info("V3 processor=RealArtifactProcessor")
        else:
            self.processor = ArtifactProcessor()
            logger.warning("V3 processor=stub (yalnız test); DINO/CLIP üretmez")
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._worker: Worker | None = None
        self._workers: list[Worker] = []
        self._session_baseline: ProgressSnapshot | None = None
        self._progress_cb: Callable[..., None] | None = None
        self._worker_failure: str = ""
        self.claim_timeout_sec = 180.0

    def _ocr_enabled(self) -> bool:
        s = self.settings
        if s is None:
            return False
        auto = bool(getattr(s, "auto_ocr_after_ai_final", True))
        return bool(getattr(s, "ocr_enabled", False)) and auto

    def _patch_enabled(self) -> bool:
        s = self.settings
        if s is None:
            return True
        return bool(getattr(s, "auto_patch_after_ai_final", True))

    def begin_session(self, source_ids: list[int] | None = None) -> ProgressSnapshot:
        snap = count_progress(self.db, source_ids=source_ids)
        self._session_baseline = snap
        return snap

    def request_stop(self) -> None:
        self._stop.set()
        self._pause.set()
        for worker in list(self._workers) or ([self._worker] if self._worker else []):
            worker.request_stop()

    def request_pause(self) -> None:
        self._pause.clear()
        for worker in list(self._workers) or ([self._worker] if self._worker else []):
            worker.request_pause()

    def request_resume(self) -> None:
        self._pause.set()
        for worker in list(self._workers) or ([self._worker] if self._worker else []):
            worker.request_resume()

    def reset_stale(self) -> int:
        return self.store.reset_stale_claims()

    def reclaim_stale_claims(self) -> int:
        return self.store.requeue_stale_claims(self.claim_timeout_sec)

    def include_source_id(self, source_id: int) -> None:
        """Çalışan oturuma yeni aktif kaynak ekle — SSOT/claim aynı listede büyür."""
        sid = int(source_id or 0)
        if sid <= 0:
            return
        ids = getattr(self, "_session_source_ids", None)
        if ids is None:
            self._session_source_ids = [sid]
            return
        if sid not in ids:
            ids.append(sid)

    def drop_source_id(self, source_id: int) -> None:
        sid = int(source_id or 0)
        ids = getattr(self, "_session_source_ids", None)
        if not ids or sid <= 0:
            return
        try:
            ids.remove(sid)
        except ValueError:
            pass

    def run(
        self,
        *,
        mode: Mode | str,
        sources: list[dict[str, Any]],
        walk_disk: bool | None = None,
        max_jobs_per_queue: int | None = None,
        progress_callback: Callable[..., None] | None = None,
    ) -> EngineReport:
        from core.index_freeze import (
            INDEX_FROZEN,
            allow_index_writes,
            process_search_active,
        )

        if INDEX_FROZEN:
            while process_search_active() and not self._stop.is_set():
                self._stop.wait(0.1)
            if self._stop.is_set() and process_search_active():
                return EngineReport(
                    mode=str(mode),
                    worker={"skipped": True, "reason": "search_session"},
                )

        with allow_index_writes():
            return self._run_inner(
                mode=mode,
                sources=sources,
                walk_disk=walk_disk,
                max_jobs_per_queue=max_jobs_per_queue,
                progress_callback=progress_callback,
            )

    def _run_inner(
        self,
        *,
        mode: Mode | str,
        sources: list[dict[str, Any]],
        walk_disk: bool | None = None,
        max_jobs_per_queue: int | None = None,
        progress_callback: Callable[..., None] | None = None,
    ) -> EngineReport:
        mode = Mode(mode) if not isinstance(mode, Mode) else mode
        self._stop.clear()
        self._pause.set()
        self.reset_stale()
        self.reclaim_stale_claims()
        self._progress_cb = progress_callback

        source_ids = [int(s["id"]) for s in sources if int(s.get("id") or 0) > 0]
        self._session_source_ids = source_ids
        try:
            with self.db.connect() as conn:
                known = [
                    int(r[0])
                    for r in conn.execute(
                        "SELECT id FROM files WHERE status NOT IN ('excluded_internal','missing')"
                    ).fetchall()
                ]
            dropped = self.store.purge_unknown_file_ids(known)
            if dropped:
                logger.warning("V3 purged orphan jobs for missing file_id: %s", dropped)
        except Exception as exc:
            logger.warning("V3 orphan job purge failed: %s", exc)

        # Fiziksel cache reconcile kritik path'te DEĞİL: ~49k dosya stat
        # IndexWorker'ı bloke edip defer_walk / light / heavy spawn'ı geciktiriyordu.
        # StatusWorker zaten throttle'lı reconcile yapar; burada arka planda tamamlanır.
        if source_ids:
            _sid_bg = list(source_ids)
            _cache_bg = (
                str(getattr(self.settings, "cache_dir", "") or "") or None
            )

            def _bg_physical_reconcile() -> None:
                try:
                    from core.index_v3.physical_reconcile import (
                        reconcile_stale_physical_flags,
                    )

                    logger.info("V3 physical-reconcile STARTED (background)")
                    reconcile_stale_physical_flags(
                        self.db, _sid_bg, cache_dir=_cache_bg
                    )
                    logger.info("V3 physical-reconcile DONE (background)")
                except Exception as exc:
                    logger.warning(
                        "V3 physical reconcile background failed: %s", exc
                    )

            threading.Thread(
                target=_bg_physical_reconcile,
                name="v3-physical-reconcile",
                daemon=True,
            ).start()

        if mode == Mode.REPAIR and source_ids:
            try:
                self.store.reopen_failed_permanent(source_ids=source_ids)
            except Exception:
                pass
        before = self.begin_session(source_ids)
        report = EngineReport(mode=mode.value, progress_before=before)

        # Full gap/missing scan arka planda (BackgroundIndexScan).
        # Index oturumu kuyruğu işler — full scan beklemez / kilitlemez.
        # Explicit index runs must reconcile the source tree by default.
        # Background recovery remains free to pass walk_disk=False. This keeps
        # new files discoverable without forcing the operator to know about an
        # internal switch, while batch discovery makes the scan cheap.
        do_walk = bool(walk_disk) if walk_disk is not None else True
        if mode in POST_GA_MODES:
            do_walk = False

        disc: dict[str, Any] = {"sources": []}
        walk_done = threading.Event()
        # Production: walk arka planda; worker lane'ler mevcut kuyruğu hemen işler.
        # max_jobs_per_queue (test) → senkron walk, deterministik drain.
        defer_walk = bool(
            do_walk
            and mode not in POST_GA_MODES
            and max_jobs_per_queue is None
        )

        def _start_deferred_walk(thread_name: str) -> None:
            def _walk() -> None:
                try:
                    for src in sources:
                        if self._stop.is_set():
                            break
                        _discover_one(src)
                    report.discovery = disc
                finally:
                    walk_done.set()

            logger.info("V3 %s STARTED (defer_walk)", thread_name)
            threading.Thread(
                target=_walk, name=thread_name, daemon=True
            ).start()

        def _discovery_still_running() -> bool:
            return bool(defer_walk and not walk_done.is_set())

        def _discover_one(src: dict[str, Any]) -> None:
            sid = int(src.get("id") or 0)
            root = str(src.get("root_path") or "")
            if not sid:
                return
            if do_walk and root:
                def _cb(payload: dict[str, Any], _sid: int = sid) -> None:
                    if self._progress_cb is None:
                        return
                    info = dict(payload or {})
                    info["phase"] = "discovery"
                    info["source_id"] = _sid
                    self._progress_cb(info)

                st = discover_source(
                    self.db,
                    self.store,
                    source_id=sid,
                    root_path=root,
                    mode=mode,
                    mark_missing=True,
                    ocr_enabled=self._ocr_enabled(),
                    patch_enabled=self._patch_enabled(),
                    settings=self.settings,
                    progress_callback=_cb if defer_walk else None,
                )
                disc["sources"].append(
                    {
                        "source_id": sid,
                        "scanned": st.scanned,
                        "inserted": st.inserted,
                        "changed": st.changed,
                        "jobs": st.jobs_enqueued,
                        "missing": st.missing_marked,
                    }
                )
            else:
                seeded = 0
                after_id = 0
                for _ in range(3):
                    n, after_id = enqueue_existing_gaps(
                        self.db,
                        self.store,
                        source_id=sid,
                        mode=mode,
                        limit=100,
                        after_id=after_id,
                        ocr_enabled=self._ocr_enabled(),
                        patch_enabled=self._patch_enabled(),
                        settings=getattr(self, "settings", None),
                    )
                    seeded += int(n)
                    if not after_id:
                        break
                disc["sources"].append(
                    {
                        "source_id": sid,
                        "jobs": seeded,
                        "db_only": True,
                        "seeded": True,
                    }
                )

        if mode not in POST_GA_MODES and not defer_walk:
            for src in sources:
                _discover_one(src)
            walk_done.set()
        elif mode in POST_GA_MODES:
            walk_done.set()
        report.discovery = disc

        def _scoped_ga_pending() -> int:
            return int(
                self.store.count_pending(
                    QueueKind.HEAVY,
                    source_ids=source_ids,
                    artifacts=HEAVY_ARTIFACTS,
                )
                + self.store.count_pending(
                    QueueKind.REPAIR,
                    source_ids=source_ids,
                    artifacts=HEAVY_ARTIFACTS,
                )
            )

        _ga_ssot_at = [0.0]
        _ga_complete_cached = [False]

        def _ga_pool_complete() -> bool:
            """Otomatik PATCH/OCR: preview havuzunun tamamı AI Final olmalı.

            Preview'siz dosyalardaki HASH…DNA pending satırları havuzu
            'bitmedi' saymaz; aksi halde Object Index gece boyunca
            dep_wait HASH claim döngüsünde aç kalır.
            """
            now = time.monotonic()
            if now - _ga_ssot_at[0] < 5.0:
                return bool(_ga_complete_cached[0])
            try:
                from core.index_v3.ui_bridge import count_v3_ssot

                ss = count_v3_ssot(self.db, source_ids)
                preview = int(ss.get("preview") or 0)
                ok = preview > 0 and int(ss.get("ai_final") or 0) >= preview
            except Exception:
                ok = False
            _ga_ssot_at[0] = now
            _ga_complete_cached[0] = ok
            return ok

        def _post_ga_arts() -> tuple[Artifact, ...]:
            if mode == Mode.PATCH:
                return (Artifact.PATCH,)
            if mode == Mode.OCR:
                return (Artifact.OCR,)
            return POST_GA_ARTIFACTS

        def _plan_gaps(*, limit: int | None = None) -> int:
            auto_post = mode != Mode.GENERAL_AI or _ga_pool_complete()
            n = 0
            for sid in source_ids:
                r = enqueue_existing_gaps(
                    self.db,
                    self.store,
                    source_id=sid,
                    mode=mode,
                    limit=limit,
                    ocr_enabled=self._ocr_enabled() and auto_post,
                    patch_enabled=self._patch_enabled() and auto_post,
                    settings=getattr(self, "settings", None),
                )
                if isinstance(r, tuple):
                    n += int(r[0])
                else:
                    n += int(r)
            return n

        # V3 dual-lane: Light/Preview ve Heavy aynı oturumda paralel ilerler.
        # Heavy lane yalnızca preview-ready gap'lerini re-plan ederek alır.
        workers: list[Worker] = []
        worker_stats_lock = threading.Lock()
        wstats = WorkerStats()
        lane_errors: list[str] = []
        lane_error_lock = threading.Lock()

        def _merge_stats(st: WorkerStats) -> None:
            with worker_stats_lock:
                wstats.processed += st.processed
                wstats.completed += st.completed
                wstats.failed += st.failed
                wstats.skipped += st.skipped
                wstats.timeouts += st.timeouts
                for k, v in st.model_calls.items():
                    wstats.model_calls[k] = wstats.model_calls.get(k, 0) + v
                for k, v in st.errors_by_ext.items():
                    wstats.errors_by_ext[k] = wstats.errors_by_ext.get(k, 0) + v

        def _record_lane_error(worker_id: str, exc: BaseException) -> None:
            message = f"{worker_id}: {exc}" or worker_id
            with lane_error_lock:
                lane_errors.append(message)
            logger.exception("V3 lane crashed worker=%s", worker_id)
            reclaimed = self.store.requeue_claims_for_worker(
                worker_id, error="worker_dead"
            )
            if self._progress_cb is not None:
                try:
                    self._progress_cb(
                        {
                            "phase": "worker_error",
                            "worker": worker_id,
                            "error": message,
                            "reclaimed": reclaimed,
                        }
                    )
                except Exception:
                    logger.exception("V3 worker error progress emit failed")

        def _make_worker(worker_id: str) -> Worker:
            worker = Worker(
                self.db,
                self.store,
                processor=self.processor,
                worker_id=worker_id,
                claim_timeout_sec=self.claim_timeout_sec,
                ocr_enabled=self._ocr_enabled(),
                patch_enabled=self._patch_enabled(),
                session_mode=mode,
            )
            if self._progress_cb is not None:
                worker.set_progress_callback(self._progress_cb)
            workers.append(worker)
            return worker

        light_worker = _make_worker("v3-light")
        self._workers = workers
        self._worker = light_worker

        lease_monitor_stop = threading.Event()

        def _lease_monitor() -> None:
            while not lease_monitor_stop.wait(5.0):
                if self._stop.is_set():
                    return
                reclaimed = self.store.requeue_stale_claims(
                    self.claim_timeout_sec
                )
                if reclaimed:
                    message = f"stale claims reclaimed: {reclaimed}"
                    logger.warning("V3 %s", message)
                    if self._progress_cb is not None:
                        try:
                            self._progress_cb(
                                {
                                    "phase": "stale_reclaimed",
                                    "error": message,
                                    "reclaimed": reclaimed,
                                }
                            )
                        except Exception:
                            logger.exception("V3 stale reclaim progress emit failed")

        lease_monitor = threading.Thread(
            target=_lease_monitor,
            name="v3-lease-monitor",
            daemon=True,
        )
        lease_monitor.start()

        def _drain(
            worker: Worker,
            queues: list[QueueKind],
            artifacts: tuple | None = None,
        ) -> None:
            for q in queues:
                if self._stop.is_set():
                    break
                st = worker.run_queue(
                    q,
                    source_ids,
                    max_jobs=max_jobs_per_queue,
                    artifacts=artifacts,
                )
                _merge_stats(st)
                reclaimed = self.reclaim_stale_claims()
                if reclaimed:
                    logger.warning(
                        "V3 reclaimed stale claims after worker drain: %s",
                        reclaimed,
                    )
                if self._progress_cb is not None:
                    try:
                        self._progress_cb()
                    except Exception:
                        pass

        _LIGHT_DRAIN_SLICE = 6
        _GAP_PLAN_CHUNK = 120

        def _light_queue_pending(q: QueueKind) -> int:
            return int(self.store.count_pending(q, source_ids=source_ids))

        def _drain_light_round(*, preview_first: bool) -> None:
            """Kısa dilimlerle PREVIEW/LIGHT — tek kuyruk diğerini aç bırakmasın."""
            try:
                released = self.store.maybe_release_jumbo_phase(
                    source_ids=source_ids
                )
                if released:
                    logger.info(
                        "V3 jumbo phase release: %s light jobs now claimable",
                        released,
                    )
            except Exception as exc:
                logger.warning("V3 jumbo phase release failed: %s", exc)
            order = (
                (QueueKind.PREVIEW, QueueKind.LIGHT)
                if preview_first
                else (QueueKind.LIGHT, QueueKind.PREVIEW)
            )
            cap = (
                _LIGHT_DRAIN_SLICE
                if max_jobs_per_queue is None
                else max_jobs_per_queue
            )
            for q in order:
                if self._stop.is_set():
                    break
                if _light_queue_pending(q) <= 0:
                    continue
                st = light_worker.run_queue(q, source_ids, max_jobs=cap)
                _merge_stats(st)
                reclaimed = self.reclaim_stale_claims()
                if reclaimed:
                    logger.warning(
                        "V3 reclaimed stale claims after light round: %s",
                        reclaimed,
                    )
                if self._progress_cb is not None:
                    try:
                        self._progress_cb()
                    except Exception:
                        pass

        def _heavy_worker_count() -> int:
            # Isolated extractor A/B was +22.9% pipeline-parallel. The same
            # 4-wide layout on IndexEngineV3 measured -1.8% (job/SQLite/FAISS
            # + DINO/Patch still serial on one model). Do not open extra
            # heavy lanes until that wall improves. settings.worker_count
            # remains unused here on purpose (Fast/Search/FAISS unchanged).
            # VEZIR_HEAVY_WORKERS=2 is bench-only (never 4).
            raw = str(os.environ.get("VEZIR_HEAVY_WORKERS") or "").strip()
            if raw == "2":
                return 2
            return 1

        heavy_lanes = heavy_artifact_lanes(_heavy_worker_count())
        heavy_workers: list[tuple[Worker, tuple | None]] = [
            (_make_worker(name), arts) for name, arts in heavy_lanes
        ]

        # Test/diagnostic çağrılarında max_jobs verilmişse deterministik eski
        # sıralı davranışı koru; production run'da iki lane birlikte çalışır.
        if max_jobs_per_queue is not None:
            if mode in (Mode.FAST, Mode.COMPLETE, Mode.REPAIR):
                _drain(light_worker, [QueueKind.LIGHT, QueueKind.PREVIEW])
                _plan_gaps()
                _drain(light_worker, [QueueKind.LIGHT, QueueKind.PREVIEW])
            if mode in POST_GA_MODES:
                for hw, _arts in heavy_workers:
                    _drain(
                        hw,
                        [QueueKind.HEAVY, QueueKind.REPAIR],
                        artifacts=_post_ga_arts(),
                    )
            if mode in (Mode.GENERAL_AI, Mode.COMPLETE, Mode.REPAIR):
                _plan_gaps()
                for hw, arts in heavy_workers:
                    drain_arts = (
                        HEAVY_ARTIFACTS + GA_SIDE_ARTIFACTS
                        if mode == Mode.GENERAL_AI
                        else arts
                    )
                    _drain(hw, [QueueKind.HEAVY, QueueKind.REPAIR], artifacts=drain_arts)
                if mode == Mode.GENERAL_AI and _ga_pool_complete():
                    for hw, _arts in heavy_workers:
                        _drain(
                            hw,
                            [QueueKind.HEAVY, QueueKind.REPAIR],
                            artifacts=POST_GA_ARTIFACTS + GA_SIDE_ARTIFACTS,
                        )
                _plan_gaps()
                for hw, arts in heavy_workers:
                    drain_arts = (
                        HEAVY_ARTIFACTS + GA_SIDE_ARTIFACTS
                        if mode == Mode.GENERAL_AI
                        else arts
                    )
                    _drain(hw, [QueueKind.HEAVY, QueueKind.REPAIR], artifacts=drain_arts)
                if mode == Mode.GENERAL_AI and _ga_pool_complete():
                    for hw, _arts in heavy_workers:
                        _drain(
                            hw,
                            [QueueKind.HEAVY, QueueKind.REPAIR],
                            artifacts=POST_GA_ARTIFACTS + GA_SIDE_ARTIFACTS,
                        )
        elif mode == Mode.FAST:
            if defer_walk:
                _start_deferred_walk("v3-discover")
            # Preview kuyruğu + Thumbnail. Hash Genel AI'dedir; burada planlanmaz.
            def _scoped_light_pending() -> int:
                return int(
                    self.store.count_pending(
                        QueueKind.LIGHT, source_ids=source_ids
                    )
                    + self.store.count_pending(
                        QueueKind.PREVIEW, source_ids=source_ids
                    )
                )

            idle_empty = 0
            while not self._stop.is_set():
                from core.index_freeze import INDEX_FROZEN, process_search_active

                if INDEX_FROZEN and process_search_active():
                    self._stop.wait(0.1)
                    continue
                if _scoped_light_pending() == 0:
                    if not walk_done.is_set():
                        idle_empty = 0
                        self._stop.wait(0.15)
                        continue
                    planned = _plan_gaps(limit=300)
                    if planned > 0 or _scoped_light_pending() > 0:
                        idle_empty = 0
                        continue
                    _plan_gaps()
                    if _scoped_light_pending() > 0:
                        idle_empty = 0
                        continue
                    self._stop.wait(0.5)
                    if _scoped_light_pending() == 0:
                        idle_empty += 1
                        if idle_empty >= 2:
                            break
                        continue
                idle_empty = 0
                # Pending varken önce claim/drain — gap plan claim'i bloke etmesin.
                _drain_light_round(preview_first=True)
                _plan_gaps(limit=_GAP_PLAN_CHUNK)
            walk_done.wait(60)
            report.discovery = disc
        elif mode in POST_GA_MODES:
            pg_arts = _post_ga_arts()
            hw = heavy_workers[0][0] if heavy_workers else light_worker

            def _pg_pending() -> int:
                return int(
                    self.store.count_pending(
                        QueueKind.HEAVY,
                        source_ids=source_ids,
                        artifacts=pg_arts,
                    )
                    + self.store.count_pending(
                        QueueKind.REPAIR,
                        source_ids=source_ids,
                        artifacts=pg_arts,
                    )
                )

            idle_empty = 0
            while not self._stop.is_set():
                from core.index_freeze import INDEX_FROZEN, process_search_active

                if INDEX_FROZEN and process_search_active():
                    self._stop.wait(0.1)
                    continue
                if _pg_pending() == 0:
                    self._stop.wait(0.5)
                    idle_empty += 1
                    if idle_empty >= 2:
                        break
                    continue
                idle_empty = 0
                _drain(
                    hw,
                    [QueueKind.HEAVY, QueueKind.REPAIR],
                    artifacts=pg_arts,
                )
        elif mode in (Mode.GENERAL_AI, Mode.COMPLETE, Mode.REPAIR):
            if defer_walk:
                _start_deferred_walk("v3-discover")
            light_done = threading.Event()

            def _scoped_light_pending() -> int:
                return int(
                    self.store.count_pending(
                        QueueKind.LIGHT, source_ids=source_ids
                    )
                    + self.store.count_pending(
                        QueueKind.PREVIEW, source_ids=source_ids
                    )
                )

            def _light_lane() -> None:
                try:
                    if mode in (Mode.COMPLETE, Mode.REPAIR):
                        idle_empty = 0
                        while not self._stop.is_set():
                            from core.index_freeze import (
                                INDEX_FROZEN,
                                process_search_active,
                            )

                            if INDEX_FROZEN and process_search_active():
                                self._stop.wait(0.1)
                                continue
                            if _scoped_light_pending() == 0:
                                if _discovery_still_running():
                                    idle_empty = 0
                                    self._stop.wait(0.15)
                                    continue
                                planned = _plan_gaps(limit=300)
                                if planned > 0 or _scoped_light_pending() > 0:
                                    idle_empty = 0
                                    continue
                                _plan_gaps()
                                if _scoped_light_pending() > 0:
                                    idle_empty = 0
                                    continue
                                self._stop.wait(0.5)
                                if _scoped_light_pending() == 0:
                                    idle_empty += 1
                                    if idle_empty >= 2:
                                        break
                                    continue
                            idle_empty = 0
                            # Mevcut preview/light pending önce claim edilsin.
                            _drain_light_round(preview_first=False)
                            _plan_gaps(limit=_GAP_PLAN_CHUNK)
                except BaseException as exc:
                    _record_lane_error(light_worker.worker_id, exc)
                finally:
                    light_done.set()

            def _scoped_heavy_pending() -> int:
                return int(
                    self.store.count_pending(
                        QueueKind.HEAVY, source_ids=source_ids
                    )
                    + self.store.count_pending(
                        QueueKind.REPAIR, source_ids=source_ids
                    )
                )

            def _ga_side_pending() -> int:
                """OWLv2 + Object Index + PATCH/OCR. HASH…DNA sayılmaz.

                Heavy lane must not exit while these rows are pending (including
                dep_wait). Otherwise the index QThread ends, workers=0, and
                OWL stalls until the user restarts the GUI.
                """
                arts = POST_GA_ARTIFACTS + GA_SIDE_ARTIFACTS
                return int(
                    self.store.count_pending(
                        QueueKind.HEAVY,
                        source_ids=source_ids,
                        artifacts=arts,
                    )
                    + self.store.count_pending(
                        QueueKind.REPAIR,
                        source_ids=source_ids,
                        artifacts=arts,
                    )
                )

            def _heavy_claim_arts(lane_arts: tuple | None):
                if mode != Mode.GENERAL_AI:
                    return lane_arts
                if not _ga_pool_complete():
                    return HEAVY_ARTIFACTS + GA_SIDE_ARTIFACTS
                return POST_GA_ARTIFACTS + GA_SIDE_ARTIFACTS

            def _heavy_lane(worker: Worker, artifacts: tuple | None) -> None:
                idle_empty = 0
                try:
                    while not self._stop.is_set():
                        from core.index_freeze import (
                            INDEX_FROZEN,
                            process_search_active,
                        )

                        if INDEX_FROZEN and process_search_active():
                            self._stop.wait(0.1)
                            continue
                        claim_arts = _heavy_claim_arts(artifacts)
                        lane_pending = int(
                            self.store.count_pending(
                                QueueKind.HEAVY,
                                source_ids=source_ids,
                                artifacts=claim_arts,
                            )
                            + self.store.count_pending(
                                QueueKind.REPAIR,
                                source_ids=source_ids,
                                artifacts=claim_arts,
                            )
                        )
                        all_pending = _scoped_heavy_pending()
                    # GENERAL_AI tek başına çalışıyorsa ve light lane yoksa,
                    # preview bekleyen heavy işleri sonsuza kadar yeniden claim etme.
                    # Hazır heavy işler varsa normal şekilde devam eder; yalnızca
                    # bütün kalan işlerin preview bağımlılığı varsa oturum biter.
                        if (
                            mode == Mode.GENERAL_AI
                            and light_done.is_set()
                            and lane_pending > 0
                        ):
                            try:
                                from core.index_v3.ui_bridge import count_v3_ssot as _count_ssot

                                _ss = _count_ssot(self.db, source_ids)
                                waiting_preview = int(_ss.get("waiting_preview") or 0)
                                preview_ready = int(_ss.get("preview") or 0)
                                total_files = int(_ss.get("total") or 0)
                                extra_n = _ga_side_pending()
                                # Preview'siz HASH…DNA bekleyebilir; Object Index /
                                # OWLv2 / PATCH hazır Preview üzerinde devam eder.
                                # extra_n>0 (OWL dep_wait dahil) iken oturumu kapatma.
                                if (
                                    waiting_preview > 0
                                    and preview_ready < total_files
                                    and int(_ss.get("ai_final") or 0) >= preview_ready
                                    and extra_n == 0
                                ):
                                    break
                            except Exception:
                                pass
                        if lane_pending == 0:
                            if _discovery_still_running():
                                idle_empty = 0
                                self._stop.wait(0.15)
                                continue
                            if mode == Mode.GENERAL_AI:
                                _plan_gaps(
                                    limit=None if light_done.is_set() else 300
                                )
                                ga_done = _ga_pool_complete()
                                if _scoped_ga_pending() > 0 and not ga_done:
                                    idle_empty = 0
                                    continue
                                if _ga_side_pending() > 0:
                                    idle_empty = 0
                                    self._stop.wait(0.5)
                                    continue
                                self._stop.wait(0.5)
                                idle_empty += 1
                                if idle_empty >= 2:
                                    break
                                continue
                            if all_pending > 0:
                                if (
                                    mode == Mode.GENERAL_AI
                                    and _ga_pool_complete()
                                    and _ga_side_pending() == 0
                                ):
                                    break
                                idle_empty = 0
                                self._stop.wait(0.05)
                                continue
                            if light_done.is_set():
                                _plan_gaps()
                                if _scoped_heavy_pending() > 0 or _ga_side_pending() > 0:
                                    idle_empty = 0
                                    continue
                                self._stop.wait(0.5)
                                if (
                                    _scoped_heavy_pending() == 0
                                    and _ga_side_pending() == 0
                                ):
                                    idle_empty += 1
                                    if idle_empty >= 2:
                                        break
                                    continue
                            else:
                                _plan_gaps(limit=300)
                                if _scoped_heavy_pending() > 0:
                                    idle_empty = 0
                                    continue
                                self._stop.wait(0.25)
                                continue
                        idle_empty = 0
                        _drain(
                            worker,
                            [QueueKind.HEAVY, QueueKind.REPAIR],
                            artifacts=claim_arts,
                        )
                        # Claimable işler drain edildikten sonra bounded gap plan.
                        _plan_gaps(limit=_GAP_PLAN_CHUNK)
                except BaseException as exc:
                    _record_lane_error(worker.worker_id, exc)

            light_thread = threading.Thread(
                target=_light_lane,
                name="v3-light-lane",
                daemon=True,
            )
            heavy_threads = [
                threading.Thread(
                    target=_heavy_lane,
                    args=(hw, arts),
                    name=hw.worker_id,
                    daemon=True,
                )
                for hw, arts in heavy_workers
            ]
            logger.info(
                "V3 light/heavy lanes STARTED mode=%s defer_walk=%s heavy=%s",
                mode.value,
                defer_walk,
                len(heavy_threads),
            )
            light_thread.start()
            for th in heavy_threads:
                th.start()
            light_thread.join()
            for th in heavy_threads:
                th.join()
            if defer_walk:
                walk_done.wait(60)
                report.discovery = disc

            # Retry immediately değil: hata alan iş kuyruğun SONUNA gider
            # (tail_seq). Mevcut kuyruk biter, sonra tekrar denenir. 3. hatada
            # failed_permanent olur ve otomatik kuyruğa alınmaz.

        lease_monitor_stop.set()
        lease_monitor.join(timeout=2.0)
        self._worker = light_worker

        # A worker lane is executed synchronously in the engine thread or in a
        # managed lane thread. run_queue returning is normal only after its
        # queue is drained; any claimed row left behind is reclaimed here.
        reclaimed = self.reclaim_stale_claims()
        if reclaimed:
            logger.warning("V3 reclaimed stale claims at engine finalization: %s", reclaimed)

        if self.settings is not None:
            try:
                from core.index_v3.faiss_sync import ensure_consistent, flush_cached_faiss

                flush_cached_faiss()
                ensure_consistent(self.db, self.settings)
            except Exception:
                pass

        report.worker = {
            "processed": wstats.processed,
            "completed": wstats.completed,
            "failed": wstats.failed,
            "skipped": wstats.skipped,
            "timeouts": wstats.timeouts,
            "errors_by_ext": dict(wstats.errors_by_ext),
            "model_calls": dict(wstats.model_calls),
            "pending": self.store.count_pending(),
            "claimed": self.store.count_claimed(),
            "worker_errors": list(lane_errors),
        }
        after = count_progress(self.db, source_ids=source_ids)
        report.progress_after = after
        if self._session_baseline:
            report.session = session_delta(after, self._session_baseline)
        return report


def _queues_for_mode(mode: Mode) -> list[QueueKind]:
    if mode == Mode.FAST:
        return [QueueKind.LIGHT, QueueKind.PREVIEW]
    if mode == Mode.GENERAL_AI:
        return [QueueKind.HEAVY]
    if mode in POST_GA_MODES:
        return [QueueKind.HEAVY, QueueKind.REPAIR]
    if mode == Mode.REPAIR:
        return [
            QueueKind.REPAIR,
            QueueKind.LIGHT,
            QueueKind.PREVIEW,
            QueueKind.HEAVY,
        ]
    return [QueueKind.LIGHT, QueueKind.PREVIEW, QueueKind.HEAVY]
