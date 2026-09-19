"""Index Engine V3 — worker: CLAIM → PROCESS → SAVE → VERIFY → COMPLETE."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.index_v3.artifact_state import assess_file
from core.index_v3.planner import plan_jobs_for_file, reconcile_stale_pendings
from core.index_v3.queues import JobStore
from core.index_v3.scheduler import claim_fair
from core.index_v3.types import (
    ARTIFACT_DEPENDENCIES,
    HEAVY_ARTIFACTS,
    POST_GA_ARTIFACTS,
    Artifact,
    Job,
    Mode,
    QueueKind,
)
from core.logger import setup_logger

logger = setup_logger(__name__)


def unstick_owlv2_dep_wait_if_preview_ready(
    store: JobStore,
    db: Any,
    queue: QueueKind,
    source_ids: list[int] | None,
    artifacts: tuple[Artifact, ...] | list[Artifact] | None = None,
) -> int:
    """Clear stale OWLv2 dep_wait when Preview is READY. Does not claim."""
    if artifacts is not None:
        arts = [
            a if isinstance(a, Artifact) else Artifact(str(a)) for a in artifacts
        ]
        if Artifact.OWLV2 not in arts:
            return 0
    fids = store.list_pending_owl_dep_wait(
        queue, source_ids=source_ids, limit=10000
    )
    if not fids:
        return 0
    ready: list[int] = []
    with db.connect() as conn:
        for i in range(0, len(fids), 400):
            chunk = fids[i : i + 400]
            ph = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT id FROM files WHERE id IN ({ph}) "
                "AND ifnull(feature_preview_path,'')!='' "
                "AND ifnull(physical_preview_ready,0)=1",
                chunk,
            ).fetchall()
            for row in rows:
                ready.append(int(row["id"] if hasattr(row, "keys") else row[0]))
    if not ready:
        return 0
    return store.clear_stale_owl_dep_wait(ready)

_TIMED_ARTIFACTS = {
    Artifact.HASH,
    Artifact.METADATA,
    Artifact.DINO,
    Artifact.CLIP,
    Artifact.TEXTURE,
    Artifact.SEMANTIC,
    Artifact.DNA,
    Artifact.OBJECT_CONCEPT,
    Artifact.OWLV2,
    Artifact.PATCH,
    Artifact.OCR,
}

DEFAULT_CLAIM_TIMEOUT_SEC = 180.0
CLAIM_HEARTBEAT_INTERVAL_SEC = 5.0
IDLE_GRACE_SEC = 2.0
IDLE_POLL_SEC = 0.10


def _isolate_ocr_failure(db: Any, file_id: int, error: str) -> None:
    """OCR fail → yalnız OCR kolonları. files.error_msg / heavy_status yok."""
    msg = str(error or "").strip() or "ocr_failed:unknown"
    if not msg.startswith("ocr_failed:"):
        msg = f"ocr_failed:{msg}"
    with db.connect() as conn:
        conn.execute(
            "UPDATE files SET ocr_processed=0, ocr_error=?, "
            "updated_at=datetime('now') WHERE id=?",
            (msg[:500], int(file_id)),
        )


def _isolate_patch_failure(db: Any, file_id: int, error: str) -> None:
    """PATCH fail → yalnız patch_error. Genel AI / OCR dokunulmaz."""
    msg = str(error or "").strip() or "patch_failed:unknown"
    if not msg.startswith("patch_failed:"):
        msg = f"patch_failed:{msg}"
    try:
        with db.connect() as conn:
            conn.execute(
                "UPDATE files SET patch_error=?, updated_at=datetime('now') WHERE id=?",
                (msg[:500], int(file_id)),
            )
    except Exception:
        logger.debug("patch_error column missing; job-store isolation still applies")


def _isolate_post_ga_failure(db: Any, job: Job, error: str) -> None:
    if job.artifact == Artifact.OCR:
        _isolate_ocr_failure(db, job.file_id, error)
    elif job.artifact == Artifact.PATCH:
        _isolate_patch_failure(db, job.file_id, error)


@dataclass
class WorkerStats:
    processed: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    timeouts: int = 0
    model_calls: dict[str, int] = field(default_factory=dict)
    errors_by_ext: dict[str, int] = field(default_factory=dict)
    file_read_ms: float = 0.0

    def bump_model(self, name: str) -> None:
        self.model_calls[name] = int(self.model_calls.get(name, 0) or 0) + 1


class ArtifactProcessor:
    """Pluggable processors — tests inject fakes; production wraps libraries."""

    def __init__(
        self,
        *,
        process_fn: Callable[[Any, Job, WorkerStats], bool] | None = None,
    ) -> None:
        self._process_fn = process_fn

    def process(self, db: Any, job: Job, stats: WorkerStats) -> bool:
        if self._process_fn is not None:
            return bool(self._process_fn(db, job, stats))
        return _default_process(db, job, stats)


def _default_process(db: Any, job: Job, stats: WorkerStats) -> bool:
    """Minimal safe stub: synthetic artifacts for isolated tests / dry-run.

    Does not call DINO/CLIP models. Production wiring will inject real processors.
    """
    row = db.get_file_by_id(job.file_id) or {}
    feat = db.get_features(job.file_id) or {}
    art = job.artifact
    stats.bump_model(art.value)

    if art == Artifact.THUMBNAIL:
        cache_dir = Path(str(row.get("path") or "x")).resolve().parent / ".v3_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        thumb = str(cache_dir / f"{job.file_id}_thumb.webp")
        if not Path(thumb).exists():
            Path(thumb).write_bytes(b"WEBP_STUB")
        db.upsert_file(
            {
                "path": row["path"],
                "thumbnail_path": thumb,
                "width": 64,
                "height": 64,
            }
        )
        prev = str(row.get("feature_preview_path") or "").strip()
        prev_ok = bool(prev) and Path(prev).is_file()
        db.update_physical_readiness(
            job.file_id,
            thumbnail_ready=True,
            preview_ready=prev_ok,
            requeue_missing=False,
        )
        return True

    if art == Artifact.PREVIEW:
        cache_dir = Path(str(row.get("path") or "x")).resolve().parent / ".v3_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        prev = str(cache_dir / f"{job.file_id}_prev.webp")
        if not Path(prev).exists():
            Path(prev).write_bytes(b"WEBP_STUB")
        db.upsert_file(
            {
                "path": row["path"],
                "feature_preview_path": prev,
                "feature_preview_version": 1,
            }
        )
        thumb = str(row.get("thumbnail_path") or "").strip()
        thumb_ok = bool(thumb) and Path(thumb).is_file()
        db.update_physical_readiness(
            job.file_id,
            thumbnail_ready=thumb_ok,
            preview_ready=True,
            requeue_missing=False,
        )
        return True

    if art == Artifact.HASH:
        db.upsert_features(
            job.file_id,
            {
                "phash": feat.get("phash") or "v3phash",
                "dhash": feat.get("dhash") or "v3dhash",
                "whash": feat.get("whash") or "v3whash",
                "texture_features": feat.get("texture_features") or [],
                "dino_embedding": feat.get("dino_embedding"),
                "clip_embedding": feat.get("clip_embedding"),
                "patch_embeddings_meta": feat.get("patch_embeddings_meta") or [],
                "texture_map": feat.get("texture_map") or {},
                "dominant_colors": feat.get("dominant_colors") or [],
            },
        )
        return True

    if art == Artifact.METADATA:
        db.upsert_file(
            {
                "path": row["path"],
                "format_metadata": row.get("format_metadata") or "{}",
                "width": int(row.get("width") or 64),
                "height": int(row.get("height") or 64),
            }
        )
        return True

    # Heavy stubs — only touch requested artifact
    tm = dict(feat.get("texture_map") or {})
    if isinstance(tm, str):
        try:
            tm = json.loads(tm)
        except Exception:
            tm = {}

    payload = {
        "phash": feat.get("phash") or "v3phash",
        "dhash": feat.get("dhash") or "v3dhash",
        "whash": feat.get("whash") or "v3whash",
        "texture_features": feat.get("texture_features") or [0.1],
        "dino_embedding": feat.get("dino_embedding"),
        "clip_embedding": feat.get("clip_embedding"),
        "patch_embeddings_meta": feat.get("patch_embeddings_meta") or [],
        "texture_map": tm,
        "dominant_colors": feat.get("dominant_colors") or [],
    }
    if art == Artifact.DINO:
        # Test-only stub (engine without settings). Production injects
        # RealArtifactProcessor and never takes this path.
        emb = feat.get("dino_embedding")
        if not emb or len(emb) % 4 != 0 or len(emb) != 384 * 4:
            emb = b"\0" * (384 * 4)
        payload["dino_embedding"] = emb
    elif art == Artifact.CLIP:
        emb = feat.get("clip_embedding")
        if not emb or len(emb) % 4 != 0 or len(emb) != 512 * 4:
            emb = b"\0" * (512 * 4)
        payload["clip_embedding"] = emb
    elif art == Artifact.TEXTURE:
        payload["texture_features"] = [0.1, 0.2, 0.3]
        payload["phash"] = payload["phash"] or "v3phash"
    elif art == Artifact.SEMANTIC:
        tm["semantic_tags"] = {"motifs": ["v3"], "confidence": 0.9}
        payload["texture_map"] = tm
    elif art == Artifact.DNA:
        tm["pattern_dna"] = {"motif_family": "v3", "confidence": 0.9}
        payload["texture_map"] = tm
    elif art == Artifact.OBJECT_CONCEPT:
        tm["visual_concept_dna"] = {"version": 1, "backend": "test", "objects": [], "concepts": []}
        payload["texture_map"] = tm
    elif art == Artifact.OWLV2:
        from core.index_freeze import allow_index_writes
        from core.index_v3.artifact_state import object_index_path_for_patterns_db
        from core.object_index import ObjectIndexStore
        from core.ovd_index import OVD_DETECTOR, OVD_MODEL_VERSION, VOCAB_VERSION

        obj_path = object_index_path_for_patterns_db(db)
        if not obj_path:
            return False
        path = str(job.path or row.get("path") or "")
        mtime = 0.0
        size = 0
        if path and Path(path).is_file():
            st = Path(path).stat()
            mtime = float(st.st_mtime)
            size = int(st.st_size)
        store = ObjectIndexStore(obj_path)
        with allow_index_writes():
            store.upsert_open_vocab_objects(
                int(job.file_id),
                path,
                [],
                mtime=mtime,
                file_size=size,
                vocab_version=VOCAB_VERSION,
                model=f"{OVD_DETECTOR}/{OVD_MODEL_VERSION}",
                model_version=OVD_MODEL_VERSION,
                threshold=0.10,
                scan_status="done",
                last_path=path,
            )
        return True
    elif art == Artifact.PATCH:
        payload["patch_embeddings_meta"] = [{"index": 0, "tag": "v3"}]
    elif art == Artifact.OCR:
        with db.connect() as conn:
            conn.execute(
                "UPDATE files SET ocr_text=?, ocr_processed=1, updated_at=datetime('now') "
                "WHERE id=?",
                ("", int(job.file_id)),
            )
        return True
    else:
        return False

    db.upsert_features(job.file_id, payload)
    return True


def _same_file_followup_artifacts(
    current: Job,
    lane_artifacts: tuple[Artifact, ...] | list[Artifact] | None,
) -> tuple[Artifact, ...] | None:
    """Aynı dosyada GA hattı PATCH/OCR'a atlamaz; POST_GA kendi içinde zincirlenir.

    HASH→…→DNA aynı dosyada kalır. DNA sonrası claim_fair'e dönülür; böylece
    başka dosyanın HASH'i, bu dosyanın otomatik PATCH/OCR'ından önce gelir.
    Manuel PATCH claim edildikten sonra aynı dosyanın OCR'ı zincirlenebilir.
    """
    if current.artifact in POST_GA_ARTIFACTS:
        allowed: tuple[Artifact, ...] = POST_GA_ARTIFACTS
    elif current.artifact in HEAVY_ARTIFACTS:
        allowed = HEAVY_ARTIFACTS
    else:
        return None
    if lane_artifacts:
        lane = set(lane_artifacts)
        allowed = tuple(a for a in allowed if a in lane)
        if not allowed:
            return None
    return allowed


class Worker:
    def __init__(
        self,
        db: Any,
        store: JobStore,
        *,
        processor: ArtifactProcessor | None = None,
        worker_id: str = "v3-worker",
        max_attempts: int = 3,
        claim_timeout_sec: float = DEFAULT_CLAIM_TIMEOUT_SEC,
        ocr_enabled: bool = False,
        patch_enabled: bool = True,
        session_mode: Mode | str | None = None,
    ) -> None:
        self.db = db
        self.store = store
        self.processor = processor or ArtifactProcessor()
        self.worker_id = worker_id
        self.max_attempts = max_attempts
        self.claim_timeout_sec = max(30.0, float(claim_timeout_sec))
        self._ocr_enabled = bool(ocr_enabled)
        self._patch_enabled = bool(patch_enabled)
        # Default COMPLETE: preview→heavy prefetch (file-level parallel).
        # FAST must NOT seed HEAVY; GENERAL_AI does not produce Preview.
        if session_mode is None:
            self.session_mode = Mode.COMPLETE
        elif isinstance(session_mode, Mode):
            self.session_mode = session_mode
        else:
            self.session_mode = Mode(str(session_mode))
        self.stats = WorkerStats()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._progress_cb: Callable[..., None] | None = None

    def set_progress_callback(self, cb: Callable[..., None] | None) -> None:
        self._progress_cb = cb

    def request_stop(self) -> None:
        self._stop.set()
        self._pause.set()

    def request_pause(self) -> None:
        self._pause.clear()

    def request_resume(self) -> None:
        self._pause.set()

    def _emit_progress(self, job: Job | None = None, *, phase: str = "") -> None:
        if self._progress_cb is None:
            return
        try:
            info: dict[str, Any] = {
                "worker": self.worker_id,
                "phase": phase,
            }
            if job is not None:
                from pathlib import Path as _P

                info.update(
                    {
                        "file_id": int(job.file_id),
                        "artifact": job.artifact.value,
                        "queue": job.queue.value,
                        "source_id": int(job.source_id or 0),
                        "path": str(job.path or ""),
                        "filename": _P(str(job.path or "")).name or str(job.file_id),
                        "stage": job.artifact.value,
                    }
                )
            self._progress_cb(info)
        except TypeError:
            # eski imza: cb()
            try:
                self._progress_cb()  # type: ignore[call-arg]
            except Exception:
                pass
        except Exception:
            pass

    def _wait(self) -> bool:
        while not self._pause.is_set():
            if self._stop.is_set():
                return False
            self._pause.wait(0.2)
        return not self._stop.is_set()

    def run_queue(
        self,
        queue: QueueKind,
        source_ids: list[int],
        *,
        max_jobs: int | None = None,
        artifacts: tuple[Artifact, ...] | list[Artifact] | None = None,
    ) -> WorkerStats:
        self.stats = WorkerStats()
        done = 0
        consecutive_defer = 0
        idle_since: float | None = None
        try:
            while not self._stop.is_set():
                if not self._wait():
                    break
                if max_jobs is not None and done >= max_jobs:
                    break
                try:
                    unstick_owlv2_dep_wait_if_preview_ready(
                        self.store,
                        self.db,
                        queue,
                        source_ids,
                        artifacts,
                    )
                    batch = claim_fair(
                        self.store,
                        queue,
                        self.worker_id,
                        source_ids,
                        limit=1,
                        artifacts=artifacts,
                    )
                except Exception:
                    logger.exception(
                        "V3 worker claim failed worker=%s queue=%s",
                        self.worker_id,
                        queue.value,
                    )
                    # Geçici SQLite/NAS tarama yarışında worker ölmesin.
                    # Grace penceresi içinde tekrar dene; kalıcı hata ise
                    # engine tarafından görülebilsin diye normal çıkış yap.
                    if idle_since is None:
                        idle_since = time.monotonic()
                    if (time.monotonic() - idle_since) < IDLE_GRACE_SEC:
                        self._stop.wait(IDLE_POLL_SEC)
                        continue
                    break
                if not batch:
                    # Kuyruk tarama sırasında geçici olarak boşalabilir.
                    # Worker kapanırsa daha sonra eklenen işler sahipsiz kalıyordu.
                    # Kısa bir grace window boyunca kuyruğu yeniden yokla.
                    if idle_since is None:
                        idle_since = time.monotonic()
                    if (time.monotonic() - idle_since) < IDLE_GRACE_SEC:
                        self._stop.wait(IDLE_POLL_SEC)
                        continue
                    break
                idle_since = None
                for job in batch:
                    current: Job | None = job
                    while current is not None:
                        t_wait = time.perf_counter()
                        if not self._wait():
                            self._fail_claimed_job(current, "stopped", max_attempts=10_000)
                            current = None
                            break
                        worker_wait_ms = (time.perf_counter() - t_wait) * 1000.0
                        try:
                            processed_before = self.stats.processed
                            self._emit_progress(current, phase="claimed")
                            hb_stop = threading.Event()
                            running = current

                            def _heartbeat() -> None:
                                while not hb_stop.wait(CLAIM_HEARTBEAT_INTERVAL_SEC):
                                    if self._stop.is_set():
                                        return
                                    try:
                                        self.store.heartbeat(
                                            running.file_id,
                                            running.artifact,
                                            self.worker_id,
                                        )
                                    except Exception:
                                        logger.exception(
                                            "V3 worker heartbeat failed worker=%s file_id=%s artifact=%s",
                                            self.worker_id,
                                            running.file_id,
                                            running.artifact.value,
                                        )
                                    self._emit_progress(running, phase="claimed")

                            hb = threading.Thread(
                                target=_heartbeat,
                                name=f"{self.worker_id}-heartbeat",
                                daemon=True,
                            )
                            hb.start()
                            try:
                                deferred = self._run_one(
                                    current, worker_wait_ms=worker_wait_ms
                                )
                            finally:
                                hb_stop.set()
                                hb.join(timeout=0.2)
                        except Exception as exc:
                            deferred = False
                            self._record_task_failure(
                                current,
                                exc,
                                already_processed=self.stats.processed > processed_before,
                            )
                        if deferred:
                            consecutive_defer += 1
                            if consecutive_defer >= 1:
                                self._stop.wait(0.02)
                            current = None
                            break
                        consecutive_defer = 0
                        done += 1
                        self._emit_progress(None, phase="idle")
                        if current.artifact == Artifact.PREVIEW:
                            nxt = self.store.claim(
                                QueueKind.LIGHT,
                                self.worker_id,
                                source_id=current.source_id,
                                limit=1,
                                artifacts=(Artifact.THUMBNAIL,),
                                file_id=current.file_id,
                            )
                            current = nxt[0] if nxt else None
                            if current is not None:
                                continue
                        if max_jobs is not None and done >= max_jobs:
                            current = None
                            break
                        if current is None or current.queue not in (
                            QueueKind.HEAVY,
                            QueueKind.REPAIR,
                        ):
                            current = None
                            break
                        follow = _same_file_followup_artifacts(current, artifacts)
                        if follow is None:
                            current = None
                            break
                        nxt = self.store.claim(
                            queue,
                            self.worker_id,
                            source_id=current.source_id,
                            limit=1,
                            artifacts=follow,
                            file_id=current.file_id,
                        )
                        current = nxt[0] if nxt else None
                    if consecutive_defer >= 1:
                        break
                if consecutive_defer >= 1:
                    # dep_wait claim-release aynı kuyrukta sonsuza gitmesin;
                    # engine sonraki kuyruğa (REPAIR) geçebilsin.
                    break
        finally:
            # Worker herhangi bir nedenle run_queue'dan çıkarsa kendi claim'lerini
            # DB'de kilitli bırakma. Normal stop sonrası da claim temizlenir.
            try:
                reclaimed = self.store.requeue_claims_for_worker(
                    self.worker_id, error="worker_loop_closed"
                )
                if reclaimed:
                    logger.warning(
                        "V3 worker released claims on loop exit worker=%s count=%s",
                        self.worker_id,
                        reclaimed,
                    )
            except Exception:
                logger.exception(
                    "V3 worker could not release claims on loop exit worker=%s",
                    self.worker_id,
                )
        return self.stats

    def _fail_claimed_job(
        self, job: Job, error: str, *, max_attempts: int | None = None
    ) -> None:
        try:
            self.store.fail(
                job.file_id,
                job.artifact,
                error=error,
                permanent=False,
                max_attempts=max_attempts or self.max_attempts,
            )
        except Exception:
            logger.exception(
                "V3 worker could not finalize failed job file_id=%s artifact=%s",
                job.file_id,
                job.artifact.value,
            )

    def _record_task_failure(
        self, job: Job, exc: Exception, *, already_processed: bool = False
    ) -> None:
        error = str(exc) or exc.__class__.__name__
        ext = Path(str(job.path or "")).suffix.lower() or "?"
        if not already_processed:
            self.stats.processed += 1
        self.stats.failed += 1
        self.stats.errors_by_ext[ext] = int(
            self.stats.errors_by_ext.get(ext, 0) or 0
        ) + 1
        logger.exception(
            "V3 worker task failed before completion file_id=%s artifact=%s",
            job.file_id,
            job.artifact.value,
        )
        self._fail_claimed_job(job, error)
        _isolate_post_ga_failure(self.db, job, error)

    def _timing_begin(self, job: Job, worker_wait_ms: float) -> float | None:
        if job.artifact not in _TIMED_ARTIFACTS:
            return None
        started = time.time()
        try:
            claimed, enqueued = self.store.claim_clock(job.file_id, job.artifact)
            qwait = None
            if enqueued > 0 and claimed >= enqueued:
                qwait = (claimed - enqueued) * 1000.0
            return self.store.timings.stage_start(
                job.file_id,
                job.artifact.value,
                claimed_at=claimed,
                queue_wait_ms=qwait,
                worker_wait_ms=float(worker_wait_ms or 0),
            )
        except Exception:
            logger.exception(
                "V3 file timing start failed file_id=%s artifact=%s",
                job.file_id,
                job.artifact.value,
            )
            return started

    def _timing_end(self, job: Job, started_at: float | None) -> None:
        if started_at is None:
            return
        try:
            self.store.timings.stage_end(
                job.file_id,
                job.artifact.value,
                started_at=started_at,
                file_read_ms=float(getattr(self.stats, "file_read_ms", 0) or 0),
            )
        except Exception:
            logger.exception(
                "V3 file timing end failed file_id=%s artifact=%s",
                job.file_id,
                job.artifact.value,
            )
        finally:
            self.stats.file_read_ms = 0.0

    def _enqueue_downstream_after_preview(self, job: Job) -> None:
        """Preview READY → Thumbnail; HEAVY for COMPLETE/REPAIR/GENERAL_AI.

        FAST oturumunda Genel AI job yazılmaz (bağımsız Hızlı İndeks).
        COMPLETE/REPAIR/GENERAL_AI'de preview biter bitmez HEAVY kuyruğa düşer.
        """
        if job.artifact != Artifact.PREVIEW:
            return
        report = assess_file(self.db, job.file_id, require_disk=False)
        if not report.ready(Artifact.PREVIEW):
            return
        jobs: list[Job] = [
            j
            for j in plan_jobs_for_file(report, Mode.FAST)
            if j.artifact == Artifact.THUMBNAIL
        ]
        if self.session_mode in (Mode.COMPLETE, Mode.REPAIR, Mode.GENERAL_AI):
            jobs.extend(
                plan_jobs_for_file(
                    report,
                    Mode.GENERAL_AI,
                    ocr_enabled=self._ocr_enabled,
                    patch_enabled=self._patch_enabled,
                )
            )
        if jobs:
            self.store.enqueue(jobs)

    def _enqueue_thumbnail_after_preview(self, job: Job) -> None:
        """Geriye uyum; downstream planlama tek giriş noktası."""
        self._enqueue_downstream_after_preview(job)

    def _run_one(self, job: Job, worker_wait_ms: float = 0.0) -> bool:
        self.stats.processed += 1
        self.stats.file_read_ms = 0.0
        if job.artifact in POST_GA_ARTIFACTS or job.artifact in (
            Artifact.OBJECT_CONCEPT,
            Artifact.OWLV2,
        ):
            from core.index_freeze import INDEX_FROZEN, process_search_active

            if INDEX_FROZEN and process_search_active():
                self.store.release_dep_wait(
                    job.file_id, job.artifact, reason="search_session"
                )
                self.stats.processed -= 1
                return True
        # Skip if already READY (idempotent; thumb/preview = path+physical_*)
        report = assess_file(self.db, job.file_id, require_disk=False)
        if report.ready(job.artifact):
            self.store.complete(job.file_id, job.artifact)
            self.stats.skipped += 1
            self.stats.completed += 1
            self._enqueue_downstream_after_preview(job)
            return False
        # Path var ama physical_*_ready=0 → diskte varsa bayrağı yükselt (eski kayıt)
        if job.artifact in (Artifact.THUMBNAIL, Artifact.PREVIEW):
            if _promote_physical_ready(self.db, job.file_id, job.artifact):
                self.store.complete(job.file_id, job.artifact)
                self.stats.skipped += 1
                self.stats.completed += 1
                self._enqueue_downstream_after_preview(job)
                return False
        # Preview is the single image source gate for Thumbnail + all General AI.
        # General AI never falls back to the original source.
        if (
            job.artifact == Artifact.THUMBNAIL
            or (job.queue in (QueueKind.HEAVY, QueueKind.REPAIR) and job.artifact in (
                Artifact.HASH,
                Artifact.METADATA,
                Artifact.DINO,
                Artifact.CLIP,
                Artifact.TEXTURE,
                Artifact.SEMANTIC,
                Artifact.DNA,
                Artifact.OBJECT_CONCEPT,
                Artifact.OWLV2,
                Artifact.PATCH,
                Artifact.OCR,
            ))
        ):
            if not report.preview_ready:
                # Permanent Preview failure → drop this claim as stale (no spin).
                if (
                    self.store.job_state(job.file_id, Artifact.PREVIEW)
                    == "failed_permanent"
                ):
                    reconcile_stale_pendings(self.store, report)
                    # Current claim is not pending — drop it too.
                    self.store.cancel_pending_artifact(job.file_id, job.artifact)
                    self.stats.skipped += 1
                    return False
                # Preview gate yarış koşuludur; hata değildir.
                # Preview aynı oturumda hazır olabilir, bu yüzden attempt
                # tüketmeden işi pending bırak ve worker'ın sonraki turunda
                # tekrar claim edilmesini sağla.
                self.store.release_dep_wait(job.file_id, job.artifact)
                return True
            if self._defer_unready_deps(job, report):
                self.stats.processed -= 1
                return True
        started = None
        try:
            started = self._timing_begin(job, worker_wait_ms)
            if job.artifact == Artifact.OWLV2:
                from core.index_freeze import allow_index_writes

                logger.info("CLAIM file_id=%s artifact=owlv2", job.file_id)
                with allow_index_writes():
                    ok = self.processor.process(self.db, job, self.stats)
            else:
                ok = self.processor.process(self.db, job, self.stats)
            if not ok:
                raise RuntimeError(f"processor_failed:{job.artifact.value}")
            # VERIFY — thumb/preview disk zorunlu
            require_disk = job.artifact in (Artifact.THUMBNAIL, Artifact.PREVIEW)
            after = assess_file(
                self.db, job.file_id, require_disk=require_disk
            )
            if not after.ready(job.artifact):
                # process physical yazdıysa tekrar dene; yoksa disk promote
                if job.artifact in (Artifact.THUMBNAIL, Artifact.PREVIEW):
                    if _promote_physical_ready(self.db, job.file_id, job.artifact):
                        after = assess_file(
                            self.db, job.file_id, require_disk=require_disk
                        )
                if not after.ready(job.artifact):
                    raise RuntimeError(f"verify_failed:{job.artifact.value}")
            self.store.complete(job.file_id, job.artifact)
            self._timing_end(job, started)
            self.stats.completed += 1
            if job.artifact == Artifact.PATCH:
                try:
                    with self.db.connect() as conn:
                        conn.execute(
                            "UPDATE files SET patch_error='' WHERE id=?",
                            (int(job.file_id),),
                        )
                except Exception:
                    pass
            if job.artifact in (Artifact.THUMBNAIL, Artifact.PREVIEW) and after.light_complete:
                from core.index_ssot import complete_light
                complete_light(self.db, job.file_id)
            self._enqueue_downstream_after_preview(job)
        except Exception as exc:
            err = str(exc)
            wait_gate = (
                "preview_required" in err
                or "preview_required_for_thumbnail" in err
                or "search_session" in err
                or "frozen_or_search" in err
                or "INDEX_FROZEN_WRITE_BLOCKED" in err
            )
            if wait_gate or exc.__class__.__name__ == "IndexFrozenWriteBlocked":
                reason = "dep_wait"
                if (
                    "search_session" in err
                    or "frozen_or_search" in err
                    or "INDEX_FROZEN_WRITE_BLOCKED" in err
                    or exc.__class__.__name__ == "IndexFrozenWriteBlocked"
                ):
                    reason = "search_session"
                elif "preview_required" in err:
                    reason = "preview_required"
                self.store.release_dep_wait(
                    job.file_id, job.artifact, reason=reason
                )
                self.stats.processed -= 1
                return True
            self._timing_end(job, started)
            if "timeout" in err.lower() or isinstance(exc, TimeoutError):
                self.stats.timeouts += 1
                terminal_timeout = True
            else:
                terminal_timeout = False
            ext = Path(str(job.path or "")).suffix.lower() or "?"
            self.stats.errors_by_ext[ext] = int(
                self.stats.errors_by_ext.get(ext, 0) or 0
            ) + 1
            terminal = terminal_timeout or any(
                t in err
                for t in (
                    "unsupported_visual",
                    "renderer_missing",
                    "empty_file",
                    "corrupt_file",
                    "corrupt_eps_ai",
                    "decode_failed",
                    "ghostscript_render_error",
                    "ghostscript_timeout",
                    "ghostscript_permission",
                    "pymupdf_render_error",
                    "pymupdf_permission",
                    "v3_decode_memory",
                    "v3_unreadable",
                    "v3_decode_exception",
                    "not a known file format",
                    "VipsForeignLoad",
                    "unavailable/invalid",
                    "stub_processor",
                )
            )
            if "file_missing" in err and job.artifact in (
                Artifact.PREVIEW,
                Artifact.THUMBNAIL,
            ):
                terminal = True
            if job.artifact in POST_GA_ARTIFACTS:
                _isolate_post_ga_failure(self.db, job, err)
            self.store.fail(
                job.file_id,
                job.artifact,
                error=err,
                permanent=terminal,
                max_attempts=self.max_attempts,
            )
            self.stats.failed += 1
        return False

    def _defer_unready_deps(self, job: Job, report: Any) -> bool:
        deps = ARTIFACT_DEPENDENCIES.get(job.artifact) or ()
        for dep in deps:
            if report.ready(dep):
                continue
            st = self.store.job_state(job.file_id, dep)
            if st == "failed_permanent":
                self.store.fail(
                    job.file_id,
                    job.artifact,
                    error=f"dep_failed:{dep.value}",
                    permanent=True,
                    max_attempts=self.max_attempts,
                )
                self.stats.failed += 1
                return False
            self.store.release_dep_wait(job.file_id, job.artifact)
            return True
        return False


def _promote_physical_ready(
    db: Any, file_id: int, artifact: Artifact | None = None
) -> bool:
    """Path + disk varsa physical_*_ready=1. Artifact verilirse yalnız o lane için True."""
    from core.index_analysis_guards import local_artifact_exists

    row = db.get_file_by_id(int(file_id)) or {}
    thumb = str(row.get("thumbnail_path") or "").strip()
    prev = str(row.get("feature_preview_path") or "").strip()
    thumb_ok = bool(thumb) and local_artifact_exists(thumb)
    prev_ok = bool(prev) and local_artifact_exists(prev)
    if not thumb_ok and not prev_ok:
        return False
    db.update_physical_readiness(
        int(file_id),
        thumbnail_ready=thumb_ok,
        preview_ready=prev_ok,
        requeue_missing=False,
    )
    if artifact == Artifact.THUMBNAIL:
        return thumb_ok
    if artifact == Artifact.PREVIEW:
        return prev_ok
    return thumb_ok or prev_ok
