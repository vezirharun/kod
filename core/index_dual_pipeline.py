"""Hızlı index + Genel Index — tamamen bağımsız çift bant."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

from core.index_file_status import WORKER_HEAVY, WORKER_LIGHT
from core.index_modes import IndexMode
from core.logger import setup_logger

logger = setup_logger(__name__)

_LIGHT_CHUNK = 2000


class IndexDualPipeline:
    """
    Fast ve Genel Index bağımsız çalışır.

    - Light bitince Heavy'yi beklemez (join yalnızca light beslemesi bitince,
      kuyruktaki kalan heavy işleri boşaltmak için yapılır).
    - Seed heavy_jobs hemen kuyruğa girer; Light hızına eşik bağlı değildir.
    - Light başarı → anında offer; Heavy hazır işi alır.
    - AI yükleme Heavy tarafında (Light'ı bloklamaz).
    """

    def __init__(self, indexer: Any) -> None:
        self.indexer = indexer
        self.db = indexer.db
        self._heavy_queue: queue.Queue[tuple[int, str, str] | None] = queue.Queue()
        self._seen_heavy: set[int] = set()
        self._seen_lock = threading.Lock()
        self._ai_lock = threading.Lock()
        self._ai_loaded = False
        self._emit: Callable[..., None] | None = None
        self._stats: dict | None = None

    def run(
        self,
        *,
        source: dict[str, Any],
        light_jobs: list[tuple[int, str, str]],
        heavy_jobs: list[tuple[int, str, str]],
        stats: dict,
        deep: bool,
        emit: Callable[..., None],
        light_chunk_size: int = _LIGHT_CHUNK,
    ) -> None:
        if not light_jobs and not heavy_jobs:
            return

        heavy_workers = self.indexer._effective_worker_count(
            source, ai_enabled=True, light_pass=False
        )
        light_workers = self.indexer._effective_worker_count(
            source, light_pass=True
        )
        # CPU'da 4'e kadar; GPU yokken de havuzu daha hızlı tüket
        n_heavy = max(1, min(int(heavy_workers or 1), 4))
        n_light = max(1, min(int(light_workers or 1), 6))
        stats["dual_heavy_workers"] = n_heavy
        stats["dual_light_workers"] = n_light
        emit(
            "dual_workers",
            heavy_workers=n_heavy,
            light_workers=n_light,
            active_phase="Hızlı + Genel",
        )
        chunk = max(200, int(light_chunk_size or _LIGHT_CHUNK))

        stop = threading.Event()
        heavy_threads: list[threading.Thread] = []
        self._emit = emit
        self._stats = stats

        light_total = len(light_jobs)
        heavy_seed = len(heavy_jobs)
        with self.indexer._stats_lock:
            stats["light_jobs_total"] = int(stats.get("light_jobs_total", 0) or 0) + light_total
            stats["heavy_jobs_total"] = int(stats.get("heavy_jobs_total", 0) or 0) + heavy_seed
            stats["fast_lane"] = True
            stats["general_ai_lane"] = True
            # Genel jobs_total: her iki hat (UI geriye uyum)
            if int(stats.get("jobs_total", 0) or 0) <= 0:
                stats["jobs_total"] = light_total + heavy_seed
            jt = int(stats.get("jobs_total", 0) or 0)
            lt = int(stats.get("light_jobs_total", 0) or 0)
            ht = int(stats.get("heavy_jobs_total", 0) or 0)

        emit(
            "process_batch_start",
            jobs_total=jt,
            jobs_done=int(stats.get("jobs_done", 0) or 0),
            light_jobs_total=lt,
            heavy_jobs_total=ht,
            light_processed=int(stats.get("light_processed", 0) or 0),
            heavy_processed=int(stats.get("heavy_processed", 0) or 0),
            index_pass="dual",
            active_phase="Hızlı + Genel",
            dual_pipeline=True,
            independent_lanes=True,
        )

        # Light extractor hemen — AI Heavy içinde yüklenecek
        if light_jobs:
            self.indexer._ensure_extractor(use_ai=False, fast_hash_only=False)

        def _ensure_ai() -> None:
            if self._ai_loaded:
                return
            with self._ai_lock:
                if self._ai_loaded:
                    return
                emit("ai_loading", active_phase="Genel Index yükleniyor")
                self.indexer._ensure_extractor_for_index()
                self._ai_loaded = True

        def _bump_done(name: str, *, lane: str) -> None:
            with self.indexer._stats_lock:
                stats["jobs_done"] = int(stats.get("jobs_done", 0) or 0) + 1
                jd = int(stats.get("jobs_done", 0) or 0)
                jt = int(stats.get("jobs_total", 0) or 0)
                proc = int(stats.get("processed", 0) or 0)
                lp = int(stats.get("light_processed", 0) or 0)
                hp = int(stats.get("heavy_processed", 0) or 0)
                lt = int(stats.get("light_jobs_total", 0) or 0)
                ht = int(stats.get("heavy_jobs_total", 0) or 0)
            emit(
                "process_done",
                current_file=name,
                index_pass="light" if lane == "light" else "full",
                active_phase="Hızlı Index" if lane == "light" else "Genel Index",
                jobs_done=jd,
                jobs_total=jt,
                processed=proc,
                light_processed=lp,
                heavy_processed=hp,
                light_jobs_total=lt,
                heavy_jobs_total=ht,
                dual_pipeline=True,
                independent_lanes=True,
            )

        def _heavy_loop() -> None:
            while not stop.is_set() and not self.indexer.is_stopped():
                if not self.indexer._wait_if_paused():
                    stop.set()
                    break
                try:
                    job = self._heavy_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if job is None:
                    self._heavy_queue.task_done()
                    break
                if self.indexer.is_stopped() or not self.indexer._wait_if_paused():
                    self._heavy_queue.put(job)
                    self._heavy_queue.task_done()
                    stop.set()
                    break
                fid, path, name = job
                from core.index_ssot import claim_heavy, complete_heavy, release_claim

                # RAM kuyruk yalnızca tampon — otorite DB SSOT
                if not claim_heavy(self.db, fid):
                    self._heavy_queue.put(job)
                    self._heavy_queue.task_done()
                    time.sleep(0.05)
                    continue
                try:
                    _ensure_ai()
                    stats["active_phase"] = "Genel Index"
                    task = self.indexer._detect_pipeline_task(
                        self.db.get_file_by_id(fid) or {},
                        file_id=fid,
                        light=False,
                        source_path=path,
                    )
                    emit(
                        "process",
                        current_file=name,
                        index_pass="full",
                        active_phase="Genel Index",
                        pipeline_task=task,
                        worker=WORKER_HEAVY,
                        dual_pipeline=True,
                    )
                    if self.indexer._process_file(
                        fid, path, name, stats, deep, light_pass=False
                    ):
                        row = self.db.get_file_by_id(fid) or {}
                        if str(row.get("heavy_status") or "") != "done":
                            complete_heavy(self.db, fid)
                        self.indexer._bump_stat(stats, "heavy_processed")
                        self.indexer._bump_stat(stats, "processed")
                finally:
                    release_claim(self.db, fid, WORKER_HEAVY)
                    self._heavy_queue.task_done()
                    _bump_done(name, lane="heavy")

        for i in range(n_heavy):
            t = threading.Thread(
                target=_heavy_loop,
                name=f"vezir-heavy-{i}",
                daemon=True,
            )
            t.start()
            heavy_threads.append(t)

        # Seed: zaten light hazır işler — Light'ı beklemeden
        if heavy_jobs:
            emit(
                "heavy_seed_start",
                active_phase="Genel Index",
                current_file=f"GENERAL AI kuyruk: {len(heavy_jobs):,} dosya",
                heavy_jobs_total=len(heavy_jobs),
                dual_pipeline=True,
                independent_lanes=True,
            )
        for job in heavy_jobs:
            self._offer_heavy(job, count_toward_total=False)

        claim_attempts: dict[int, int] = {}
        claim_lock = threading.Lock()
        max_claims = 10

        def _light_one(job: tuple[int, str, str]) -> None:
            fid, path, name = job
            if self.indexer.is_stopped() or stop.is_set():
                return
            if not self.indexer._wait_if_paused():
                stop.set()
                return
            from core.index_ssot import (
                claim_light,
                ensure_light_terminal,
                release_claim,
            )

            row_before = self.db.get_file_by_id(fid) or {}
            preview_repair = (
                str(row_before.get("light_status") or "") == "done"
                and int(row_before.get("needs_medium_preview") or 0) == 1
            )
            with claim_lock:
                n_try = int(claim_attempts.get(fid, 0) or 0) + 1
                claim_attempts[fid] = n_try
            if n_try > max_claims:
                if preview_repair:
                    self.db.mark_heavy_waiting_preview(fid)
                    _bump_done(name, lane="light")
                    return
                logger.warning(
                    "claim_loop_guard file_id=%s attempts=%s → failed",
                    fid,
                    n_try,
                )
                self.db.mark_light_failed(
                    fid,
                    reason="other",
                    error_msg=f"claim_loop_exceeded:{n_try}",
                )
                _bump_done(name, lane="light")
                return

            if preview_repair:
                if not self.db.try_acquire_processing_lock(fid, WORKER_LIGHT):
                    return
            elif not claim_light(self.db, fid):
                return
            err_msg = ""
            ok = False
            preview_repaired = False
            try:
                task = self.indexer._detect_pipeline_task(
                    self.db.get_file_by_id(fid) or {},
                    file_id=fid,
                    light=True,
                    source_path=path,
                )
                emit(
                    "process",
                    current_file=name,
                    index_pass="light",
                    active_phase="Hızlı Index",
                    pipeline_task=task,
                    worker=WORKER_LIGHT,
                    dual_pipeline=True,
                )
                try:
                    ok = bool(
                        self.indexer._process_file(
                            fid, path, name, stats, deep, light_pass=True
                        )
                    )
                except Exception as exc:
                    # Bomb / executor shutdown vb. — pipeline ölmesin
                    err_msg = str(exc)
                    logger.warning("light job exception %s: %s", path, exc)
                    ok = False
                    from core.quarantine import classify_quarantine_reason

                    self.db.mark_light_failed(
                        fid,
                        reason=classify_quarantine_reason(err_msg) or "other",
                        error_msg=err_msg,
                    )
                row_after = self.db.get_file_by_id(fid) or {}
                if ok and str(row_after.get("light_status") or "") != "done":
                    self.indexer._ensure_light_persisted(fid)
                    row_after = self.db.get_file_by_id(fid) or {}
                newly_done = (
                    ok
                    and str(row_before.get("light_status") or "") != "done"
                    and str(row_after.get("light_status") or "") == "done"
                )
                if preview_repair and ok:
                    from core.index_analysis_guards import has_medium_preview

                    preview_repaired = has_medium_preview(
                        row_after,
                        source_path=path,
                        feature_preview_cache=self.indexer.feature_preview,
                    )
                if newly_done:
                    self.indexer._bump_stat(stats, "light_processed")
                    self.indexer._bump_stat(stats, "processed")
                    self._offer_heavy(
                        (fid, path, name),
                        count_toward_total=True,
                        from_light=True,
                    )
                elif preview_repaired:
                    self.indexer._bump_stat(stats, "preview_backfill")
                    self.indexer._bump_stat(stats, "processed")
                    self._offer_heavy(
                        (fid, path, name),
                        count_toward_total=True,
                        from_light=True,
                    )
            finally:
                # Orphan processing bırakma (kuyruk başı kilidi)
                row_fin = self.db.get_file_by_id(fid) or {}
                st = str(row_fin.get("light_status") or "")
                if preview_repair and not preview_repaired:
                    # Preview repair başarısız olsa da kalıcı Light Done bozulmaz.
                    from core.index_stages import NEEDS_MEDIUM_PREVIEW

                    from datetime import datetime, timezone

                    now = datetime.now(timezone.utc).isoformat()
                    with self.db.connect() as conn:
                        conn.execute(
                            """
                            UPDATE files SET
                                light_status='done',
                                heavy_status='pending',
                                status=CASE
                                    WHEN status='error' THEN 'indexed'
                                    ELSE status
                                END,
                                index_stage=?,
                                needs_medium_preview=1,
                                processing_lock='',
                                quarantine_reason='',
                                updated_at=?
                            WHERE id=?
                            """,
                            (NEEDS_MEDIUM_PREVIEW, now, int(fid)),
                        )
                elif st == "processing":
                    from datetime import datetime, timezone

                    now = datetime.now(timezone.utc).isoformat()
                    stopping = self.indexer.is_stopped() or stop.is_set()
                    if ok:
                        self.indexer._ensure_light_persisted(fid)
                    elif stopping:
                        # Durduruldu → kaldığı yerden devam; pending↔processing spin yok
                        with self.db.connect() as conn:
                            conn.execute(
                                """
                                UPDATE files SET light_status='pending',
                                    processing_lock='', updated_at=?
                                WHERE id=? AND light_status='processing'
                                """,
                                (now, int(fid)),
                            )
                    else:
                        # Sessiz False / boş hata → pending'e atma (kalan 227↔228)
                        ensure_light_terminal(
                            self.db,
                            fid,
                            error_msg=err_msg or "light worker incomplete",
                        )
                release_claim(self.db, fid, WORKER_LIGHT)
                _bump_done(name, lane="light")

        def _run_light_batch(batch: list[tuple[int, str, str]]) -> None:
            if n_light <= 1 or len(batch) <= 1:
                for job in batch:
                    if self.indexer.is_stopped() or stop.is_set():
                        break
                    _light_one(job)
                return
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Executor yalnızca bu batch için; dosya hatası kapatmaz
            with ThreadPoolExecutor(
                max_workers=n_light, thread_name_prefix="vezir-light"
            ) as pool:
                futures = [pool.submit(_light_one, job) for job in batch]
                for fut in as_completed(futures):
                    if self.indexer.is_stopped():
                        stop.set()
                        break
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.warning("dual light worker error: %s", exc)

        # Light: DB boşalana kadar besle (tek snapshot ile ölme).
        # Night-only (light_jobs boş, heavy dolu) → light drain YOK.
        from core.index_ssot import (
            count_lanes,
            fail_orphan_light_processing,
            list_fast_jobs,
            reset_stale_processing,
        )

        source_id = int(source.get("id") or 0)
        from core.index_ssot import heal_pending_error_conflict

        reset_stale_processing(self.db)
        fail_orphan_light_processing(self.db)
        conflict_heal = heal_pending_error_conflict(self.db)
        if conflict_heal.get("conflict_total"):
            logger.warning("pending+error heal: %s", conflict_heal)
            stats["pending_error_heal"] = conflict_heal

        drain_light = bool(light_jobs) or not heavy_jobs
        seed = list(light_jobs) if drain_light else []
        stats["active_phase"] = "Hızlı Index"
        last_done = int(count_lanes(self.db).get("light_done", 0) or 0)
        last_failed = int(count_lanes(self.db).get("light_failed", 0) or 0)
        last_progress_t = time.time()
        stall_sec = 30.0
        empty_rounds = 0
        processed_offset = 0
        watchdog_heals = 0

        while drain_light and not self.indexer.is_stopped() and not stop.is_set():
            if not self.indexer._wait_if_paused():
                stop.set()
                break

            if seed:
                batch = seed[:chunk]
                seed = seed[chunk:]
            else:
                batch = list_fast_jobs(self.db, source_id, limit=chunk)
                if not batch:
                    empty_rounds += 1
                    if empty_rounds >= 2:
                        break
                    time.sleep(0.2)
                    continue
            empty_rounds = 0
            processed_offset += len(batch)

            emit(
                "queue_prepare",
                active_phase="Hızlı Index",
                current_file=(
                    f"FAST batch {processed_offset:,} · workers={n_light}"
                ),
                light_jobs_total=int(stats.get("light_jobs_total", 0) or 0),
                light_processed=int(stats.get("light_processed", 0) or 0),
                heavy_jobs_total=int(stats.get("heavy_jobs_total", 0) or 0),
                heavy_processed=int(stats.get("heavy_processed", 0) or 0),
                dual_pipeline=True,
                independent_lanes=True,
            )
            _run_light_batch(batch)

            lanes = count_lanes(self.db)
            done_now = int(lanes.get("light_done", 0) or 0)
            failed_now = int(lanes.get("light_failed", 0) or 0)
            pending_now = int(lanes.get("light_pending", 0) or 0)
            processing_now = int(lanes.get("light_processing", 0) or 0)
            progressed = done_now > last_done or failed_now > last_failed
            if progressed:
                last_done = done_now
                last_failed = failed_now
                last_progress_t = time.time()
            elif (
                pending_now > 0
                and processing_now == 0
                and (time.time() - last_progress_t) >= stall_sec
            ):
                watchdog_heals += 1
                # Aynı spin'i üretme: önce conflict heal, sonra sınırlı stale reset
                logger.warning(
                    "Light stall watchdog #%s: pending=%s — conflict heal",
                    watchdog_heals,
                    pending_now,
                )
                heal_pending_error_conflict(self.db)
                fail_orphan_light_processing(self.db)
                reset_stale_processing(self.db)
                # Claim loop'ta takılanları failed yap
                with claim_lock:
                    hot = [i for i, n in claim_attempts.items() if n >= max_claims]
                for hid in hot:
                    self.db.mark_light_failed(
                        hid,
                        reason="other",
                        error_msg=f"claim_loop_exceeded:{claim_attempts.get(hid)}",
                    )
                try:
                    from core.tif_thumbnail import _reset_tif_pool

                    _reset_tif_pool(reason="light stall watchdog")
                except Exception:
                    pass
                emit(
                    "light_watchdog_heal",
                    active_phase="Hızlı Index",
                    current_file="Light worker yeniden canlandırıldı",
                    light_pending=pending_now,
                )
                last_progress_t = time.time()
                seed = []
                if watchdog_heals >= 3 and pending_now > 0:
                    # Aynı boş tur — çık, sonsuz heal yok
                    logger.error(
                        "Light watchdog abort: %s heal sonrası hâlâ pending=%s",
                        watchdog_heals,
                        pending_now,
                    )
                    break
            elif pending_now == 0 and processing_now == 0 and not seed:
                break

        # Light beslemesi bitti — Fast hattı tamam; Heavy kalan kuyruğu bitirir.
        # Night-only (drain_light=False): işler zaten seed edildi; sentinel sonda.
        emit(
            "fast_lane_complete",
            active_phase="Genel Index",
            current_file=(
                "GENERAL AI çalışıyor…"
                if not drain_light
                else "FAST bitti · GENERAL AI devam ediyor"
            ),
            light_processed=int(stats.get("light_processed", 0) or 0),
            light_jobs_total=int(stats.get("light_jobs_total", 0) or 0),
            heavy_processed=int(stats.get("heavy_processed", 0) or 0),
            heavy_jobs_total=int(stats.get("heavy_jobs_total", 0) or 0),
            dual_pipeline=True,
            independent_lanes=True,
        )
        for _ in heavy_threads:
            self._heavy_queue.put(None)
        for t in heavy_threads:
            t.join(timeout=3600)

    def _offer_heavy(
        self,
        job: tuple[int, str, str],
        *,
        count_toward_total: bool = True,
        from_light: bool = False,
    ) -> None:
        """RAM heavy tamponuna ekle (otorite değil — sonraki tur yine DB SSOT).

        from_light=True: light az önce bitti — disk/DB gecikmesine takılma.
        Seed: thumb veya preview yolu varsa al; sıkı deep-preview kapısı yok
        (aksi halde Hızlı+Genel'de heavy hiç dolmuyordu).
        """
        fid = job[0]
        with self._seen_lock:
            if fid in self._seen_heavy:
                return
            # SSOT seed (count_toward_total=False): list_general_ai_jobs zaten
            # filtreledi — 40k× get_file_by_id Genel AI'yi dakikalarca "sahte çalışıyor"
            # bırakıyordu. from_light / canlı offer'da doğrula.
            if from_light or count_toward_total:
                existing = self.db.get_file_by_id(fid) or {}
                if str(existing.get("heavy_status") or "") == "done":
                    return
                if not from_light:
                    has_thumb = bool(
                        str(existing.get("thumbnail_path") or "").strip()
                    )
                    has_fp = bool(
                        str(existing.get("feature_preview_path") or "").strip()
                    )
                    if not has_thumb and not has_fp:
                        return
            self._seen_heavy.add(fid)
            if count_toward_total:
                try:
                    emit = self._emit
                    stats = self._stats
                    if emit is not None and stats is not None:
                        with self.indexer._stats_lock:
                            stats["heavy_jobs_total"] = (
                                int(stats.get("heavy_jobs_total", 0) or 0) + 1
                            )
                            stats["jobs_total"] = int(stats.get("jobs_total", 0) or 0) + 1
                            qsize = self._heavy_queue.qsize() + 1
                            payload = {
                                "jobs_total": int(stats.get("jobs_total", 0) or 0),
                                "jobs_done": int(stats.get("jobs_done", 0) or 0),
                                "light_processed": int(
                                    stats.get("light_processed", 0) or 0
                                ),
                                "heavy_processed": int(
                                    stats.get("heavy_processed", 0) or 0
                                ),
                                "light_jobs_total": int(
                                    stats.get("light_jobs_total", 0) or 0
                                ),
                                "heavy_jobs_total": int(
                                    stats.get("heavy_jobs_total", 0) or 0
                                ),
                                "heavy_queue_size": qsize,
                                "dual_pipeline": True,
                                "independent_lanes": True,
                            }
                        emit("jobs_total_update", **payload)
                except Exception:
                    pass
        self._heavy_queue.put(job)


def is_dual_pipeline_mode(index_mode: str) -> bool:
    return index_mode in (
        IndexMode.COMPLETE.value,
        IndexMode.NIGHT_COMPLETE.value,
        IndexMode.BACKFILL.value,
    )
