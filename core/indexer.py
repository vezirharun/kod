"""Index oluşturma ve incremental güncelleme — çok kaynaklı."""

from __future__ import annotations

import os
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.db import Database
from core.faiss_store import FaissStore
from core.feature_extractor import FeatureExtractor
from core.index_modes import IndexMode, apply_index_mode, restore_index_settings
from core.index_dual_pipeline import IndexDualPipeline
from core.light_asset_extractor import (
    LightAssetResult,
    extract_light_assets_once,
    try_reuse_light_assets,
)
from core.pipeline_step_stats import (
    bump_pipeline_step,
    format_pipeline_step_summary,
    init_pipeline_steps,
)
from core.metadata_sanitize import metadata_to_json
from core.index_analysis_guards import (
    has_medium_preview,
    has_preview_artifacts,
    has_preview_artifacts_for_deep,
    has_thumbnail_artifact,
    is_quarantined_file,
    local_artifact_exists,
    needs_medium_preview,
    resolve_medium_preview_path,
    texture_map_has_pattern_dna,
    texture_map_has_semantic_tags,
)
from core.network_index_throttle import (
    bind_deep_from_cache,
    deep_from_cache_scope,
    is_network_path,
    network_original_open_slot,
    network_preview_open_slot,
    network_read_slot,
    track_db_metadata_read,
    track_local_cache_read,
    unbind_deep_from_cache,
)
from core.quarantine import classify_quarantine_reason, is_hard_quarantine
from core.index_queue_manager import IndexQueueManager
from core.index_speed_report import IndexSpeedSession, write_index_speed_report
from core.index_stages import (
    FAILED,
    FULL_DONE,
    LIGHT_DONE,
    NEEDS_MEDIUM_PREVIEW,
    NEEDS_REVIEW,
    PENDING_FULL,
    PENDING_LIGHT,
    SKIPPED_HEAVY_FORMAT,
)
from core.logger import (
    log_error_file,
    setup_logger,
    write_index_report,
    write_source_scan_report,
)
from core.ocr_engine import OCREngine
from core.preview_cache import FEATURE_PREVIEW_VERSION, FeaturePreviewCache
from core.preview_renderer import render_preview_for_index
from core.progress_emit import ThrottledProgress
from core.scanner import ArchiveScanner, is_ignored_artifact_path
from core.settings import OPTIONAL_PREVIEW_EXTENSIONS, AppSettings
from core.sources import ScanMode, SourceManager
from core.texture_profile import (
    FEATURE_VERSION,
    TEXTURE_MAP_VERSION,
    THUMBNAIL_VERSION,
    TextureAnalyzer,
    TextureProfile,
)
from core.thumbnailer import Thumbnailer, ThumbnailResult
from core.utils import (
    full_file_hash,
    normalize_path,
    normalize_source_root,
    partial_file_hash,
)
from plugins.base import PluginRuntime
from plugins.registry import PluginRegistry

logger = setup_logger(__name__)


class Indexer:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.db = Database(settings.db_path)
        excluded = self.db.exclude_os_metadata_artifacts()
        if excluded:
            logger.info("OS metadata index kayitlari atlandi: %d", excluded)
        self.source_manager = SourceManager(settings)
        self.thumbnailer = Thumbnailer(
            settings.cache_dir,
            settings.thumbnail_max_edge,
            settings.thumbnail_format,
        )
        self.plugin_registry = PluginRegistry(
            PluginRuntime(
                cache_dir=settings.cache_dir,
                thumbnail_max_edge=settings.thumbnail_max_edge,
                thumbnail_format=settings.thumbnail_format,
                thumbnailer=self.thumbnailer,
            )
        )
        self.feature_preview = FeaturePreviewCache(
            settings.cache_dir,
            settings.feature_preview_max_edge,
            settings.thumbnail_format,
        )
        self.extractor = FeatureExtractor(
            use_ai=False,
            use_gpu=settings.use_gpu,
            fast_hash_only=settings.fast_hash_only,
        )
        self.ocr = OCREngine(enabled=settings.ocr_enabled)
        self.faiss = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
        # FAISS rebuild __init__'te YASAK — IndexWorker "Çalışıyor" gösterir ama
        # 58k embedding yükü dakikalar sürer; heavy hiç başlamaz gibi görünür.
        self._faiss_ready_checked = False
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._pause.set()
        self._write_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.queue_mgr = IndexQueueManager(self.db)
        self._search_priority = False
        self._ai_fallback_warning = ""
        self._speed_session: IndexSpeedSession | None = None
        self._faiss_dirty = 0

    def _format_descriptor_record(self, path: str) -> dict[str, Any]:
        descriptor = self.plugin_registry.describe(path)
        if not descriptor:
            return {}
        return {
            **descriptor,
            "format_capabilities": json.dumps(
                descriptor.get("format_capabilities", []), ensure_ascii=False
            ),
        }

    def _refresh_project_candidate(self, file_id: int) -> None:
        try:
            from core.project_search import ProjectSearchManager

            ProjectSearchManager(self.db).refresh_for_file(file_id)
        except Exception as exc:
            logger.debug("Project Search aday güncellemesi atlandı: %s", exc)

    def _maybe_save_faiss(self, *, force: bool = False) -> None:
        if not self.settings.ai_embedding_enabled:
            return
        self._faiss_dirty += 1
        if force or self._faiss_dirty >= 50:
            self.faiss.save()
            self._faiss_dirty = 0

    def _maybe_rebuild_faiss(self) -> None:
        """FAISS boşsa DB'den yeniden oluştur (lazy — __init__ dışında)."""
        if self._faiss_ready_checked:
            return
        self._faiss_ready_checked = True
        if not self.faiss.available or not self.settings.ai_embedding_enabled:
            return
        db_counts = self.db.count_embeddings()
        consistency = self.faiss.consistency(db_counts)
        if consistency["consistent"]:
            return
        # Tam rebuild burada YAPILMAZ — Genel AI başlangıcını 5–15 dk kilitler.
        # Index bitişinde faiss.rebuild_from_db / save zaten çalışır.
        logger.warning(
            "FAISS tutarsız (db=%s); index başında rebuild atlandı — bitişte yenilenecek",
            db_counts,
        )

    def _ensure_faiss_ready(
        self, progress_callback: Callable[[dict], None] | None = None
    ) -> None:
        """Index başlamadan önce FAISS tutarlılığı — UI'ye aşama bildir."""
        if self._faiss_ready_checked:
            return
        if progress_callback:
            try:
                progress_callback(
                    {
                        "stage": "faiss_prepare",
                        "active_phase": "Hazırlık",
                        "current_file": "FAISS / vektör indeksi kontrol ediliyor…",
                        "pipeline_task": "FAISS hazırlanıyor",
                    }
                )
            except Exception:
                pass
        self._maybe_rebuild_faiss()

    def _needs_feature_preview_update(self, existing: dict[str, Any] | None) -> bool:
        if not existing:
            return True
        fpv = int(existing.get("feature_preview_version") or 0)
        if fpv < FEATURE_PREVIEW_VERSION:
            return True
        fp = existing.get("feature_preview_path", "")
        if not fp or not local_artifact_exists(fp):
            return True
        return float(existing.get("feature_preview_mtime") or 0) < float(
            existing.get("mtime") or 0
        )

    def _resolve_feature_analysis_source(
        self,
        *,
        path: str,
        render_path: str,
        thumb_path: str,
        existing: dict[str, Any],
        light: bool,
        unsupported: int,
        stats: dict,
        allow_original_open: bool = True,
        deep_from_cache: bool = False,
    ) -> dict[str, Any]:
        """Analiz görüntüsü: önce feature preview cache, sonra thumbnail."""
        defer_preview = light and getattr(self.settings, "index_defer_preview", True)
        if deep_from_cache or (not light and not allow_original_open):
            defer_preview = True
        fp_path = ""
        fp_version = 0
        fp_mtime = float(existing.get("feature_preview_mtime") or existing.get("mtime") or 0)
        preview_generated = False
        original_reopened = False

        existing_fp = str(existing.get("feature_preview_path") or "").strip()
        if (
            existing_fp
            and local_artifact_exists(existing_fp, stats=stats)
            and not self._needs_feature_preview_update(existing)
        ):
            fp_path = existing_fp
            fp_version = int(existing.get("feature_preview_version") or FEATURE_PREVIEW_VERSION)

        if not fp_path and not unsupported:
            disk = self.feature_preview.get_existing(path)
            if disk.success:
                src_mtime = float(existing.get("mtime") or 0)
                try:
                    disk_mtime = os.path.getmtime(disk.preview_path)
                    track_local_cache_read(stats)
                except OSError:
                    disk_mtime = 0.0
                if src_mtime <= 0 or disk_mtime >= src_mtime - 1.0:
                    fp_path = disk.preview_path
                    fp_version = FEATURE_PREVIEW_VERSION

        if (
            not fp_path
            and not unsupported
            and self._needs_feature_preview_update(existing)
            and not defer_preview
            and allow_original_open
            and not deep_from_cache
        ):
            with network_preview_open_slot(
                self.settings,
                render_path or path,
                stats,
                reason="feature_preview_create",
                deep_from_cache=deep_from_cache,
            ):
                fp = self.feature_preview.create(path, render_path=render_path)
            original_reopened = True
            if fp.success:
                fp_path = fp.preview_path
                fp_version = FEATURE_PREVIEW_VERSION
                preview_generated = True
                fp_mtime = float(existing.get("mtime") or 0)

        feature_source = fp_path or thumb_path
        if fp_path:
            analysis_source = "feature_preview_cache"
        elif feature_source == thumb_path:
            analysis_source = "thumbnail_cache"
            if deep_from_cache:
                track_local_cache_read(stats)
            else:
                stats["cache_miss"] = int(stats.get("cache_miss", 0)) + 1
        else:
            analysis_source = "original"

        ai_analysis_from_cache = bool(fp_path) and not original_reopened

        with self._stats_lock:
            if preview_generated:
                stats["preview_generated"] = int(stats.get("preview_generated", 0)) + 1
            if ai_analysis_from_cache:
                stats["ai_from_cache"] = int(stats.get("ai_from_cache", 0)) + 1
                stats["cache_hit"] = int(stats.get("cache_hit", 0)) + 1
                stats["local_cache_read_count"] = int(
                    stats.get("local_cache_read_count", 0)
                ) + 1
            elif feature_source == thumb_path and not deep_from_cache:
                stats["cache_miss"] = int(stats.get("cache_miss", 0)) + 1
            if original_reopened:
                stats["analysis_preview_from_original"] = int(
                    stats.get("analysis_preview_from_original", 0)
                ) + 1

        if self._speed_session:
            if preview_generated:
                self._speed_session.preview_generated += 1
            if ai_analysis_from_cache:
                self._speed_session.ai_from_cache += 1
            if original_reopened:
                self._speed_session.original_reopened += 1

        return {
            "fp_path": fp_path,
            "fp_version": fp_version,
            "fp_mtime": fp_mtime,
            "feature_source": feature_source,
            "analysis_source": analysis_source,
            "ai_analysis_from_cache": ai_analysis_from_cache,
            "original_reopened": original_reopened,
            "preview_generated": preview_generated,
        }

    def _extract_ocr_text(
        self,
        *,
        fp_path: str,
        thumb_path: str,
        path: str,
        render_path: str,
        stats: dict,
        deep_from_cache: bool = False,
    ) -> tuple[str, bool]:
        """OCR: önce medium preview, deep pass'te orijinal açılmaz."""
        if not self.settings.ocr_enabled:
            return "", False
        try:
            from core.work_profile import apply_current_thread_priority

            apply_current_thread_priority("ocr")
        except Exception:
            pass
        self._sync_ocr_engine()
        for candidate in (fp_path, thumb_path):
            if candidate and local_artifact_exists(candidate, stats=stats):
                text = self.ocr.extract_text(candidate)
                if text.strip():
                    return text, False
        if deep_from_cache or fp_path:
            return "", False
        with network_preview_open_slot(
            self.settings,
            render_path or path,
            stats,
            reason="ocr_feature_preview_create",
            deep_from_cache=deep_from_cache,
        ):
            fp = self.feature_preview.create(path, render_path=render_path)
        if not fp.success:
            return "", False
        with self._stats_lock:
            stats["preview_generated"] = int(stats.get("preview_generated", 0)) + 1
            stats["analysis_preview_from_original"] = int(
                stats.get("analysis_preview_from_original", 0)
            ) + 1
            stats["cache_miss"] = int(stats.get("cache_miss", 0)) + 1
        if self._speed_session:
            self._speed_session.preview_generated += 1
            self._speed_session.original_reopened += 1
        return self.ocr.extract_text(fp.preview_path), True

    def request_stop(self) -> None:
        self._stop.set()
        # Pause bekleyen worker'ları uyandır — yoksa Durdur takılı kalır
        self._pause.set()
        try:
            from core.index_session import mark_interrupted

            mark_interrupted(self.db)
        except Exception:
            pass

    def request_pause(self) -> None:
        self._pause.clear()
        try:
            from core.index_session import touch_session

            touch_session(self.db, status="paused")
        except Exception:
            pass

    def request_resume(self) -> None:
        self._pause.set()
        try:
            from core.index_session import touch_session

            touch_session(self.db, status="running")
        except Exception:
            pass

    def is_stopped(self) -> bool:
        return self._stop.is_set()

    def is_paused(self) -> bool:
        return not self._pause.is_set() and not self._stop.is_set()

    def _wait_if_paused(self) -> bool:
        while not self._pause.is_set():
            if self._stop.is_set():
                return False
            self._pause.wait(timeout=0.5)
        return not self._stop.is_set()

    def _sync_ocr_engine(self) -> None:
        if self.ocr.enabled != self.settings.ocr_enabled:
            self.ocr = OCREngine(enabled=self.settings.ocr_enabled)

    def _index_uses_ai(self) -> bool:
        return self.settings.index_uses_ai()

    def _ensure_extractor_for_index(self) -> None:
        if self._index_uses_ai():
            # Zaten bellekteyse probe/reload yok — her heavy dosyada model yeniden yükleme.
            if (
                self.extractor.use_ai
                and not self.extractor.fast_hash_only
                and self.extractor.ai_available
            ):
                return
            from core.capability_check import AI_FALLBACK_MSG, try_load_ai_extractor

            ok, msg = try_load_ai_extractor(self.settings)
            if not ok:
                logger.warning(msg or AI_FALLBACK_MSG)
                self._ai_fallback_warning = msg or AI_FALLBACK_MSG
                self._ensure_extractor(use_ai=False, fast_hash_only=False)
                return
        self._ensure_extractor(
            use_ai=self._index_uses_ai(),
            fast_hash_only=False,
        )

    def _ensure_extractor(
        self, use_ai: bool | None = None, fast_hash_only: bool | None = None
    ) -> None:
        desired_ai = self.settings.ai_embedding_enabled if use_ai is None else use_ai
        desired_fast = (
            self.settings.fast_hash_only if fast_hash_only is None else fast_hash_only
        )
        if (
            self.extractor.use_ai == desired_ai
            and self.extractor.fast_hash_only == desired_fast
        ):
            return
        # Dual pipeline: Light AI'yi boşaltmasın. Modeller bellekte kalsın;
        # fast_hash_only=True ile embedding atlanır.
        if self.extractor.ai_available and self.extractor.use_ai:
            if desired_fast or not desired_ai:
                self.extractor.fast_hash_only = True
                return
            if desired_ai and not desired_fast:
                self.extractor.fast_hash_only = False
                self.extractor.use_ai = True
                return
        self.extractor = FeatureExtractor(
            use_ai=desired_ai,
            use_gpu=self.settings.use_gpu,
            fast_hash_only=desired_fast,
        )

    def _sources_for_index(self) -> list[dict[str, Any]]:
        """Index kapsamı — varsayılan: yalnızca seçili kaynaklar."""
        sources = self.db.list_sources(active_only=True)
        if not sources and self.settings.archive_root:
            sid = self.source_manager.add_source(
                "Ana Arşiv", self.settings.archive_root
            )
            row = self.db.get_source(sid)
            sources = [row] if row else [{"id": sid}]

        scope = (
            getattr(self.settings, "index_scope", "selected_sources")
            or "selected_sources"
        )
        if scope == "all":
            return sources

        if scope == "single_source" and int(self.settings.selected_source_id or 0) > 0:
            sid = int(self.settings.selected_source_id)
            return [s for s in sources if int(s["id"]) == sid]

        ids = {int(x) for x in (self.settings.selected_source_ids or []) if int(x) > 0}
        if scope == "selected_sources" and ids:
            picked = [s for s in sources if int(s["id"]) in ids]
            if picked:
                return picked
        return sources

    def _sources_ordered_by_backlog(
        self, sources: list[dict[str, Any]], index_mode: str
    ) -> list[dict[str, Any]]:
        """Gerçek Light pending > preview repair > hazır Heavy önceliği."""
        from core.index_ssot import (
            count_fast_jobs,
            count_general_ai_jobs,
            count_light_pending_jobs,
        )

        mode = str(index_mode or "")
        scored: list[tuple[int, int, int, int, dict[str, Any]]] = []
        for src in sources:
            sid = int(src.get("id") or 0)
            light_pending = 0
            repair = 0
            heavy_ready = 0
            try:
                if mode == IndexMode.FAST_ARCHIVE.value:
                    light_pending = count_light_pending_jobs(self.db, sid)
                    repair = max(
                        0, count_fast_jobs(self.db, sid) - light_pending
                    )
                elif mode == IndexMode.NIGHT_COMPLETE.value:
                    heavy_ready = count_general_ai_jobs(self.db, sid)
                else:
                    light_pending = count_light_pending_jobs(self.db, sid)
                    repair = max(
                        0, count_fast_jobs(self.db, sid) - light_pending
                    )
                    heavy_ready = count_general_ai_jobs(self.db, sid)
            except Exception:
                light_pending = repair = heavy_ready = 0
            scored.append(
                (
                    light_pending,
                    repair,
                    heavy_ready,
                    int(src.get("id") or 0),
                    src,
                )
            )
        scored.sort(
            key=lambda x: (
                0 if x[0] > 0 else 1,  # light pending kaynaklar önce
                -x[0],
                -x[2],  # heavy ready — WAITING_PREVIEW (repair) önünde
                -x[1],  # repair / WAITING_PREVIEW en sonda
                x[3],
            )
        )
        return [s for _, _, _, _, s in scored]

    def run(
        self,
        progress_callback: Callable[[dict], None] | None = None,
        customer_filter: str = "",
        scan_mode: str = ScanMode.QUICK.value,
        index_mode: str | IndexMode = IndexMode.STANDARD,
    ) -> dict:
        """Tüm aktif kaynakları incremental tarar."""
        mode = IndexMode(index_mode) if isinstance(index_mode, str) else index_mode
        snapshot = apply_index_mode(self.settings, mode)
        self._speed_session = IndexSpeedSession(mode=mode.value)
        self._speed_session.begin_light()
        self._stop.clear()
        self._pause.set()
        from core.index_freeze import INDEX_FROZEN, process_search_active

        while INDEX_FROZEN and process_search_active() and not self._stop.is_set():
            self._stop.wait(0.1)
        if self._stop.is_set() and process_search_active():
            restore_index_settings(self.settings, snapshot)
            return {
                "skipped": True,
                "reason": "search_session",
                "processed": 0,
                "errors": 0,
                "scanned": 0,
            }
        self.queue_mgr.reset_stale_running()
        try:
            from core.index_ssot import (
                heal_demoted_light_done,
                heal_pending_error_conflict,
                requeue_retryable_heavy_preview_failures,
                reset_stale_processing,
            )

            reset_stale_processing(self.db)
            restored = heal_demoted_light_done(self.db)
            if restored:
                logger.warning(
                    "SSOT heal: %s light pending→done (light_done_at dolu demote)",
                    restored,
                )
            requeued_preview = requeue_retryable_heavy_preview_failures(self.db)
            if requeued_preview:
                logger.warning(
                    "SSOT heal: %s retry edilebilir preview heavy_failed→pending",
                    requeued_preview,
                )
            heal_pending_error_conflict(self.db)
        except Exception:
            pass
        # Lazy FAISS — UI "Çalışıyor" iken dakikalarca sessiz kalmayı önle
        self._ensure_faiss_ready(progress_callback)
        self.source_manager.migrate_legacy_archive_root()
        sources = self._sources_ordered_by_backlog(
            self._sources_for_index(), mode.value
        )

        if not sources:
            restore_index_settings(self.settings, snapshot)
            raise ValueError("Aktif kaynak yok. Önce bir kaynak ekleyin.")

        from core.index_session import (
            begin_session,
            mark_completed,
            mark_interrupted,
            snapshot_progress_totals,
            touch_session,
        )

        try:
            begin_session(
                self.db,
                index_mode=mode.value,
                scan_mode=scan_mode,
                totals=snapshot_progress_totals(self.db),
            )
        except Exception:
            pass

        combined = {
            "scanned": 0,
            "processed": 0,
            "errors": 0,
            "skipped": 0,
            "skipped_fast": 0,
            "texture_updated": 0,
            "new": 0,
            "changed": 0,
            "sources_scanned": 0,
            "source_results": [],
            "index_mode": mode.value,
        }
        try:
            for source in sources:
                if self._stop.is_set():
                    break
                while INDEX_FROZEN and process_search_active() and not self._stop.is_set():
                    self._stop.wait(0.1)
                if self._stop.is_set():
                    break
                stats = self.run_source(
                    source=source,
                    scan_mode=scan_mode,
                    progress_callback=progress_callback,
                    customer_filter=customer_filter,
                )
                combined["sources_scanned"] += 1
                for k in (
                    "scanned",
                    "processed",
                    "errors",
                    "skipped",
                    "skipped_fast",
                    "texture_updated",
                    "new",
                    "changed",
                ):
                    combined[k] += stats.get(k, 0)
                combined["source_results"].append(stats)
                try:
                    totals = snapshot_progress_totals(self.db)
                    touch_session(
                        self.db,
                        jobs_done=int(combined.get("processed", 0) or 0),
                        light_done=int(totals.get("light_done", 0) or 0),
                        heavy_done=int(totals.get("heavy_done", 0) or 0),
                        search_ready=int(totals.get("search_ready", 0) or 0),
                        total_files=int(totals.get("total_files", 0) or 0),
                    )
                except Exception:
                    pass

            # RC1 hijyen: kaynak döngüsünden kaçan orphan + eksik dosyalar
            if not self._stop.is_set():
                try:
                    from core.index_ssot import (
                        fail_missing_light_pending,
                        list_orphan_fast_jobs,
                        write_pending_lt100_report,
                    )

                    missing_n = fail_missing_light_pending(self.db)
                    if missing_n:
                        combined["missing_failed"] = missing_n
                        logger.warning(
                            "RC1: diskte olmayan %s light pending → failed",
                            missing_n,
                        )
                    orphans = list_orphan_fast_jobs(self.db)
                    if orphans and (
                        mode
                        in (
                            IndexMode.FAST_ARCHIVE,
                            IndexMode.COMPLETE,
                            IndexMode.STANDARD,
                        )
                        or mode.value
                        in (
                            IndexMode.FAST_ARCHIVE.value,
                            IndexMode.COMPLETE.value,
                            IndexMode.STANDARD.value,
                        )
                    ):
                        logger.info("RC1 orphan light drain: %s dosya", len(orphans))
                        emit_orphan = progress_callback or (lambda _d: None)

                        def _emit(stage: str, **extra):
                            emit_orphan(
                                {
                                    "stage": stage,
                                    "current_source": "orphan",
                                    **combined,
                                    **extra,
                                }
                            )

                        src0 = sources[0] if sources else {}
                        self._process_pending_jobs(
                            orphans,
                            self._effective_worker_count(src0, light_pass=True),
                            combined,
                            False,
                            _emit,
                            light_pass=True,
                        )
                        combined["orphan_drained"] = len(orphans)
                    write_pending_lt100_report(
                        self.db,
                        str(Path(self.settings.cache_dir) / "reports" / "rc1_final"),
                    )
                except Exception as exc:
                    logger.warning("RC1 orphan/missing hijyen: %s", exc)

            self.db.set_meta("last_index_at", datetime.now(timezone.utc).isoformat())
            if not self._stop.is_set():
                if self.faiss.needs_rebuild and self.settings.ai_embedding_enabled:
                    self.faiss.rebuild_from_db(self.db.get_indexed_files())
                self.faiss.save()
            write_index_report(combined)
            if self._ai_fallback_warning:
                combined["ai_fallback_warning"] = self._ai_fallback_warning
            if progress_callback:
                progress_callback({"stage": "complete", **combined})
            return combined
        finally:
            try:
                if self._stop.is_set():
                    mark_interrupted(self.db)
                    combined["stopped"] = True
                else:
                    mark_completed(self.db)
            except Exception:
                pass
            if self._speed_session:
                self._speed_session.end_light()
                self._speed_session.total_files = combined.get(
                    "processed", 0
                ) + combined.get("skipped", 0)
                self._speed_session.light_processed = combined.get("processed", 0)
                self._speed_session.errors = combined.get("errors", 0)
                self._speed_session.pending_full_queue = self.db.count_pending_full()
                report = write_index_speed_report(
                    self._speed_session,
                    Path(self.settings.cache_dir) / "reports",
                )
                combined["speed_report_path"] = str(report)
            restore_index_settings(self.settings, snapshot)
            self._speed_session = None

    def run_source(
        self,
        source: dict[str, Any],
        scan_mode: str = ScanMode.QUICK.value,
        progress_callback: Callable[[dict], None] | None = None,
        customer_filter: str = "",
    ) -> dict:
        """Tek kaynağı incremental tarar."""
        self._stop.clear()
        started = time.perf_counter()
        monitor = None
        if getattr(self.settings, "resource_monitor_enabled", True):
            try:
                from core.resource_monitor import start_resource_monitor
                from core.work_profile import apply_process_priority

                apply_process_priority(self.settings)
                log_dir = Path(self.settings.db_path).parent / "logs"

                def _queue_snap() -> dict:
                    out: dict[str, Any] = {}
                    try:
                        out["pending_full"] = int(self.db.count_pending_full() or 0)
                    except Exception:
                        out["pending_full"] = None
                    return out

                monitor = start_resource_monitor(
                    log_dir=log_dir,
                    queue_snapshot=_queue_snap,
                    interval_sec=5.0,
                )
            except Exception:
                monitor = None
        source_id = source["id"]
        raw_path = source["root_path"]
        root_path = normalize_source_root(raw_path)
        deep = scan_mode == ScanMode.DEEP.value
        index_mode = (
            getattr(self.settings, "index_mode", IndexMode.STANDARD.value)
            or IndexMode.STANDARD.value
        )
        night_mode = index_mode == IndexMode.NIGHT_COMPLETE.value
        backfill_mode = index_mode == IndexMode.BACKFILL.value
        complete_mode = index_mode == IndexMode.COMPLETE.value
        completion_mode = backfill_mode
        fast_mode = index_mode == IndexMode.FAST_ARCHIVE.value

        stats = {
            "source_id": source_id,
            "source_name": source.get("name", ""),
            "scan_mode": scan_mode,
            "scanned": 0,
            "processed": 0,
            "errors": 0,
            "skipped": 0,
            "skipped_fast": 0,
            "texture_updated": 0,
            "new": 0,
            "changed": 0,
            "raw_path": raw_path,
            "normalized_path": root_path,
            "exists": os.path.isdir(root_path),
            "readable": False,
            "first_50_errors": [],
            "preview_generated": 0,
            "ai_from_cache": 0,
            "original_reopened": 0,
            "cache_hit": 0,
            "cache_miss": 0,
            "skipped_already_analyzed": 0,
            "skipped_quarantine": 0,
            "network_read_count": 0,
            "network_metadata_check_count": 0,
            "network_original_open_count": 0,
            "network_preview_open_count": 0,
            "original_file_open_count": 0,
            "analysis_preview_from_original": 0,
            "light_network_original_open_count": 0,
            "full_network_original_open_count": 0,
            "light_original_reopened": 0,
            "full_original_reopened": 0,
            "network_open_log": [],
            "local_cache_read_count": 0,
            "db_metadata_read_count": 0,
            "needs_medium_preview": 0,
            "ai_completed": 0,
            "ai_saved": 0,
            "embedding_saved": 0,
            "embedding_complete_saved": 0,
            "dino_embedding_saved": 0,
            "clip_embedding_saved": 0,
            "patch_embedding_saved": 0,
            "ai_final_saved": 0,
            "semantic_saved": 0,
            "dna_saved": 0,
            "ocr_saved": 0,
            "texture_saved": 0,
            "fast_index_done": 0,
            "skipped_unchanged": 0,
            "preview_backfill": 0,
            "hash_computed": 0,
            "light_open_once": 0,
            "light_open_reused": 0,
            "light_preview_created": 0,
            "light_metadata_saved": 0,
            "light_processed": 0,
            "heavy_processed": 0,
        }
        init_pipeline_steps(stats)

        def emit(stage: str, **extra):
            progress_emit.emit(stage, **extra)

        progress_emit = ThrottledProgress(
            lambda payload: (
                progress_callback(
                    {
                        "current_source": source.get("name", ""),
                        "scan_mode": scan_mode,
                        **stats,
                        **payload,
                    }
                )
                if progress_callback
                else None
            ),
            scan_every=250,
            process_every=25,
        )

        known_paths: list[str] = []
        pending_jobs: list[tuple[int, str, str]] = []
        pending_ids: set[int] = set()
        root_unreachable = not bool(stats["exists"])

        # SSOT backlog kök erişiminden BAĞIMSIZ — kaldığı yerden devam
        emit(
            "queue_prepare",
            active_phase="Kuyruk hazırlanıyor",
            current_file="SSOT backlog…",
        )
        light_jobs, heavy_jobs = self._collect_ssot_lanes(source_id, index_mode)
        pending_jobs = list(light_jobs) + list(heavy_jobs)
        pending_ids = {fid for fid, _, _ in pending_jobs}
        if pending_jobs:
            stats["resumed"] = len(pending_jobs)

        if root_unreachable:
            logger.warning(
                "Kaynak erişilemiyor (%s) — tarama atlandı, SSOT backlog=%s",
                root_path,
                len(pending_jobs),
            )
            stats["first_50_errors"].append(
                {
                    "path": root_path,
                    "stage": "source",
                    "error": "Kaynak erişilemiyor",
                }
            )
            stats["root_unreachable"] = True
            stats["failed_dirs"] = 1
            if not pending_jobs:
                stats.update(
                    {
                        "total_dirs_seen": 0,
                        "total_files_seen": 0,
                        "supported_images_found": 0,
                        "indexed_new": 0,
                        "updated_existing": 0,
                        "skipped_unchanged": 0,
                        "skipped_unsupported": 0,
                        "failed_files": 0,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                    }
                )
                stats["report_path"] = str(write_source_scan_report(stats))
                return stats
            stats["backlog_only"] = True
            stats["backlog_size"] = len(pending_jobs)
            stats["readable"] = False
            emit(
                "queue_prepare",
                active_phase="Kuyruk hazır (kök yok)",
                current_file=(
                    f"FAST {len(light_jobs):,} · GENERAL AI {len(heavy_jobs):,}"
                ),
                jobs_total=len(pending_jobs),
                jobs_done=0,
                light_jobs_total=len(light_jobs),
                heavy_jobs_total=len(heavy_jobs),
            )
            emit("backlog_only", backlog=len(pending_jobs))
            # Tarama yok — doğrudan aşağıdaki process bloğuna düş
            scanner = None
        else:
            stats["readable"] = True
            scanner = ArchiveScanner(
                root_path,
                self.settings.network_timeout_sec,
                self.settings.network_retry_count,
                excluded_roots=self._scanner_excluded_roots(root_path),
            )

        emit(
            "queue_prepare",
            active_phase="Kuyruk hazır",
            current_file=(
                f"FAST {len(light_jobs):,} · GENERAL AI {len(heavy_jobs):,}"
            ),
            jobs_total=len(pending_jobs),
            jobs_done=0,
            light_jobs_total=len(light_jobs),
            heavy_jobs_total=len(heavy_jobs),
        )

        emit("scan_start")
        cust = customer_filter or self.settings.customer_filter
        # Heavy/backfill modlarında uygun iş yoksa uzun envanteri atla.
        # Fast mode boş DB'de önce kaynağı tarayıp ilk light kuyruğunu kurmalıdır.
        empty_mode_backlog = (
            (night_mode and not heavy_jobs)
            or (
                complete_mode
                and not light_jobs
                and not heavy_jobs
            )
        )
        # RC2: backfill (Eksikleri Tamamla) asla boş diye taramayı atlamaz —
        # discovery + artifact reconcile zorunlu.
        skip_scan = (
            root_unreachable
            or empty_mode_backlog
            or night_mode
            or complete_mode
            or (
                (not backfill_mode)
                and getattr(self.settings, "index_drain_backlog_first", True)
                and len(pending_jobs)
                >= int(
                    getattr(self.settings, "index_backlog_skip_scan_threshold", 200)
                    or 200
                )
            )
        )
        if empty_mode_backlog and not pending_jobs:
            stats["skipped_empty_backlog"] = True
            emit(
                "queue_prepare",
                active_phase="Kaynak atlandı",
                current_file="Bu kaynakta bekleyen iş yok",
            )
        elif skip_scan and pending_jobs:
            # Önce SSOT kuyruğunu işle — envanter SAYAÇLARI BLOKLAMASIN.
            # Arşiv senkronu iş bitince (_sync_archive_inventory) yapılır.
            stats["backlog_only"] = True
            stats["backlog_size"] = len(pending_jobs)
            emit("backlog_only", backlog=len(pending_jobs))
        elif root_unreachable:
            pass  # tarama yok, backlog da boşsa zaten return edildi
        elif (complete_mode or backfill_mode or night_mode) and scanner is not None:
            # Güncelle / complete: güvenli envanter — done kayıtlarını bozma
            emit("archive_inventory_start")
            known_paths = self._sync_archive_inventory(
                scanner,
                source_id=source_id,
                customer_filter=cust,
                stats=stats,
                emit=emit,
            )
            stats["safe_inventory"] = True
        else:
            for scan_item in scanner.iter_files(
                customer_filter=cust,
                source_id=source_id,
            ):
                if not self._wait_if_paused():
                    break

                known_paths.append(scan_item.path)
                stats["scanned"] += 1
                existing = self.db.get_file_by_path(scan_item.path)

                partial = ""
                if deep:
                    partial = self._compute_partial_hash(scan_item.path, stats)

                if fast_mode:
                    handled = self._fast_scan_dispatch(
                        scan_item,
                        existing,
                        stats,
                        pending_jobs,
                        pending_ids,
                        emit,
                    )
                    if handled:
                        continue
                    # light_done + değişmemiş → dokunma
                    if (
                        existing is not None
                        and str(existing.get("light_status") or "") == "done"
                        and not ArchiveScanner.content_changed(
                            existing, scan_item.file_size, scan_item.mtime
                        )
                    ):
                        stats["skipped_fast"] += 1
                        stats["skipped"] += 1
                        continue
                    file_id = self.db.upsert_file(
                        {
                            "path": scan_item.path,
                            "filename": scan_item.filename,
                            "customer": scan_item.customer,
                            "file_size": scan_item.file_size,
                            "mtime": scan_item.mtime,
                            "partial_hash": (
                                existing.get("partial_hash", "") if existing else ""
                            ),
                            "source_id": source_id,
                            "status": "processing",
                        }
                    )
                    if existing is not None and ArchiveScanner.content_changed(
                        existing, scan_item.file_size, scan_item.mtime
                    ):
                        self.db.mark_file_pipeline_reset_on_change(
                            file_id,
                            content_changed=True,
                            cache_dir=self.settings.cache_dir,
                        )
                    elif existing is None:
                        pass
                    self.db.set_file_index_stage(file_id, PENDING_LIGHT)
                    if file_id not in pending_ids:
                        pending_jobs.append(
                            (file_id, scan_item.path, scan_item.filename)
                        )
                        pending_ids.add(file_id)
                    if stats["scanned"] % 250 == 0:
                        emit(
                            "scan",
                            current_file=scan_item.filename,
                            scanned=stats["scanned"],
                        )
                    continue

                content_changed = ArchiveScanner.content_changed(
                    existing, scan_item.file_size, scan_item.mtime
                )
                if (
                    deep
                    and existing
                    and partial
                    and str(existing.get("partial_hash") or "")
                    and partial != str(existing.get("partial_hash") or "")
                ):
                    content_changed = True
                light_done = str((existing or {}).get("light_status") or "") == "done"
                # Eski needs_reindex yalnız light henüz done değilken (status tuzağı yok)
                needs_reindex = content_changed or (
                    not light_done
                    and ArchiveScanner.needs_reindex(
                        existing,
                        scan_item.file_size,
                        scan_item.mtime,
                        existing.get("partial_hash", "") if existing else "",
                        partial,
                        deep_scan=deep,
                    )
                )
                stage = str((existing or {}).get("index_stage") or PENDING_LIGHT)
                if light_done and not content_changed:
                    stats["skipped_fast"] += 1
                    stats["skipped"] += 1
                    emit(
                        "scan",
                        current_file=scan_item.filename,
                        scanned=stats["scanned"],
                    )
                    continue
                if not needs_reindex and not completion_mode and not complete_mode:
                    if stage in (FULL_DONE, LIGHT_DONE):
                        if existing and self._needs_texture_update(existing):
                            if self._process_texture_only(
                                existing["id"],
                                existing.get("thumbnail_path", ""),
                                scan_item.filename,
                                scan_item.path,
                            ):
                                stats["texture_updated"] += 1
                            else:
                                stats["skipped"] += 1
                        else:
                            stats["skipped_fast"] += 1
                            stats["skipped"] += 1
                        emit(
                            "scan",
                            current_file=scan_item.filename,
                            scanned=stats["scanned"],
                        )
                        continue
                    if stage == PENDING_FULL and fast_mode:
                        stats["skipped_fast"] += 1
                        stats["skipped"] += 1
                        stats["skipped_deferred"] = (
                            int(stats.get("skipped_deferred", 0)) + 1
                        )
                        emit(
                            "scan",
                            current_file=scan_item.filename,
                            scanned=stats["scanned"],
                        )
                        continue
                    if not self._needs_feature_backfill(existing):
                        stats["skipped_fast"] += 1
                        stats["skipped"] += 1
                        emit(
                            "scan",
                            current_file=scan_item.filename,
                            scanned=stats["scanned"],
                        )
                        continue

                if existing is None:
                    stats["new"] += 1
                else:
                    stats["changed"] += 1

                file_id = self.db.upsert_file(
                    {
                        "path": scan_item.path,
                        "filename": scan_item.filename,
                        "customer": scan_item.customer,
                        "file_size": scan_item.file_size,
                        "mtime": scan_item.mtime,
                        "partial_hash": (
                            partial or existing.get("partial_hash", "")
                            if existing
                            else partial
                        ),
                        "source_id": source_id,
                        "status": "processing",
                        **self._format_descriptor_record(scan_item.path),
                    }
                )
                # SSOT: pipeline reset yalnız gerçek içerik değişiminde
                if existing is not None and content_changed:
                    self.db.mark_file_pipeline_reset_on_change(
                        file_id,
                        content_changed=True,
                        cache_dir=self.settings.cache_dir,
                    )
                self.db.set_file_index_stage(file_id, PENDING_LIGHT)
                if file_id not in pending_ids:
                    pending_jobs.append((file_id, scan_item.path, scan_item.filename))
                    pending_ids.add(file_id)
                # index_queue yazılmaz — tek kuyruk = files.light/heavy_status
                if stats["scanned"] % 250 == 0:
                    emit(
                        "scan",
                        current_file=scan_item.filename,
                        scanned=stats["scanned"],
                    )

        workers = self._effective_worker_count(source, light_pass=True)
        stats["worker_count_used"] = workers
        stats["index_ai"] = self._index_uses_ai()
        stats["index_mode"] = index_mode
        # Tarama yeni pending eklediyse SSOT'tan yeniden oku (queue rebuild değil — DB status)
        light_jobs, heavy_jobs = self._collect_ssot_lanes(source_id, index_mode)
        pending_jobs = list(light_jobs) + list(heavy_jobs)
        if pending_jobs and not self._stop.is_set():
            if complete_mode or completion_mode or night_mode:
                if self._speed_session:
                    self._speed_session.begin_light()
                stats["jobs_total"] = len(light_jobs) + len(heavy_jobs)
                stats["jobs_done"] = int(stats.get("jobs_done", 0) or 0)
                stats["light_jobs_total"] = len(light_jobs)
                stats["heavy_jobs_total"] = len(heavy_jobs)
                phase = (
                    "Genel Index"
                    if night_mode and not light_jobs
                    else "Hızlı + Genel"
                )
                emit(
                    "queue_prepare",
                    active_phase=phase,
                    current_file=(
                        f"FAST {len(light_jobs):,} · GENERAL AI {len(heavy_jobs):,}"
                    ),
                    light_jobs_total=len(light_jobs),
                    heavy_jobs_total=len(heavy_jobs),
                    jobs_total=stats["jobs_total"],
                    independent_lanes=True,
                    dual_pipeline=True,
                )
                IndexDualPipeline(self).run(
                    source=source,
                    light_jobs=light_jobs,
                    heavy_jobs=heavy_jobs,
                    stats=stats,
                    deep=deep,
                    emit=emit,
                )
                if self._speed_session:
                    self._speed_session.end_light()
                    self._speed_session.end_full()
                    self._speed_session.full_processed += int(
                        stats.get("heavy_processed", 0)
                    )
            elif fast_mode:
                if self._speed_session:
                    self._speed_session.begin_light()
                self._process_pending_jobs(
                    light_jobs,
                    workers,
                    stats,
                    deep,
                    emit,
                    light_pass=True,
                )
                if self._speed_session:
                    self._speed_session.end_light()
            else:
                light = self._is_light_index(deep)
                use_dual = light and (
                    self.settings.index_uses_ai() or self.settings.ocr_enabled
                )
                if use_dual:
                    if self._speed_session:
                        self._speed_session.begin_light()
                    IndexDualPipeline(self).run(
                        source=source,
                        light_jobs=light_jobs,
                        heavy_jobs=heavy_jobs,
                        stats=stats,
                        deep=deep,
                        emit=emit,
                    )
                    if self._speed_session:
                        self._speed_session.end_light()
                        self._speed_session.end_full()
                        self._speed_session.full_processed += int(
                            stats.get("heavy_processed", 0)
                        )
                else:
                    if self._speed_session and light:
                        self._speed_session.begin_light()
                    self._process_pending_jobs(
                        light_jobs if light else pending_jobs,
                        workers,
                        stats,
                        deep,
                        emit,
                        light_pass=light,
                    )
                    if self._speed_session and light:
                        self._speed_session.end_light()
                    if (
                        light
                        and heavy_jobs
                        and not self._stop.is_set()
                        and (
                            self.settings.index_uses_ai() or self.settings.ocr_enabled
                        )
                    ):
                        emit("heavy_backlog_start", jobs_total=len(heavy_jobs))
                        if self._speed_session:
                            self._speed_session.begin_full()
                        self._process_pending_jobs(
                            heavy_jobs,
                            self._effective_worker_count(
                                source, ai_enabled=True, light_pass=False
                            ),
                            stats,
                            deep,
                            emit,
                            light_pass=False,
                        )
                        if self._speed_session:
                            self._speed_session.end_full()
                            self._speed_session.full_processed += len(heavy_jobs)

        # Backlog-only turda tarama atlandıysa: iş BİTTİKTEN sonra arşiv senkronu
        # (önce sayaçlar ilerlesin; needs_reindex status tuzağı light_done sıfırlamasın).
        # Kök erişilemiyorsa envanter yok — backlog yine işlendi.
        if (
            stats.get("backlog_only")
            and not known_paths
            and not self._stop.is_set()
            and scanner is not None
            and not root_unreachable
        ):
            emit("archive_inventory_start")
            known_paths = self._sync_archive_inventory(
                scanner,
                source_id=source_id,
                customer_filter=cust,
                stats=stats,
                emit=emit,
            )

        if known_paths and not self._stop.is_set():
            stats["missing_marked"] = self.db.mark_missing_files_for_source(
                source_id, known_paths
            )
            if self.settings.purge_missing_after_scan and stats["missing_marked"]:
                stats["missing_purged"] = self.db.purge_missing_files(
                    source_id=source_id,
                    cache_dir=self.settings.cache_dir,
                )

        if not self._stop.is_set():
            self._maybe_save_faiss(force=True)
            self.db.update_source_scan_times(source_id, scan_mode)
            self.source_manager.update_source_stats(source_id)

        diag = (
            scanner.diagnostics
            if scanner is not None
            else {
                "total_dirs_seen": 0,
                "total_files_seen": 0,
                "supported_images_found": 0,
                "skipped_unsupported": 0,
                "failed_files": 0,
                "failed_dirs": 1 if root_unreachable else 0,
                "first_50_errors": [],
            }
        )
        stats.update(
            {
                "total_dirs_seen": diag["total_dirs_seen"],
                "total_files_seen": diag["total_files_seen"],
                "supported_images_found": diag["supported_images_found"],
                "indexed_new": stats["new"],
                "updated_existing": stats["changed"],
                "skipped_unchanged": stats["skipped_fast"],
                "skipped_unsupported": diag["skipped_unsupported"],
                "failed_files": stats["errors"] + diag["failed_files"],
                "failed_dirs": diag["failed_dirs"],
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        )
        stats["first_50_errors"] = (
            stats.get("first_50_errors", []) + diag["first_50_errors"]
        )[:50]
        stats["report_path"] = str(write_source_scan_report(stats))

        if self._ai_fallback_warning:
            stats["ai_fallback_warning"] = self._ai_fallback_warning

        emit("source_complete")
        try:
            from core.pipeline_diagnostics import format_pipeline_dashboard_log

            dash = self.db.count_index_pipeline_dashboard()
            logger.info(format_pipeline_dashboard_log(dash))
        except Exception:
            pass
        try:
            from core.resource_monitor import stop_resource_monitor

            stop_resource_monitor()
        except Exception:
            pass
        return stats

    def _process_pending_jobs(
        self,
        pending_jobs: list[tuple[int, str, str]],
        workers: int,
        stats: dict,
        deep: bool,
        emit,
        *,
        light_pass: bool = False,
    ) -> None:
        if not pending_jobs:
            return
        stats["jobs_total"] = len(pending_jobs)
        stats["jobs_done"] = 0
        stats["index_pass"] = "light" if light_pass else "full"
        mode = str(getattr(self.settings, "index_mode", "") or "")
        if mode == IndexMode.BACKFILL.value:
            phase = "Eksik Tamamlama"
        elif light_pass:
            phase = "Hızlı Index"
        else:
            phase = "Ağır Analiz"
        stats["active_phase"] = phase
        emit(
            "process_batch_start",
            jobs_total=len(pending_jobs),
            index_pass=stats["index_pass"],
            active_phase=phase,
        )
        if not light_pass:
            emit("ai_loading")
            self._ensure_extractor_for_index()
        else:
            self._ensure_extractor(
                use_ai=False,
                fast_hash_only=bool(
                    getattr(self.settings, "index_mode", "")
                    == IndexMode.FAST_ARCHIVE.value
                ),
            )
        if workers > 1 and len(pending_jobs) > 1 and (
            light_pass or not self._index_uses_ai()
        ):
            # Light (preview/hash) her zaman paralel olabilir.
            # AI/full pass GIL yüzünden varsayılan seri (DualPipeline heavy ayrı).
            from core.db import clear_active_write_batch, set_active_write_batch
            from core.db_write_batch import DbWriteBatch
            from core.work_profile import (
                apply_current_thread_priority,
                apply_process_priority,
                get_profile,
                refresh_adaptive_cap,
                release_batch_memory,
            )

            apply_process_priority(self.settings)
            profile = get_profile(self.settings)
            chunk_size = max(8, int(profile.chunk_size))
            base_workers = max(1, int(workers))
            batch = DbWriteBatch(
                self.settings.db_path,
                batch_size=profile.db_batch_size,
            )
            batch.open()
            set_active_write_batch(batch)
            try:
                for offset in range(0, len(pending_jobs), chunk_size):
                    if self._stop.is_set():
                        break
                    chunk = pending_jobs[offset : offset + chunk_size]
                    workers = max(
                        1,
                        min(base_workers, refresh_adaptive_cap(self.settings)),
                    )
                    stats["worker_count_used"] = workers
                    mon = None
                    try:
                        from core.resource_monitor import get_resource_monitor

                        mon = get_resource_monitor()
                        if mon:
                            mon.set_extra(
                                workers=workers,
                                jobs_done=stats.get("jobs_done", 0),
                                jobs_total=stats.get("jobs_total", 0),
                                index_pass=stats.get("index_pass"),
                            )
                    except Exception:
                        pass

                    def _run_one(fid, path, name):
                        apply_current_thread_priority("worker")
                        return self._process_file(
                            fid, path, name, stats, deep, light_pass
                        )

                    with ThreadPoolExecutor(
                        max_workers=workers, thread_name_prefix="vezir-index"
                    ) as pool:
                        futures = {
                            pool.submit(_run_one, fid, path, name): (fid, path, name)
                            for fid, path, name in chunk
                        }
                        for future in as_completed(futures):
                            if self._stop.is_set():
                                break
                            _fid, _path, name = futures[future]
                            if future.result():
                                stats["processed"] += 1
                                if light_pass:
                                    self._bump_stat(stats, "light_processed")
                                else:
                                    self._bump_stat(stats, "heavy_processed")
                            stats["jobs_done"] = int(stats.get("jobs_done", 0)) + 1
                            emit(
                                "process_done",
                                current_file=name,
                                jobs_done=stats["jobs_done"],
                                jobs_total=stats["jobs_total"],
                                light_processed=int(stats.get("light_processed", 0) or 0),
                                heavy_processed=int(stats.get("heavy_processed", 0) or 0),
                            )
                    batch.flush()
                    release_batch_memory()
            finally:
                clear_active_write_batch()
                batch.close()
        else:
            from core.db import clear_active_write_batch, set_active_write_batch
            from core.db_write_batch import DbWriteBatch
            from core.work_profile import (
                apply_current_thread_priority,
                apply_process_priority,
                get_profile,
                release_batch_memory,
            )

            apply_process_priority(self.settings)
            apply_current_thread_priority("worker")
            profile = get_profile(self.settings)
            batch = DbWriteBatch(
                self.settings.db_path,
                batch_size=profile.db_batch_size,
            )
            batch.open()
            set_active_write_batch(batch)
            try:
                for i, (file_id, path, filename) in enumerate(pending_jobs):
                    if self._stop.is_set():
                        break
                    stats["jobs_done"] = int(stats.get("jobs_done", 0))
                    current = self.db.get_file_by_id(file_id) or {}
                    pipeline_task = self._detect_pipeline_task(
                        current,
                        file_id=file_id,
                        light=light_pass,
                        source_path=path,
                    )
                    stats["pipeline_task"] = pipeline_task
                    emit(
                        "process",
                        current_file=filename,
                        jobs_done=stats["jobs_done"],
                        jobs_total=stats["jobs_total"],
                        index_pass="light" if light_pass else "full",
                        active_phase=phase,
                        pipeline_task=pipeline_task,
                    )
                    if self._process_file(
                        file_id, path, filename, stats, deep, light_pass
                    ):
                        stats["processed"] += 1
                        if light_pass:
                            self._bump_stat(stats, "light_processed")
                        else:
                            self._bump_stat(stats, "heavy_processed")
                    stats["jobs_done"] = int(stats.get("jobs_done", 0)) + 1
                    emit(
                        "process_done",
                        current_file=filename,
                        jobs_done=stats["jobs_done"],
                        jobs_total=stats["jobs_total"],
                        pipeline_task=pipeline_task,
                        light_processed=int(stats.get("light_processed", 0) or 0),
                        heavy_processed=int(stats.get("heavy_processed", 0) or 0),
                    )
                    if (i + 1) % max(8, profile.chunk_size) == 0:
                        batch.flush()
                        release_batch_memory()
            finally:
                clear_active_write_batch()
                batch.close()

        self._log_pipeline_save_summary(stats, force=True)

    def _split_light_heavy_jobs(
        self,
        pending_jobs: list[tuple[int, str, str]],
    ) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
        """DB alanlarıyla hızlı ayır — disk/preview I/O yok (toplu başlangıç)."""
        light_jobs: list[tuple[int, str, str]] = []
        full_jobs: list[tuple[int, str, str]] = []
        if not pending_jobs:
            return light_jobs, full_jobs

        by_id = {int(j[0]): j for j in pending_jobs}
        ids = list(by_id.keys())
        meta: dict[int, dict[str, Any]] = {}
        chunk = 900
        with self.db.connect() as conn:
            for i in range(0, len(ids), chunk):
                part = ids[i : i + chunk]
                ph = ",".join("?" * len(part))
                rows = conn.execute(
                    f"""
                    SELECT id, light_status, heavy_status, index_stage, status,
                           COALESCE(thumbnail_path,'') AS thumbnail_path,
                           COALESCE(feature_preview_path,'') AS feature_preview_path
                    FROM files WHERE id IN ({ph})
                    """,
                    part,
                ).fetchall()
                for row in rows:
                    meta[int(row["id"])] = dict(row)

        for fid, path, name in pending_jobs:
            existing = meta.get(fid) or {}
            stage = str(existing.get("index_stage") or PENDING_LIGHT)
            light_status = str(existing.get("light_status") or "")
            heavy_status = str(existing.get("heavy_status") or "")
            # Kalıcı ilerleme: light_done asla tekrar Hızlı Index kuyruğuna girmez
            if light_status == "done":
                if heavy_status != "done":
                    full_jobs.append((fid, path, name))
                continue
            light_jobs.append((fid, path, name))
        return light_jobs, full_jobs

    def _compute_partial_hash(self, path: str, stats: dict) -> str:
        self._bump_stat(stats, "hash_computed")
        return partial_file_hash(path)

    def _metadata_unchanged(
        self, existing: dict, file_size: int, mtime: float
    ) -> bool:
        if str(existing.get("status") or "") == "missing":
            return False
        if int(existing.get("file_size") or 0) != int(file_size):
            return False
        if abs(float(existing.get("mtime") or 0) - float(mtime)) > 0.5:
            return False
        return True

    def _needs_light_work_db_only(self, existing: dict) -> bool:
        """Yalnızca DB alanları — disk/NAS dokunulmaz."""
        stage = str(existing.get("index_stage") or "")
        light_status = str(existing.get("light_status") or "")
        if stage in (PENDING_LIGHT, NEEDS_MEDIUM_PREVIEW, FAILED):
            return True
        if light_status and light_status not in ("done",):
            return True
        if int(existing.get("needs_medium_preview") or 0):
            return True
        if (
            existing.get("physical_thumbnail_ready") != 1
            or existing.get("physical_preview_ready") != 1
        ):
            return True
        thumb = str(existing.get("thumbnail_path") or "").strip()
        fp = str(existing.get("feature_preview_path") or "").strip()
        return not thumb or not fp

    def _is_fast_db_skip(
        self, existing: dict, file_size: int, mtime: float
    ) -> bool:
        """Değişmemiş dosya — path+size+mtime + DB light_done, disk yok."""
        if not self._metadata_unchanged(existing, file_size, mtime):
            return False
        if str(existing.get("status") or "") not in ("indexed",):
            return False
        stage = str(existing.get("index_stage") or "")
        light_status = str(existing.get("light_status") or "")
        if stage not in (LIGHT_DONE, FULL_DONE) and light_status != "done":
            return False
        return not self._needs_light_work_db_only(existing)

    def _fast_scan_dispatch(
        self,
        scan_item,
        existing: dict | None,
        stats: dict,
        pending_jobs: list[tuple[int, str, str]],
        pending_ids: set[int],
        emit,
    ) -> bool:
        """Hızlı index tarama — True ise dosya işlendi, devam et."""
        if existing is None:
            stats["new"] += 1
            return False

        if self._metadata_unchanged(existing, scan_item.file_size, scan_item.mtime):
            physical_ready = (
                existing.get("physical_thumbnail_ready") == 1
                and existing.get("physical_preview_ready") == 1
            )
            if not physical_ready:
                thumb_ready = local_artifact_exists(
                    str(existing.get("thumbnail_path") or ""), stats=stats
                )
                preview_ready = local_artifact_exists(
                    str(existing.get("feature_preview_path") or ""), stats=stats
                )
                self.db.update_physical_readiness(
                    int(existing["id"]),
                    thumbnail_ready=thumb_ready,
                    preview_ready=preview_ready,
                    requeue_missing=True,
                )
                existing["physical_thumbnail_ready"] = 1 if thumb_ready else 0
                existing["physical_preview_ready"] = 1 if preview_ready else 0
                existing["needs_medium_preview"] = (
                    0 if thumb_ready and preview_ready else 1
                )

        if self._is_fast_db_skip(existing, scan_item.file_size, scan_item.mtime):
            stats["skipped_fast"] += 1
            stats["skipped"] += 1
            stats["skipped_unchanged"] += 1
            emit(
                "scan",
                current_file=scan_item.filename,
                scanned=stats["scanned"],
            )
            return True

        if self._metadata_unchanged(existing, scan_item.file_size, scan_item.mtime):
            if self._needs_light_work_db_only(existing):
                stats["preview_backfill"] = int(stats.get("preview_backfill", 0)) + 1
                fid = int(existing["id"])
                if fid not in pending_ids:
                    pending_jobs.append((fid, scan_item.path, scan_item.filename))
                    pending_ids.add(fid)
                emit(
                    "scan",
                    current_file=scan_item.filename,
                    scanned=stats["scanned"],
                )
                return True
            stats["skipped_fast"] += 1
            stats["skipped"] += 1
            stats["skipped_unchanged"] += 1
            emit(
                "scan",
                current_file=scan_item.filename,
                scanned=stats["scanned"],
            )
            return True

        stats["changed"] += 1
        return False

    def _is_light_unchanged(self, existing: dict, source_path: str = "") -> bool:
        """Geriye uyumluluk — hızlı atlama için DB-only skip kullan."""
        return self._is_fast_db_skip(
            existing,
            int(existing.get("file_size") or 0),
            float(existing.get("mtime") or 0),
        )

    def _run_light_asset_extract(
        self,
        path: str,
        current: dict[str, Any],
        stats: dict,
    ) -> LightAssetResult | None:
        """Raster light pass — tek açılışta metadata + thumb + medium preview."""
        plugin = self.plugin_registry.resolve(path)
        if not plugin or getattr(plugin, "format_family", "") != "raster":
            return None
        ext = Path(path).suffix.lower()
        if ext in OPTIONAL_PREVIEW_EXTENSIONS:
            return None

        mtime = float(current.get("mtime") or 0)
        reused = try_reuse_light_assets(
            source_path=path,
            thumbnailer=self.thumbnailer,
            feature_preview=self.feature_preview,
            source_mtime=mtime,
        )
        if reused is not None:
            self._bump_stat(stats, "light_open_reused")
            return reused

        # Light asset extract her zaman orijinal açabilir (heavy repair dahil).
        # Explicit False: shared stats / heavy TLS bayrağından etkilenmez.
        with network_original_open_slot(
            self.settings,
            path,
            stats,
            reason="light_extract_assets_once",
            deep_from_cache=False,
        ):
            result = extract_light_assets_once(
                path,
                thumbnailer=self.thumbnailer,
                feature_preview=self.feature_preview,
                source_mtime=mtime,
            )
        if result.success:
            self._bump_stat(stats, "light_open_once")
            if result.metadata_saved:
                self._bump_stat(stats, "light_metadata_saved")
            if result.preview_generated:
                self._bump_stat(stats, "light_preview_created")
                self._bump_stat(stats, "preview_generated")
                if self._speed_session:
                    self._speed_session.preview_generated += 1
        return result

    def _analysis_from_light_assets(
        self,
        light_assets: LightAssetResult,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        fp_path = str(light_assets.feature_preview_path or "")
        thumb_path = str(light_assets.thumbnail_path or "")
        return {
            "fp_path": fp_path,
            "fp_version": FEATURE_PREVIEW_VERSION,
            "fp_mtime": float(current.get("mtime") or 0),
            "feature_source": fp_path or thumb_path,
            "analysis_source": "light_extract_once",
            "ai_analysis_from_cache": bool(light_assets.reused_cache),
            "original_reopened": False,
            "preview_generated": bool(light_assets.preview_generated),
        }

    def _partition_deep_jobs(
        self,
        jobs: list[tuple[int, str, str]],
    ) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
        """Medium preview eksiklerini light pass'e ayır; deep yalnızca cache'ten."""
        light_jobs: list[tuple[int, str, str]] = []
        deep_jobs: list[tuple[int, str, str]] = []
        for fid, path, name in jobs:
            existing = self.db.get_file_by_id(fid) or {}
            stage = str(existing.get("index_stage") or "")
            if stage == NEEDS_MEDIUM_PREVIEW or needs_medium_preview(
                existing,
                source_path=path,
                feature_preview_cache=self.feature_preview,
            ):
                light_jobs.append((fid, path, name))
            elif has_preview_artifacts_for_deep(
                existing,
                source_path=path,
                feature_preview_cache=self.feature_preview,
            ):
                deep_jobs.append((fid, path, name))
            elif has_thumbnail_artifact(existing):
                self.db.set_file_index_stage(fid, NEEDS_MEDIUM_PREVIEW)
                light_jobs.append((fid, path, name))
            else:
                light_jobs.append((fid, path, name))
        return light_jobs, deep_jobs

    def _collect_ssot_lanes(
        self,
        source_id: int,
        index_mode: str,
    ) -> tuple[list[tuple[int, str, str]], list[tuple[int, str, str]]]:
        """SSOT: Hızlı = light pending; Genel AI = light done + heavy pending.

        RC2: backfill/complete/night öncesi artifact reconcile —
        heavy_done ama eksik blob gizlenemez.
        """
        from core.artifact_status import reconcile_source_artifacts
        from core.index_ssot import list_fast_jobs, list_general_ai_jobs

        mode = str(index_mode or "")
        if mode in (
            IndexMode.BACKFILL.value,
            IndexMode.COMPLETE.value,
            IndexMode.NIGHT_COMPLETE.value,
            IndexMode.STANDARD.value,
        ):
            try:
                reconcile_source_artifacts(self.db, source_id)
            except Exception:
                logger.exception(
                    "artifact reconcile failed source_id=%s", source_id
                )
        if mode == IndexMode.FAST_ARCHIVE.value:
            return list_fast_jobs(self.db, source_id), []
        if mode == IndexMode.NIGHT_COMPLETE.value:
            return [], list_general_ai_jobs(self.db, source_id)
        return (
            list_fast_jobs(self.db, source_id),
            list_general_ai_jobs(self.db, source_id),
        )

    def _sync_archive_inventory(
        self,
        scanner,
        *,
        source_id: int,
        customer_filter: str | None,
        stats: dict,
        emit,
    ) -> list[str]:
        """Disk ↔ index senkronu (işlem sonrası). light_done'u status tuzağıyla sıfırlamaz.

        Yalnız: yeni dosya, missing geri gelen, gerçek size/mtime değişimi.
        """
        known: list[str] = []
        for scan_item in scanner.iter_files(
            customer_filter=customer_filter,
            source_id=source_id,
        ):
            if not self._wait_if_paused():
                break
            known.append(scan_item.path)
            stats["scanned"] = int(stats.get("scanned", 0) or 0) + 1
            existing = self.db.get_file_by_path(scan_item.path)
            if existing is None:
                file_id = self.db.upsert_file(
                    {
                        "path": scan_item.path,
                        "filename": scan_item.filename,
                        "customer": scan_item.customer,
                        "file_size": scan_item.file_size,
                        "mtime": scan_item.mtime,
                        "partial_hash": "",
                        "source_id": source_id,
                        "status": "pending",
                    }
                )
                self.db.set_file_index_stage(file_id, PENDING_LIGHT)
                stats["new"] = int(stats.get("new", 0) or 0) + 1
                continue
            if ArchiveScanner.content_changed(
                existing, scan_item.file_size, scan_item.mtime
            ):
                file_id = int(existing["id"])
                self.db.mark_file_pipeline_reset_on_change(
                    file_id,
                    content_changed=True,
                    cache_dir=self.settings.cache_dir,
                )
                self.db.upsert_file(
                    {
                        "path": scan_item.path,
                        "filename": scan_item.filename,
                        "customer": scan_item.customer,
                        "file_size": scan_item.file_size,
                        "mtime": scan_item.mtime,
                        "source_id": source_id,
                        "status": "pending",
                    }
                )
                self.db.set_file_index_stage(file_id, PENDING_LIGHT)
                stats["changed"] = int(stats.get("changed", 0) or 0) + 1
            if int(stats.get("scanned", 0) or 0) % 500 == 0:
                emit(
                    "scan",
                    current_file=scan_item.filename,
                    scanned=stats["scanned"],
                    archive_inventory=True,
                )
        return known

    def _collect_resume_jobs(
        self,
        source_id: int,
        index_mode: str,
    ) -> list[tuple[int, str, str]]:
        """Geriye uyum — SSOT lane birleşimi (queue rebuild yok)."""
        light, heavy = self._collect_ssot_lanes(source_id, index_mode)
        seen: set[int] = set()
        out: list[tuple[int, str, str]] = []
        for job in list(light) + list(heavy):
            fid = int(job[0])
            if fid in seen:
                continue
            seen.add(fid)
            out.append(job)
        return out

    def _scanner_excluded_roots(self, source_root: str) -> list[str]:
        root_key = normalize_source_root(source_root).casefold().rstrip("\\/")
        project_root = os.path.dirname(os.path.abspath(self.settings.cache_dir))
        protected = [
            self.settings.cache_dir,
            os.path.dirname(os.path.abspath(self.settings.db_path)),
            os.path.join(project_root, ".venv"),
        ]
        return [
            path
            for path in protected
            if normalize_source_root(path).casefold().startswith(root_key + "\\")
        ]

    def _is_light_index(self, deep_scan: bool) -> bool:
        """Hızlı katman: önizleme + hash + doku — PC'yi yormadan aranabilir yap."""
        if deep_scan:
            return False
        if self._search_priority:
            return True
        return bool(getattr(self.settings, "index_light_first", True))

    def _effective_worker_count(
        self,
        source: dict[str, Any],
        *,
        ai_enabled: bool | None = None,
        light_pass: bool = True,
    ) -> int:
        from core.background_index import effective_worker_count

        return effective_worker_count(
            self.settings,
            source_type=source.get("source_type", ""),
            ai_enabled=self._index_uses_ai() if ai_enabled is None else ai_enabled,
            light_pass=light_pass,
            source_path=str(source.get("root_path") or ""),
        )

    def _should_skip_quarantine(self, record: dict[str, Any]) -> bool:
        if not getattr(self.settings, "index_skip_quarantine_retry", True):
            return False
        return is_quarantined_file(record)

    def _bump_stat(self, stats: dict, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            stats[key] = int(stats.get(key, 0)) + amount

    def _log_pipeline_save_summary(self, stats: dict, *, force: bool = False) -> None:
        completed = int(stats.get("ai_completed", 0) or 0)
        if not force and completed > 0 and completed % 50 != 0:
            return
        logger.info(
            "%s | AI completed: %s | embedding_saved: %s | ocr_saved: %s",
            format_pipeline_step_summary(stats),
            completed,
            int(stats.get("embedding_saved", 0) or 0),
            int(stats.get("ocr_saved", 0) or 0),
        )

    def _record_pipeline_saves(
        self,
        stats: dict,
        *,
        light: bool,
        existing: dict[str, Any],
        existing_feat: dict[str, Any],
        existing_tm: dict[str, Any],
        features: Any,
        merged_texture_map: dict[str, Any],
        ocr_text: str,
    ) -> None:
        """DB'ye yazılan artefaktları oturum sayacına işle."""
        init_pipeline_steps(stats)
        if not light:
            self._bump_stat(stats, "ai_completed")
            bump_pipeline_step(stats, "ai", "attempted")
        had_dino = bool(existing_feat.get("dino_embedding"))
        had_clip = bool(existing_feat.get("clip_embedding"))
        has_dino = bool(getattr(features, "dino_embedding", None))
        has_clip = bool(getattr(features, "clip_embedding", None))
        had_emb = had_dino or had_clip
        has_emb = has_dino or has_clip
        had_complete_emb = had_dino and had_clip
        has_complete_emb = has_dino and has_clip
        if not light:
            if self.settings.index_uses_ai():
                bump_pipeline_step(stats, "embedding", "attempted")
                if has_dino and not had_dino:
                    self._bump_stat(stats, "dino_embedding_saved")
                if has_clip and not had_clip:
                    self._bump_stat(stats, "clip_embedding_saved")
                if has_complete_emb and not had_complete_emb:
                    self._bump_stat(stats, "embedding_complete_saved")
                if has_emb and not had_emb:
                    self._bump_stat(stats, "embedding_saved")
                    bump_pipeline_step(stats, "embedding", "saved")
                elif not has_emb:
                    bump_pipeline_step(stats, "embedding", "failed")
            else:
                bump_pipeline_step(stats, "embedding", "skipped")
        had_patch = bool(existing_feat.get("patch_embeddings_meta"))
        has_patch = bool(getattr(features, "patch_embeddings_meta", None))
        if not light and has_patch and not had_patch:
            self._bump_stat(stats, "patch_embedding_saved")
        had_ocr = bool(str(existing.get("ocr_text") or "").strip())
        has_ocr = bool(str(ocr_text or "").strip())
        if not light:
            if self.settings.ocr_enabled:
                bump_pipeline_step(stats, "ocr", "attempted")
                if has_ocr and not had_ocr:
                    self._bump_stat(stats, "ocr_saved")
                    bump_pipeline_step(stats, "ocr", "saved")
                elif not has_ocr:
                    bump_pipeline_step(stats, "ocr", "failed")
            else:
                bump_pipeline_step(stats, "ocr", "skipped")
        had_sem = texture_map_has_semantic_tags(existing_tm)
        has_sem = texture_map_has_semantic_tags(merged_texture_map)
        if not light:
            bump_pipeline_step(stats, "semantic", "attempted")
            if has_sem and not had_sem:
                self._bump_stat(stats, "semantic_saved")
                bump_pipeline_step(stats, "semantic", "saved")
            elif not has_sem:
                bump_pipeline_step(stats, "semantic", "failed")
        had_dna = texture_map_has_pattern_dna(existing_tm)
        has_dna = texture_map_has_pattern_dna(merged_texture_map)
        if not light:
            bump_pipeline_step(stats, "dna", "attempted")
            if has_dna and not had_dna:
                self._bump_stat(stats, "dna_saved")
                bump_pipeline_step(stats, "dna", "saved")
            elif not has_dna:
                bump_pipeline_step(stats, "dna", "failed")
        had_tex = bool(existing_feat.get("phash")) and bool(
            existing_feat.get("texture_features")
        )
        has_tex = bool(getattr(features, "phash", None)) and bool(
            getattr(features, "texture_features", None)
        )
        if not light:
            bump_pipeline_step(stats, "texture", "attempted")
            if has_tex and not had_tex:
                self._bump_stat(stats, "texture_saved")
                bump_pipeline_step(stats, "texture", "saved")
            elif not has_tex:
                bump_pipeline_step(stats, "texture", "failed")
            had_final = (
                bool(existing.get("thumbnail_path"))
                and bool(existing.get("feature_preview_path"))
                and had_complete_emb
                and had_patch
                and had_sem
                and had_dna
                and had_tex
            )
            has_final = (
                has_complete_emb
                and has_patch
                and has_sem
                and has_dna
                and has_tex
            )
            if has_final and not had_final:
                self._bump_stat(stats, "ai_final_saved")
        if light:
            self._bump_stat(stats, "fast_index_done")
        if not light:
            self._bump_stat(stats, "ai_saved")
            if has_sem or has_dna or has_tex or has_emb:
                bump_pipeline_step(stats, "ai", "saved")
            else:
                bump_pipeline_step(stats, "ai", "failed")
        self._log_pipeline_save_summary(stats)

    def _should_skip_deep_reanalysis(
        self, file_id: int, light: bool, stats: dict
    ) -> bool:
        if light:
            return False
        if getattr(self.settings, "index_mode", "") not in (
            IndexMode.BACKFILL.value,
            IndexMode.COMPLETE.value,
        ):
            return False
        existing = self.db.get_file_by_id(file_id) or {}
        feat = self.db.get_features(file_id) or {}
        from core.manual_label_guard import parse_texture_map

        tm = parse_texture_map(feat.get("texture_map"))
        skip_dna = getattr(self.settings, "index_skip_reanalyse_dna", True)
        skip_sem = getattr(self.settings, "index_skip_reanalyse_semantic", True)
        has_dna = texture_map_has_pattern_dna(tm) if skip_dna else False
        has_sem = texture_map_has_semantic_tags(tm) if skip_sem else False
        needs_ocr = self.settings.ocr_enabled and not (
            existing.get("ocr_text") or ""
        ).strip()
        needs_ai = self.settings.index_uses_ai() and (
            not feat.get("dino_embedding") or not feat.get("clip_embedding")
        )
        needs_texture = not feat.get("phash") or not feat.get("texture_features")
        needs_patch = bool(self.settings.ai_embedding_enabled) and not feat.get(
            "patch_embeddings_meta"
        )
        if (
            has_dna
            and has_sem
            and not needs_ocr
            and not needs_ai
            and not needs_texture
            and not needs_patch
        ):
            self._bump_stat(stats, "skipped_already_analyzed")
            return True
        return False

    def _detect_pipeline_task(
        self,
        record: dict[str, Any],
        *,
        file_id: int,
        light: bool,
        source_path: str = "",
    ) -> str:
        """UI için şu anki index adımı."""
        from core.manual_label_guard import parse_texture_map

        path = source_path or str(record.get("path") or "")
        if needs_medium_preview(
            record,
            source_path=path,
            feature_preview_cache=self.feature_preview,
        ):
            return "Preview üretiliyor"
        if not has_thumbnail_artifact(record):
            return "Thumbnail üretiliyor"
        feat = self.db.get_features(file_id) or {}
        if self.settings.index_uses_ai() and (
            not feat.get("dino_embedding") or not feat.get("clip_embedding")
        ):
            return "Embedding üretiliyor"
        if light:
            return "Önizleme ve embedding hazırlanıyor"
        tm = parse_texture_map(feat.get("texture_map"))
        if self.settings.ocr_enabled and not (record.get("ocr_text") or "").strip():
            return "OCR yapılıyor"
        if not texture_map_has_semantic_tags(tm):
            return "Semantic tag üretiliyor"
        if not texture_map_has_pattern_dna(tm):
            return "Pattern DNA çıkarılıyor"
        if not feat.get("texture_features") or not feat.get("phash"):
            return "Texture / color / repeat çıkarılıyor"
        if not feat.get("patch_embeddings_meta") and self.settings.ai_embedding_enabled:
            return "Patch embedding üretiliyor"
        return "AI analiz tamamlanıyor"

    def _process_file(
        self,
        file_id: int,
        path: str,
        filename: str,
        stats: dict,
        deep_scan: bool = False,
        light_pass: bool | None = None,
    ) -> bool:
        if not self._wait_if_paused():
            return False
        if is_ignored_artifact_path(path):
            self.db.exclude_os_metadata_artifacts()
            return False
        light = (
            self._is_light_index(deep_scan) if light_pass is None else bool(light_pass)
        )
        deep_scope_bound = False
        try:
            from pathlib import Path as _Path

            current = self.db.get_file_by_id(file_id) or {}
            track_db_metadata_read(stats)
            if self._should_skip_quarantine(current):
                # Sessiz return False YASAK — pending↔processing spin üretmesin
                self._bump_stat(stats, "skipped_quarantine")
                qreason = str(current.get("quarantine_reason") or "").strip() or "quarantine"
                msg = str(current.get("error_msg") or "") or (
                    f"skip_quarantine:{qreason}"
                )
                logger.info(
                    "early_exit file_id=%s skip_quarantine reason=%s light=%s",
                    file_id,
                    qreason,
                    light,
                )
                if light:
                    self.db.mark_light_failed(
                        file_id, reason=qreason, error_msg=msg
                    )
                else:
                    self.db.mark_heavy_failed(
                        file_id, reason=qreason, error_msg=msg
                    )
                return False
            if self._should_skip_deep_reanalysis(file_id, light, stats):
                self.db.set_file_index_stage(file_id, FULL_DONE)
                return True

            ext = str(current.get("format_extension") or "").lower()
            if not ext:
                ext = _Path(path).suffix.lower()

            deep_from_cache = not light
            # stats alanı yalnızca log/gözlem için; guard kararı thread-local.
            stats["_deep_from_cache"] = deep_from_cache
            stats["_index_file_path"] = path
            bind_deep_from_cache(deep_from_cache)
            deep_scope_bound = True
            repaired_light_assets: LightAssetResult | None = None
            if not light:
                has_thumb = has_thumbnail_artifact(current, stats=stats)
                has_med = has_medium_preview(
                    current,
                    source_path=path,
                    feature_preview_cache=self.feature_preview,
                    stats=stats,
                )
                # DB'de path var ama dosya yok → bir kez onar.
                # SSOT: light_status='done' ASLA pending'e düşmez.
                if not has_thumb or not has_med:
                    try:
                        with deep_from_cache_scope(False):
                            repaired_light_assets = self._run_light_asset_extract(
                                path, current, stats
                            )
                        if repaired_light_assets and repaired_light_assets.success:
                            has_thumb = local_artifact_exists(
                                repaired_light_assets.thumbnail_path,
                                stats=stats,
                            )
                            has_med = local_artifact_exists(
                                repaired_light_assets.feature_preview_path,
                                stats=stats,
                            )
                    except Exception as exc:
                        logger.warning(
                            "heavy preview repair failed file_id=%s: %s",
                            file_id,
                            exc,
                        )
                if not has_thumb and not has_med:
                    # WAITING_PREVIEW: retry edilebilir, gerçek failure değildir.
                    self._bump_stat(stats, "pending_preview")
                    logger.info(
                        "early_exit file_id=%s skip=pending_preview "
                        "(WAITING_PREVIEW, light done korunur)",
                        file_id,
                    )
                    self.db.mark_heavy_waiting_preview(file_id)
                    return False
                if has_thumb and not has_med:
                    # WAITING_PREVIEW: medium henüz üretilemedi.
                    self._bump_stat(stats, "needs_medium_preview")
                    logger.info(
                        "early_exit file_id=%s skip=needs_medium_preview "
                        "(WAITING_PREVIEW, light done korunur)",
                        file_id,
                    )
                    self.db.mark_heavy_waiting_preview(file_id)
                    return False
            reusable = self.db.find_reusable_thumbnail(
                filename=filename,
                file_size=int(current.get("file_size") or 0),
                mtime=float(current.get("mtime") or 0),
                exclude_file_id=file_id,
            )
            reusable_thumb = str((reusable or {}).get("thumbnail_path") or "")
            if reusable_thumb and not local_artifact_exists(reusable_thumb, stats=stats):
                reusable_thumb = ""
            # Preview ownership contract: light_status='done' requires a real
            # thumbnail/medium preview. Heavy never owns source-file rendering,
            # so neither heavy formats nor large TIFFs may be deferred to it.

            render_path = path
            unsupported = 0
            preview_status = "preview_ok"
            render_err = ""
            plugin = self.plugin_registry.resolve(path)
            format_record = self._format_descriptor_record(path)
            plugin_metadata: dict[str, Any] = {}
            plugin_semantic_tags: list[str] = []
            parser_errors: list[str] = []
            metadata_status = "not_supported"
            semantic_status = "deferred_heavy" if light else "disabled"
            thumbnail_status = "pending"
            light_assets: LightAssetResult | None = repaired_light_assets

            if light and render_path == path:
                light_assets = self._run_light_asset_extract(path, current, stats)
                if light_assets and light_assets.success:
                    plugin_metadata = dict(light_assets.metadata or {})
                    metadata_status = light_assets.metadata_status or "metadata_ok"
                    parser_errors.extend(light_assets.parser_errors or [])
                elif light_assets and not light_assets.success:
                    if light_assets.error:
                        parser_errors.append(light_assets.error)
                    light_assets = None

            if not light:
                from plugins.base import PluginResult

                cached_meta = str(current.get("format_metadata") or "{}")
                try:
                    plugin_metadata = json.loads(cached_meta) if cached_meta else {}
                except json.JSONDecodeError:
                    plugin_metadata = {}
                metadata_result = PluginResult(
                    success=bool(plugin_metadata),
                    metadata=plugin_metadata,
                    status="cache_reused" if plugin_metadata else "not_supported",
                )
                metadata_status = metadata_result.status
                if metadata_result.success:
                    plugin_metadata = dict(metadata_result.metadata or {})
            elif plugin and light_assets is None:
                with network_original_open_slot(
                    self.settings, path, stats, reason="plugin_extract_metadata"
                ):
                    metadata_result = self.plugin_registry.safe_call(
                        plugin, "extract_metadata", path
                    )
                metadata_status = metadata_result.status
                if metadata_result.success:
                    plugin_metadata = dict(metadata_result.metadata or {})
                elif metadata_result.error:
                    parser_errors.append(metadata_result.error)

            if plugin and ext in (".pdf", ".svg") and light:
                with network_preview_open_slot(
                    self.settings, path, stats, reason="plugin_extract_preview"
                ):
                    plugin_preview = self.plugin_registry.safe_call(
                        plugin, "extract_preview", path
                    )
                if plugin_preview.success and plugin_preview.artifact_path:
                    render_path = plugin_preview.artifact_path
                    preview_status = plugin_preview.status
                elif plugin_preview.error:
                    parser_errors.append(plugin_preview.error)

            if light_assets and light_assets.success:
                thumb = ThumbnailResult(
                    success=True,
                    thumbnail_path=light_assets.thumbnail_path,
                    width=int(light_assets.thumbnail_width or 0),
                    height=int(light_assets.thumbnail_height or 0),
                )
                thumbnail_status = light_assets.thumbnail_status or "thumbnail_ok"
                if light_assets.feature_preview_path:
                    render_path = light_assets.feature_preview_path
            elif deep_from_cache:
                thumb = ThumbnailResult(
                    success=True,
                    thumbnail_path=str(current.get("thumbnail_path") or ""),
                    width=int(current.get("width") or 0),
                    height=int(current.get("height") or 0),
                )
                thumbnail_status = str(current.get("thumbnail_status") or "cache_reused")
                fp_existing = resolve_medium_preview_path(
                    current,
                    source_path=path,
                    feature_preview_cache=self.feature_preview,
                    stats=stats,
                )
                if fp_existing:
                    render_path = fp_existing
            elif reusable_thumb:
                thumb = ThumbnailResult(
                    success=True,
                    thumbnail_path=reusable_thumb,
                    width=int((reusable or {}).get("width") or 0),
                    height=int((reusable or {}).get("height") or 0),
                )
                thumbnail_status = "cache_reused"
            else:
                thumb = ThumbnailResult(success=False, error="")
                if plugin and render_path == path and plugin.format_family == "raster":
                    with network_original_open_slot(
                        self.settings, path, stats, reason="plugin_extract_thumbnail"
                    ):
                        plugin_thumb = self.plugin_registry.safe_call(
                            plugin, "extract_thumbnail", path
                        )
                    if plugin_thumb.success:
                        thumb = ThumbnailResult(
                            success=True,
                            thumbnail_path=plugin_thumb.artifact_path,
                            width=int(plugin_thumb.metadata.get("width") or 0),
                            height=int(plugin_thumb.metadata.get("height") or 0),
                        )
                        thumbnail_status = plugin_thumb.status
                    elif plugin_thumb.error:
                        parser_errors.append(plugin_thumb.error)
                if not thumb.success:
                    thumb_source = (
                        render_path if ext in OPTIONAL_PREVIEW_EXTENSIONS else path
                    )
                    with network_original_open_slot(
                        self.settings,
                        thumb_source,
                        stats,
                        reason="thumbnailer_create",
                    ):
                        thumb = self.thumbnailer.create(thumb_source)
                    if thumb.success:
                        thumbnail_status = "legacy_fallback_ok"
            if not thumb.success:
                if (
                    ext in (".tif", ".tiff")
                    and "timeout" in (thumb.error or "").lower()
                ):
                    self.db.set_file_index_stage(file_id, NEEDS_REVIEW, needs_review=1)
                raise RuntimeError(thumb.error or "Thumbnail üretilemedi")

            if self._speed_session:
                self._speed_session.thumbnails += 1

            existing = self.db.get_file_by_id(file_id) or current
            if light_assets and light_assets.success:
                analysis = self._analysis_from_light_assets(light_assets, existing)
            else:
                analysis = self._resolve_feature_analysis_source(
                    path=path,
                    render_path=render_path,
                    thumb_path=thumb.thumbnail_path,
                    existing=existing,
                    light=light,
                    unsupported=unsupported,
                    stats=stats,
                    allow_original_open=light and not deep_from_cache,
                    deep_from_cache=deep_from_cache,
                )
            fp_path = analysis["fp_path"]
            fp_version = int(analysis["fp_version"] or 0)
            fp_mtime = float(analysis["fp_mtime"] or 0)
            feature_source = analysis["feature_source"]
            analysis_source = analysis["analysis_source"]
            ai_analysis_from_cache = bool(analysis["ai_analysis_from_cache"])
            original_reopened = bool(analysis["original_reopened"])

            full_hash = ""
            if (
                self.settings.use_full_hash
                and deep_scan
                and light
                and getattr(self.settings, "index_mode", "")
                != IndexMode.FAST_ARCHIVE.value
            ):
                with network_original_open_slot(
                    self.settings, path, stats, reason="full_file_hash"
                ):
                    full_hash = full_file_hash(path)
                    self._bump_stat(stats, "hash_computed")
            elif self.settings.use_full_hash and deep_scan:
                full_hash = str(current.get("full_hash") or "")

            if not light and plugin and getattr(
                self.settings, "semantic_text_search_enabled", False
            ):
                semantic_source = fp_path or thumb.thumbnail_path or ""
                if semantic_source and local_artifact_exists(semantic_source, stats=stats):
                    semantic_result = self.plugin_registry.safe_call(
                        plugin, "extract_semantic_tags", semantic_source
                    )
                    semantic_status = semantic_result.status
                    if semantic_result.success:
                        plugin_semantic_tags = list(
                            semantic_result.semantic_tags or []
                        )
                    elif semantic_result.error:
                        parser_errors.append(semantic_result.error)
                else:
                    semantic_status = "deferred_no_cache"

            ocr_text = ""
            ocr_reopened = False
            if not light:
                ocr_text, ocr_reopened = self._extract_ocr_text(
                    fp_path=fp_path,
                    thumb_path=thumb.thumbnail_path,
                    path=path,
                    render_path=render_path,
                    stats=stats,
                    deep_from_cache=deep_from_cache,
                )
                if ocr_reopened and not original_reopened:
                    original_reopened = True

            if light:
                # Fast Index: yalnızca hash — patch/deep/AI yok.
                self._ensure_extractor(use_ai=False, fast_hash_only=True)
                features = self.extractor.extract_from_path(
                    feature_source,
                    source_path=path,
                    include_patches=False,
                    deep_analysis=False,
                )
            else:
                # RC2 sparse: yalnız eksik Heavy artifact'ları üret.
                from core.artifact_status import (
                    assess_row,
                    features_from_existing,
                    sparse_heavy_plan,
                    vision_compute_set,
                )
                from core.feature_extractor import ExtractedFeatures

                existing_feat = self.db.get_features(file_id) or {}
                report = assess_row(
                    self.db.get_file_by_id(file_id) or existing,
                    existing_feat,
                    require_physical=False,
                )
                missing = sparse_heavy_plan(report)
                stats.setdefault("sparse_missing", [])
                stats.setdefault("sparse_computed", [])
                stats.setdefault("sparse_skipped", [])
                if not missing and report.ai_final:
                    from core.index_ssot import complete_heavy

                    complete_heavy(self.db, file_id)
                    return True
                compute = vision_compute_set(missing)
                skipped = [
                    a
                    for a in ("dino", "clip", "texture", "patch", "semantic", "dna")
                    if a not in missing
                ]
                stats["sparse_missing"] = list(missing)
                stats["sparse_skipped"] = skipped
                stats["sparse_computed"] = list(missing)

                if compute:
                    self._ensure_extractor_for_index()
                    features = self.extractor.extract_from_path(
                        feature_source,
                        source_path=path,
                        include_patches="patch" in compute,
                        deep_analysis=True,
                        compute=compute,
                    )
                else:
                    # Yalnız semantic/DNA — vision extract yok
                    features = ExtractedFeatures()
                    base = features_from_existing(existing_feat)
                    features.phash = str(base.get("phash") or "")
                    features.dhash = str(base.get("dhash") or "")
                    features.whash = str(base.get("whash") or "")
                    features.color_hist = base.get("color_hist")
                    features.dominant_colors = base.get("dominant_colors") or []
                    features.texture_features = base.get("texture_features") or []
                    features.dino_embedding = base.get("dino_embedding")
                    features.clip_embedding = base.get("clip_embedding")
                    features.patch_embeddings_meta = (
                        base.get("patch_embeddings_meta") or []
                    )
                    features.texture_map = base.get("texture_map") or {}

                # Mevcut artifact'ları koru (upsert None ile silmesin)
                if not features.dino_embedding and existing_feat.get("dino_embedding"):
                    features.dino_embedding = existing_feat.get("dino_embedding")
                if not features.clip_embedding and existing_feat.get("clip_embedding"):
                    features.clip_embedding = existing_feat.get("clip_embedding")
                if not features.phash and existing_feat.get("phash"):
                    features.phash = str(existing_feat.get("phash") or "")
                    features.dhash = str(existing_feat.get("dhash") or "")
                    features.whash = str(existing_feat.get("whash") or "")
                if not features.texture_features and existing_feat.get(
                    "texture_features"
                ):
                    features.texture_features = (
                        existing_feat.get("texture_features") or []
                    )
                if not features.patch_embeddings_meta and (
                    existing_feat.get("patch_embeddings_meta")
                    or existing_feat.get("patch_embeddings")
                ):
                    features.patch_embeddings_meta = (
                        existing_feat.get("patch_embeddings_meta")
                        or existing_feat.get("patch_embeddings")
                        or []
                    )
                    if isinstance(features.patch_embeddings_meta, str):
                        import json

                        try:
                            features.patch_embeddings_meta = json.loads(
                                features.patch_embeddings_meta
                            )
                        except Exception:
                            features.patch_embeddings_meta = []
            if self._speed_session and features.phash:
                self._speed_session.hashes += 1

            from core.category_tree import apply_auto_category_to_texture_map
            from core.manual_label_guard import (
                merge_texture_map_preserve_manual,
                parse_texture_map,
            )

            existing_feat = self.db.get_features(file_id) or {}
            # Patch yalnızca Heavy — light'ta üretme / silme.
            if not light:
                from core.artifact_status import sparse_heavy_plan, assess_row

                _miss = sparse_heavy_plan(
                    assess_row(
                        self.db.get_file_by_id(file_id) or existing,
                        existing_feat,
                    )
                )
                if "patch" in _miss:
                    self._ensure_patch_embeddings(
                        features,
                        image_path=str(
                            feature_source or fp_path or thumb.thumbnail_path or ""
                        ),
                        existing_feat=existing_feat,
                    )
                elif existing_feat.get("patch_embeddings_meta") or existing_feat.get(
                    "patch_embeddings"
                ):
                    features.patch_embeddings_meta = (
                        existing_feat.get("patch_embeddings_meta")
                        or existing_feat.get("patch_embeddings")
                        or []
                    )
            elif existing_feat.get("patch_embeddings_meta"):
                features.patch_embeddings_meta = existing_feat.get(
                    "patch_embeddings_meta"
                ) or []
                if existing_feat.get("patch_embeddings"):
                    features.patch_embeddings = existing_feat.get("patch_embeddings")
            existing_tm = parse_texture_map(existing_feat.get("texture_map"))
            merged_texture_map = merge_texture_map_preserve_manual(
                existing_tm,
                features.texture_map,
            )
            merged_texture_map, auto_category_path = apply_auto_category_to_texture_map(
                merged_texture_map,
                filename=filename,
                path=path,
            )
            from core.semantic_tags import build_semantic_tags

            semantic_category = str(
                merged_texture_map.get("manual_category_path")
                or (
                    merged_texture_map.get("category_path")
                    if str(merged_texture_map.get("category_source") or "")
                    in ("manual_user", "gold_dataset")
                    else ""
                )
                or ""
            )
            skip_dna = getattr(self.settings, "index_skip_reanalyse_dna", True)
            skip_sem = getattr(self.settings, "index_skip_reanalyse_semantic", True)
            reanalysed_sem = False
            reanalysed_dna = False

            if not light:
                from core.artifact_status import assess_row, sparse_heavy_plan

                _miss2 = set(
                    sparse_heavy_plan(
                        assess_row(
                            self.db.get_file_by_id(file_id) or existing,
                            existing_feat,
                        )
                    )
                )
                # Sparse: yalnız eksikse üret; mevcutsa koru (skip flags üstüne)
                need_sem = "semantic" in _miss2 or not (
                    skip_sem and texture_map_has_semantic_tags(existing_tm)
                )
                # Eğer sparse plan semantic içermiyorsa kesinlikle yeniden üretme
                if "semantic" not in _miss2 and texture_map_has_semantic_tags(
                    existing_tm
                ):
                    need_sem = False
                if need_sem:
                    reanalysed_sem = True
                    merged_texture_map["semantic_tags"] = build_semantic_tags(
                        merged_texture_map,
                        category_path=semantic_category,
                        ocr_text=ocr_text,
                        filename=filename,
                    )
                    if plugin_semantic_tags:
                        semantic_tags = dict(merged_texture_map["semantic_tags"])
                        semantic_tags["motifs"] = list(
                            dict.fromkeys(
                                [
                                    *list(semantic_tags.get("motifs") or []),
                                    *plugin_semantic_tags,
                                ]
                            )
                        )
                        semantic_tags["source"] = "indexed_visual_metadata+format_plugin"
                        merged_texture_map["semantic_tags"] = semantic_tags
                elif existing_tm.get("semantic_tags"):
                    merged_texture_map["semantic_tags"] = existing_tm["semantic_tags"]

                from core.classification_pipeline import (
                    apply_identity_to_texture_map,
                    resolve_pattern_identity,
                )
                from core.pattern_dna import apply_dna_to_texture_map, build_pattern_dna

                feedback_labels = self.db.get_feedback_labels_for_file(file_id)
                identity = resolve_pattern_identity(
                    texture_map=merged_texture_map,
                    filename=filename,
                    category_path=semantic_category,
                    manual_category_path=str(
                        merged_texture_map.get("manual_category_path") or ""
                    ),
                    feedback_labels=feedback_labels,
                )
                merged_texture_map = apply_identity_to_texture_map(
                    merged_texture_map, identity,
                )
                need_dna = "dna" in _miss2 or not (
                    skip_dna and texture_map_has_pattern_dna(existing_tm)
                )
                if "dna" not in _miss2 and texture_map_has_pattern_dna(existing_tm):
                    need_dna = False
                if need_dna:
                    reanalysed_dna = True
                    dna = build_pattern_dna(
                        merged_texture_map,
                        semantic_tags=merged_texture_map.get("semantic_tags"),
                        category_path=semantic_category,
                        source=identity.source,
                    )
                    if identity.source in ("user_feedback", "gold_dataset", "manual"):
                        dna["source"] = identity.source
                        dna["confidence"] = max(
                            float(dna.get("confidence") or 0), identity.confidence
                        )
                    merged_texture_map = apply_dna_to_texture_map(
                        merged_texture_map, dna
                    )
                elif existing_tm.get("pattern_dna"):
                    merged_texture_map["pattern_dna"] = existing_tm["pattern_dna"]
            else:
                if existing_tm.get("semantic_tags"):
                    merged_texture_map["semantic_tags"] = existing_tm["semantic_tags"]
                if existing_tm.get("pattern_dna"):
                    merged_texture_map["pattern_dna"] = existing_tm["pattern_dna"]
            from core.index_enrichments import enrich_texture_map_if_missing, preserve_derived_fields

            merged_texture_map = preserve_derived_fields(
                merged_texture_map,
                existing_tm,
                reanalysed_semantic=reanalysed_sem,
                reanalysed_dna=reanalysed_dna,
            )
            merged_texture_map = enrich_texture_map_if_missing(
                merged_texture_map,
                ocr_text=ocr_text,
                filename=filename,
                existing=existing_tm,
            )
            merged_texture_map["analysis_source"] = analysis_source
            merged_texture_map["ai_analysis_from_cache"] = ai_analysis_from_cache
            merged_texture_map["original_reopened"] = original_reopened
            from core.category_predictions import category_predictions_from_metadata

            merged_texture_map["ai_category_predictions"] = (
                category_predictions_from_metadata(
                    str(merged_texture_map.get("pattern_family", "") or ""),
                    str(
                        merged_texture_map.get("animal_print_type", "")
                        or merged_texture_map.get("pattern_subtype", "")
                        or ""
                    ),
                    merged_texture_map,
                )
            )

            with self._write_lock:
                feat_row = {
                    "phash": features.phash,
                    "dhash": features.dhash,
                    "whash": features.whash,
                    "color_hist": features.color_hist,
                    "dominant_colors": features.dominant_colors,
                    "texture_features": features.texture_features,
                    "dino_embedding": features.dino_embedding,
                    "clip_embedding": features.clip_embedding,
                    "patch_embeddings_meta": features.patch_embeddings_meta,
                    "texture_map": merged_texture_map,
                }
                if light:
                    # Fast Index: ağır artifact'ları silme
                    if not feat_row.get("dino_embedding"):
                        feat_row["dino_embedding"] = existing_feat.get("dino_embedding")
                    if not feat_row.get("clip_embedding"):
                        feat_row["clip_embedding"] = existing_feat.get("clip_embedding")
                    if not feat_row.get("patch_embeddings_meta"):
                        feat_row["patch_embeddings_meta"] = (
                            existing_feat.get("patch_embeddings_meta") or []
                        )
                    if not feat_row.get("texture_features"):
                        feat_row["texture_features"] = (
                            existing_feat.get("texture_features") or []
                        )
                    if not feat_row.get("color_hist"):
                        feat_row["color_hist"] = existing_feat.get("color_hist")
                    if not feat_row.get("dominant_colors"):
                        feat_row["dominant_colors"] = (
                            existing_feat.get("dominant_colors") or []
                        )
                    # texture_map: mevcut heavy alanlarını koru
                    if existing_tm:
                        feat_row["texture_map"] = merge_texture_map_preserve_manual(
                            existing_tm,
                            merged_texture_map if merged_texture_map else {},
                        )
                else:
                    # RC2 sparse: yeniden üretilmeyen artifact'ları koru
                    if not feat_row.get("dino_embedding"):
                        feat_row["dino_embedding"] = existing_feat.get("dino_embedding")
                    if not feat_row.get("clip_embedding"):
                        feat_row["clip_embedding"] = existing_feat.get("clip_embedding")
                    if not feat_row.get("patch_embeddings_meta"):
                        feat_row["patch_embeddings_meta"] = (
                            existing_feat.get("patch_embeddings_meta") or []
                        )
                    if not feat_row.get("texture_features"):
                        feat_row["texture_features"] = (
                            existing_feat.get("texture_features") or []
                        )
                    if not feat_row.get("phash"):
                        feat_row["phash"] = existing_feat.get("phash") or ""
                        feat_row["dhash"] = existing_feat.get("dhash") or ""
                        feat_row["whash"] = existing_feat.get("whash") or ""
                self.db.upsert_features(file_id, feat_row)

                # Duplicate / variant işaretleme (silme yok) — yalnızca eksikse
                if (
                    not light
                    and (features.phash or True)
                    and not merged_texture_map.get("duplicate_info")
                ):
                    try:
                        from core.duplicate_detection import apply_duplicate_marks

                        partial_for_dup = str(
                            (self.db.get_file_by_id(file_id) or {}).get("partial_hash")
                            or ""
                        )
                        marked = apply_duplicate_marks(
                            self.db,
                            file_id=file_id,
                            features={
                                "phash": features.phash,
                                "dhash": features.dhash,
                                "whash": features.whash,
                            },
                            texture_map=merged_texture_map,
                            partial_hash=partial_for_dup,
                        )
                        if marked is not merged_texture_map and marked.get("duplicate_info"):
                            merged_texture_map = marked
                            self.db.upsert_texture_map(file_id, merged_texture_map)
                            self._bump_stat(stats, "duplicates_marked", 1)
                    except Exception:
                        pass

                if features.dino_embedding:
                    self.faiss.add_dino(file_id, features.dino_embedding)
                    self._maybe_save_faiss()
                if features.clip_embedding:
                    self.faiss.add_clip(file_id, features.clip_embedding)
                    self._maybe_save_faiss()

            partial = ""
            fast_mode = (
                getattr(self.settings, "index_mode", "")
                == IndexMode.FAST_ARCHIVE.value
            )
            if deep_scan and light and not fast_mode:
                with network_original_open_slot(
                    self.settings, path, stats, reason="partial_file_hash"
                ):
                    partial = self._compute_partial_hash(path, stats)
            elif deep_scan:
                partial = str(
                    (self.db.get_file_by_id(file_id) or current).get("partial_hash") or ""
                )

            if light and (
                not fp_path
                or not thumb.thumbnail_path
                or not local_artifact_exists(fp_path, stats=stats)
                or not local_artifact_exists(thumb.thumbnail_path, stats=stats)
            ):
                raise RuntimeError(
                    "light_preview_contract_violation:"
                    "thumbnail_and_medium_preview_required"
                )

            now = datetime.now(timezone.utc).isoformat()
            if not fp_path:
                existing = self.db.get_file_by_id(file_id) or existing
                fp_path = existing.get("feature_preview_path", "")
                fp_version = int(existing.get("feature_preview_version") or 0)
            fp_mtime = float(existing.get("feature_preview_mtime") or 0)
            if fp_path:
                fp_mtime = float(
                    (self.db.get_file_by_id(file_id) or {}).get("mtime") or 0
                )
            with self._write_lock:
                self.db.upsert_file(
                    {
                        "path": path,
                        "filename": filename,
                        "thumbnail_path": thumb.thumbnail_path,
                        "feature_preview_path": fp_path,
                        "feature_preview_version": fp_version,
                        "feature_preview_mtime": fp_mtime,
                        "unsupported_preview": unsupported,
                        "preview_status": preview_status,
                        "thumbnail_status": thumbnail_status,
                        "metadata_status": metadata_status,
                        "semantic_status": semantic_status,
                        "format_metadata": metadata_to_json(plugin_metadata),
                        "parser_error": "; ".join(dict.fromkeys(parser_errors)),
                        **format_record,
                        "width": thumb.width,
                        "height": thumb.height,
                        "full_hash": full_hash,
                        "partial_hash": partial,
                        "ocr_text": ocr_text,
                        "category_path": (
                            str(existing.get("manual_category_path") or "").strip()
                            or auto_category_path
                            or str(existing.get("category_path") or "")
                        ),
                        "status": "indexed",
                        "indexed_at": now,
                        "feature_version": FEATURE_VERSION,
                        "texture_version": TEXTURE_MAP_VERSION,
                        "thumbnail_version": THUMBNAIL_VERSION,
                        "last_feature_at": now,
                        "last_texture_at": now,
                        "error_msg": "",
                        "quarantine_reason": "",
                        "is_searchable_visual": 1 if (fp_path and thumb.thumbnail_path) else 0,
                    }
                )
            self.db.update_physical_readiness(
                file_id,
                thumbnail_ready=local_artifact_exists(
                    str(thumb.thumbnail_path or ""), stats=stats
                ),
                preview_ready=local_artifact_exists(str(fp_path or ""), stats=stats),
                requeue_missing=True,
            )
            self._update_text_index(
                file_id, filename, path, ocr_text, merged_texture_map
            )
            if light:
                from core.index_ssot import complete_light

                complete_light(self.db, file_id)
                # Genel AI kuyruğu = DB heavy_status pending (index_queue yok)
            else:
                from core.index_ssot import complete_heavy

                if not complete_heavy(self.db, file_id):
                    self.db.set_file_index_stage(file_id, LIGHT_DONE)
                    logger.info(
                        "AI Final eksik; heavy yeniden kuyruğa alındı file_id=%s",
                        file_id,
                    )
                    return True
                self.db.set_file_index_stage(file_id, FULL_DONE)
            from core.index_integrity import finalize_file_status

            finalize_file_status(self.db, file_id, settings=self.settings)
            self._refresh_project_candidate(file_id)
            self._record_pipeline_saves(
                stats,
                light=light,
                existing=current,
                existing_feat=existing_feat,
                existing_tm=existing_tm,
                features=features,
                merged_texture_map=merged_texture_map,
                ocr_text=ocr_text,
            )
            return True
        except Exception as exc:
            with self._stats_lock:
                stats["errors"] += 1
            if self._speed_session:
                self._speed_session.errors += 1
            msg = str(exc)
            errors = stats.setdefault("first_50_errors", [])
            if len(errors) < 50:
                errors.append({"path": path, "stage": "index", "error": msg})
            log_error_file(path, msg, stage="index")
            current_rec = self.db.get_file_by_id(file_id) or {}
            reason = classify_quarantine_reason(msg)
            if not light and has_preview_artifacts(current_rec):
                with self._write_lock:
                    self.db.upsert_file(
                        {
                            "path": path,
                            "filename": filename,
                            "status": "indexed",
                            "quarantine_reason": reason,
                            "needs_review": 1,
                            "error_msg": msg,
                            "is_searchable_visual": 1,
                        }
                    )
                    self.db.set_file_index_stage(file_id, LIGHT_DONE, needs_review=1)
                self.db.mark_heavy_failed(
                    file_id, reason=reason or "ai_analysis_error", error_msg=msg
                )
                logger.warning(
                    "Derin analiz hatası (aranabilir kalıyor) %s: %s", path, exc
                )
                return True
            with self._write_lock:
                self.db.upsert_file(
                    {
                        "path": path,
                        "filename": filename,
                        "status": "error",
                        "error_msg": msg,
                        "quarantine_reason": reason,
                        "needs_review": 1 if is_hard_quarantine(reason) else 0,
                    }
                )
                self.db.set_file_index_stage(file_id, FAILED, needs_review=1)
            if light:
                self.db.mark_light_failed(
                    file_id, reason=reason or "other", error_msg=msg
                )
            else:
                self.db.mark_heavy_failed(
                    file_id, reason=reason or "other", error_msg=msg
                )
            logger.error("Index hatası %s: %s", path, exc)
            return False
        finally:
            if deep_scope_bound:
                unbind_deep_from_cache()

    def _ensure_light_persisted(self, file_id: int, *, stage: str | None = None) -> None:
        """SSOT: light başarı → light_status='done'."""
        from core.index_ssot import complete_light

        complete_light(self.db, file_id)
        if stage:
            self.db.set_file_index_stage(file_id, stage)

    def _process_heavy_format_skip(
        self,
        file_id: int,
        path: str,
        filename: str,
        stats: dict,
    ) -> bool:
        """PSD/PDF/CDR — hızlı modda render atla, gece kuyruğuna al."""
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            self.db.upsert_file(
                {
                    "path": path,
                    "filename": filename,
                    "status": "indexed",
                    "preview_status": "deferred_heavy",
                    "unsupported_preview": 0,
                    "indexed_at": now,
                    "error_msg": "",
                }
            )
            self.db.set_file_index_stage(file_id, SKIPPED_HEAVY_FORMAT)
        self._ensure_light_persisted(file_id, stage=SKIPPED_HEAVY_FORMAT)
        self._update_text_index(file_id, filename, path, "", {})
        self.db.mark_queue_tasks_done_for_file(file_id)
        from core.index_integrity import finalize_file_status

        finalize_file_status(self.db, file_id, settings=self.settings)
        if self._speed_session:
            self._speed_session.skipped_heavy_format += 1
        return True

    def _needs_texture_update(self, existing: dict[str, Any]) -> bool:
        tv = int(existing.get("texture_version") or 0)
        if tv < TEXTURE_MAP_VERSION:
            return True
        feat = self.db.get_features(existing["id"])
        if not feat or not feat.get("texture_map"):
            return True
        return False

    def _needs_feature_backfill(self, existing: dict[str, Any] | None) -> bool:
        """Rebuild only rows with missing production-search artifacts."""
        if not existing or existing.get("status") != "indexed":
            return False
        if int(existing.get("feature_version") or 0) < FEATURE_VERSION:
            return True
        if not existing.get("thumbnail_path") or not local_artifact_exists(
            str(existing.get("thumbnail_path") or "")
        ):
            return True
        light_ok = getattr(self.settings, "index_light_first", True)
        if self._needs_feature_preview_update(existing) and not (
            light_ok and getattr(self.settings, "index_defer_preview", True)
        ):
            return True
        feat = self.db.get_features(existing["id"])
        if not feat or not feat.get("phash") or not feat.get("texture_features"):
            return True
        # Patch arama sıralaması için zorunlu — light_first olsa bile eksikse backfill.
        if not feat.get("patch_embeddings_meta"):
            return True
        if self.settings.index_uses_ai() and (
            not feat.get("dino_embedding") or not feat.get("clip_embedding")
        ):
            return True
        if self.settings.ocr_enabled and not (existing.get("ocr_text") or "").strip():
            return True
        return False

    def _needs_full_backfill(self, existing: dict[str, Any] | None) -> bool:
        """Eksikleri tamamla — AI/OCR/patch/önizleme gerekiyor mu?"""
        if not existing:
            return False
        if self._should_skip_quarantine(existing):
            return False
        if self._needs_feature_preview_update(existing) and not (
            getattr(self.settings, "index_defer_preview", True)
            and getattr(self.settings, "index_light_first", True)
        ):
            return True
        if needs_medium_preview(existing, feature_preview_cache=self.feature_preview):
            return True
        feat = self.db.get_features(existing["id"])
        if not feat:
            return False
        from core.manual_label_guard import parse_texture_map

        tm = parse_texture_map(feat.get("texture_map"))
        if getattr(self.settings, "index_skip_reanalyse_dna", True) and not texture_map_has_pattern_dna(tm):
            return True
        if getattr(self.settings, "index_skip_reanalyse_semantic", True) and not texture_map_has_semantic_tags(tm):
            return True
        if not feat.get("phash") or not feat.get("texture_features"):
            return True
        if not feat.get("patch_embeddings_meta") and self.settings.ai_embedding_enabled:
            return True
        if self.settings.index_uses_ai() and (
            not feat.get("dino_embedding") or not feat.get("clip_embedding")
        ):
            return True
        if self.settings.ocr_enabled and not (existing.get("ocr_text") or "").strip():
            return True
        return False

    def _ensure_patch_embeddings(
        self,
        features: Any,
        *,
        image_path: str,
        existing_feat: dict[str, Any] | None = None,
    ) -> bool:
        """Patch meta yoksa thumbnail/feature_preview'dan üret; mevcutları koru.

        Orijinal TIFF açmaz — büyük dosyalarda crash/defer riskini önler.
        """
        existing_feat = existing_feat or {}
        existing_meta = existing_feat.get("patch_embeddings_meta") or []
        if not isinstance(existing_meta, list):
            existing_meta = []

        current = getattr(features, "patch_embeddings_meta", None) or []
        if isinstance(current, list) and current:
            return True
        if existing_meta:
            features.patch_embeddings_meta = existing_meta
            return True

        src = str(image_path or "").strip()
        if not src or not local_artifact_exists(src):
            return False

        try:
            # Multiscale patch meta AI gerektirmez; DINO patch opsiyonel.
            self._ensure_extractor(
                use_ai=bool(self.settings.index_uses_ai()),
                fast_hash_only=False,
            )
            filled = self.extractor.extract_from_path(
                src,
                source_path=src,
                include_patches=True,
                deep_analysis=True,
            )
            meta = list(filled.patch_embeddings_meta or [])
            if not meta:
                return False
            features.patch_embeddings_meta = meta
            if getattr(filled, "patch_embeddings", None):
                features.patch_embeddings = list(filled.patch_embeddings)
            return True
        except Exception as exc:
            logger.warning("Patch embedding üretilemedi (%s): %s", src, exc)
            return False

    def backfill_patches_for_file(self, file_id: int) -> dict[str, Any]:
        """Tek dosya için patch backfill (thumbnail/feature_preview üzerinden)."""
        rec = self.db.get_file_by_id(int(file_id)) or {}
        if not rec:
            return {"ok": False, "reason": "file_not_found", "file_id": file_id}
        feat = self.db.get_features(int(file_id)) or {}
        existing_meta = feat.get("patch_embeddings_meta") or []
        if isinstance(existing_meta, list) and existing_meta:
            return {
                "ok": True,
                "file_id": file_id,
                "already_had_patches": True,
                "patch_count": len(existing_meta),
            }

        candidates = [
            str(rec.get("feature_preview_path") or ""),
            str(rec.get("thumbnail_path") or ""),
        ]
        image_path = next((p for p in candidates if p and local_artifact_exists(p)), "")
        if not image_path:
            return {"ok": False, "reason": "no_preview_or_thumbnail", "file_id": file_id}

        from core.feature_extractor import ExtractedFeatures

        features = ExtractedFeatures(
            phash=str(feat.get("phash") or ""),
            dhash=str(feat.get("dhash") or ""),
            whash=str(feat.get("whash") or ""),
            color_hist=feat.get("color_hist"),
            dominant_colors=feat.get("dominant_colors") or [],
            texture_features=feat.get("texture_features") or [],
            dino_embedding=feat.get("dino_embedding") or b"",
            clip_embedding=feat.get("clip_embedding") or b"",
            texture_map=feat.get("texture_map")
            if isinstance(feat.get("texture_map"), dict)
            else {},
        )
        ok = self._ensure_patch_embeddings(
            features, image_path=image_path, existing_feat=feat
        )
        if not ok:
            return {"ok": False, "reason": "patch_generate_failed", "file_id": file_id}

        # Mevcut texture_map / embedding'leri koru
        tm = feat.get("texture_map")
        if isinstance(tm, str):
            try:
                import json as _json

                tm = _json.loads(tm) if tm else {}
            except Exception:
                tm = {}
        with self._write_lock:
            self.db.upsert_features(
                int(file_id),
                {
                    "phash": features.phash or feat.get("phash") or "",
                    "dhash": features.dhash or feat.get("dhash") or "",
                    "whash": features.whash or feat.get("whash") or "",
                    "color_hist": features.color_hist
                    if features.color_hist is not None
                    else feat.get("color_hist"),
                    "dominant_colors": features.dominant_colors
                    or feat.get("dominant_colors")
                    or [],
                    "texture_features": features.texture_features
                    or feat.get("texture_features")
                    or [],
                    "dino_embedding": features.dino_embedding
                    or feat.get("dino_embedding"),
                    "clip_embedding": features.clip_embedding
                    or feat.get("clip_embedding"),
                    "patch_embeddings_meta": features.patch_embeddings_meta,
                    "texture_map": tm or {},
                },
            )
        return {
            "ok": True,
            "file_id": file_id,
            "patch_count": len(features.patch_embeddings_meta or []),
            "source": image_path,
        }

    def _process_texture_only(
        self,
        file_id: int,
        thumb_path: str,
        filename: str = "",
        path: str = "",
    ) -> bool:
        if not thumb_path or not local_artifact_exists(thumb_path):
            return False
        from core.manual_label_guard import is_manual_labeled, parse_texture_map

        existing_feat = self.db.get_features(file_id) or {}
        if is_manual_labeled(parse_texture_map(existing_feat.get("texture_map"))):
            return True
        try:
            image = Thumbnailer.load_image(thumb_path)
            if image is None:
                return False
            feat = self.db.get_features(file_id) or {}
            profile = TextureAnalyzer.analyze(
                image,
                filename=filename,
                path=path,
                patch_metas=feat.get("patch_embeddings_meta", []),
                dominant_colors=feat.get("dominant_colors", []),
            )
            self.db.upsert_texture_map(file_id, profile.to_dict())
            now = datetime.now(timezone.utc).isoformat()
            self.db.upsert_file(
                {
                    "path": path or thumb_path,
                    "filename": filename,
                    "texture_version": TEXTURE_MAP_VERSION,
                    "last_texture_at": now,
                }
            )
            return True
        except Exception as exc:
            logger.warning("Texture-only güncelleme hatası %s: %s", path, exc)
            return False

    def _update_text_index(
        self,
        file_id: int,
        filename: str,
        path: str,
        ocr_text: str,
        texture_map: dict,
    ) -> None:
        from core.text_index import build_text_search_blob, extract_index_fields

        rec = self.db.get_file_by_id(file_id) or {}
        source_id = int(rec.get("source_id") or 0)
        source_name = ""
        if source_id:
            src = self.db.get_source(source_id)
            source_name = (src or {}).get("name", "")
        group = self.db.get_pattern_group_for_file(file_id)
        group_label = (group or {}).get("label", "")
        feedback_labels = self.db.get_feedback_labels_for_file(file_id)
        custom_tags = self.db.get_feedback_custom_tags_for_file(file_id)
        pf, pt, tf = extract_index_fields(texture_map)
        prof = TextureProfile.from_dict(texture_map)
        subtype = prof.pattern_subtype or pt
        confidence = float(prof.classification_confidence or 0)
        blob = build_text_search_blob(
            filename=filename,
            path=path,
            customer=rec.get("customer", ""),
            source_name=source_name,
            ocr_text=ocr_text,
            texture_map=texture_map,
            feedback_labels=[*feedback_labels, *custom_tags],
            group_label=group_label,
            pattern_family=pf,
            pattern_type=pt,
            texture_family=tf,
            pattern_subtype=subtype,
            category_path=str(
                rec.get("manual_category_path") or rec.get("category_path") or ""
            ),
            category_aliases=list(texture_map.get("category_aliases") or []),
            semantic_enabled=bool(
                getattr(self.settings, "semantic_text_search_enabled", True)
            ),
        )
        self.db.update_file_text_index(
            file_id,
            pattern_family=pf,
            pattern_type=pt,
            texture_family=tf,
            pattern_subtype=subtype,
            pattern_confidence=confidence,
            text_search_blob=blob,
        )

    def quick_index_folder(
        self,
        folder_path: str,
        progress_callback: Callable[[dict], None] | None = None,
        fast_only: bool = False,
    ) -> dict:
        """Seçili klasörü hızlı indexle — anlık arama için."""
        from core.index_freeze import search_write_protection_active
        from core.utils import normalize_path

        # B only — INDEX_FROZEN (A) must not skip folder indexing forever.
        if search_write_protection_active():
            return {
                "skipped": True,
                "reason": "search_session",
                "processed": 0,
                "errors": 0,
            }

        folder = normalize_path(folder_path)
        if not os.path.isdir(folder):
            logger.warning("Klasör erişilemiyor: %s", folder)
            return {"errors": 1, "processed": 0}

        existing = self.db.get_source_by_path(folder)
        if existing:
            source = existing
        else:
            sid = self.source_manager.add_source(
                name=os.path.basename(folder.rstrip("\\/")) or "Hızlı Arama",
                root_path=folder,
                source_type="selected_folder",
            )
            source = self.db.get_source(sid) or {
                "id": sid,
                "root_path": folder,
                "name": folder,
            }

        prev_fast = self.settings.fast_hash_only
        prev_ai = self.settings.ai_embedding_enabled
        try:
            if fast_only:
                self.settings.fast_hash_only = True
                self.settings.ai_embedding_enabled = False
                self._ensure_extractor(use_ai=False, fast_hash_only=True)
            return self.run_source(
                source=source,
                scan_mode=ScanMode.QUICK.value,
                progress_callback=progress_callback,
            )
        finally:
            self.settings.fast_hash_only = prev_fast
            self.settings.ai_embedding_enabled = prev_ai
            if not fast_only:
                self._ensure_extractor(use_ai=False, fast_hash_only=prev_fast)

    def needs_search_features(self, record: dict[str, Any] | None) -> bool:
        """Arama için gerekli özellikler eksik mi?"""
        if not record:
            return True
        if record.get("status") != "indexed":
            return True
        return self._needs_feature_backfill(record)

    def index_paths(
        self,
        paths: list[str],
        *,
        deep: bool = False,
    ) -> dict[str, Any]:
        """Verilen dosya yollarını tam özellikle indexle."""
        stats: dict[str, Any] = {"processed": 0, "errors": 0, "paths": len(paths)}
        from core.index_freeze import INDEX_FROZEN, process_search_active

        if INDEX_FROZEN and process_search_active():
            stats["skipped"] = True
            stats["reason"] = "search_session"
            return stats
        if not paths:
            return stats
        self._ensure_extractor()
        for path in paths:
            if self._stop.is_set():
                break
            norm = normalize_path(path)
            if not norm or not os.path.isfile(norm):
                continue
            filename = os.path.basename(norm)
            existing = self.db.get_file_by_path(norm)
            file_id = (
                int(existing["id"])
                if existing
                else self.db.upsert_file(
                    {
                        "path": norm,
                        "filename": filename,
                        "status": "processing",
                        "source_id": existing.get("source_id") if existing else 0,
                    }
                )
            )
            if self._process_file(file_id, norm, filename, stats, deep_scan=deep):
                stats["processed"] += 1
        if stats["processed"]:
            self.faiss.save()
        return stats

    def get_status(self) -> dict:
        counts = self.db.count_by_status()
        total = self.db.total_searchable_files()
        indexed = counts.get("indexed", 0)
        errors = counts.get("error", 0)
        pending = (
            counts.get("pending", 0)
            + counts.get("processing", 0)
            + counts.get("changed", 0)
        )
        sources = self.db.list_sources()
        embedding_counts = self.db.count_embeddings()
        faiss_consistency = self.faiss.consistency(embedding_counts)
        gaps = self.db.feature_gap_counts(self.settings.ai_embedding_enabled)
        return {
            "total": total,
            "indexed": indexed,
            "errors": errors,
            "pending": pending,
            "missing_texture_maps": self.db.count_missing_texture_maps(),
            "missing_feature_previews": self.db.count_files_missing_feature_preview(),
            "missing_thumbnails": gaps["thumbnail"],
            "missing_patches": gaps["patch"],
            "missing_structure": gaps["structure"],
            "missing_ai_dino": gaps["ai_dino"],
            "missing_ai_clip": gaps["ai_clip"],
            "missing_files": self.db.count_missing_files(),
            "counts": counts,
            "faiss_dino": self.faiss.dino_count,
            "faiss_clip": self.faiss.clip_count,
            "db_dino_embeddings": embedding_counts["dino"],
            "db_clip_embeddings": embedding_counts["clip"],
            "faiss_consistent": faiss_consistency["consistent"],
            "ai_available": self.extractor.ai_available,
            "sources": sources,
            "source_count": len(sources),
        }
