"""DB ile preview/thumbnail cache fiziksel durumunu uzlaştırır."""

from __future__ import annotations

import json
import multiprocessing
import threading
from collections import defaultdict
from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from core.db import Database
from core.index_analysis_guards import local_artifact_exists
from core.light_asset_extractor import extract_light_assets_once
from core.logger import setup_logger
from core.preview_cache import FeaturePreviewCache
from core.settings import AppSettings
from core.thumbnailer import Thumbnailer

logger = setup_logger(__name__)


def classify_light_repair_failure(path: str, error: str) -> tuple[str, bool]:
    """Gerçek repair nedeni ve otomatik retry uygunluğu."""
    text = str(error or "").lower()
    source = str(path or "")
    ext = Path(source).suffix.lower()
    is_network = source.startswith("\\\\")
    if not source:
        return "invalid_path", False
    if "decompression bomb" in text or "exceeds limit" in text:
        return "decompression_limit", False
    if "too large" in text or "file_too_large" in text:
        return "too_large", False
    if "timeout" in text or "timed out" in text:
        return ("nas_timeout" if is_network else "io_timeout"), True
    if (
        "no such file" in text
        or "not found" in text
    ):
        return "source_unavailable", is_network
    if "permission" in text or "access is denied" in text:
        return "permission_error", True
    if "unsupported" in text or "desteklenm" in text:
        return "unsupported", False
    if "renderer" in text or "ghostscript" in text or "dependency" in text:
        return "renderer_error", False
    if (
        "cannot identify image" in text
        or "corrupt" in text
        or "truncated" in text
        or "seek outside sequence" in text
    ):
        return ("corrupt_tiff" if ext in (".tif", ".tiff") else "corrupt_image"), False
    return "unknown", True


def _isolated_light_repair_worker(
    settings_data: dict[str, Any],
    row: dict[str, Any],
    result_queue: Any,
) -> None:
    """Riskli decoder işi child process'te; parent hard-timeout uygulayabilir."""
    try:
        allowed = {field.name for field in fields(AppSettings)}
        settings = AppSettings(
            **{key: value for key, value in settings_data.items() if key in allowed}
        )
        engine = CacheReconciliationEngine(
            settings,
            repair_ai=False,
            isolate_risky_repairs=False,
        )
        result_queue.put(engine._repair_row_direct(row))
    except BaseException as exc:
        result_queue.put((False, False, str(exc)))


class CacheReconciliationEngine:
    """Küçük batch'lerle doğrular, stale kayıtları onarır veya SSOT'ye alır."""

    CURSOR_META = "cache_reconcile_cursor"
    AI_CURSOR_META = "ai_integrity_reconcile_cursor"
    LAST_RUN_META = "cache_reconcile_last_at"
    LAST_STATS_META = "cache_reconcile_last_stats"

    def __init__(
        self,
        settings: AppSettings,
        *,
        batch_size: int = 100,
        reverify_hours: int = 24,
        repair_ai: bool = True,
        repair_timeout_sec: float | None = None,
        max_repair_retries: int = 2,
        isolate_risky_repairs: bool = True,
    ):
        self.settings = settings
        self.db = Database(settings.db_path)
        self.batch_size = max(1, int(batch_size))
        self.reverify_hours = max(1, int(reverify_hours))
        self.repair_ai = bool(repair_ai)
        configured_timeout = float(getattr(settings, "network_timeout_sec", 30) or 30)
        self.repair_timeout_sec = max(
            1.0,
            float(repair_timeout_sec or configured_timeout),
        )
        self.max_repair_retries = max(1, int(max_repair_retries))
        self.isolate_risky_repairs = bool(isolate_risky_repairs)
        self.thumbnailer = Thumbnailer(
            settings.cache_dir,
            max_edge=settings.thumbnail_max_edge,
            fmt=settings.thumbnail_format,
        )
        self.feature_preview = FeaturePreviewCache(
            settings.cache_dir,
            settings.feature_preview_max_edge,
            settings.thumbnail_format,
        )
        self._stop = threading.Event()
        self._repair_paused = threading.Event()
        self._ai_indexer = None

    def request_stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        """Index aktifken yalnız onarımı duraklat; verify devam eder."""
        self._repair_paused.set()

    def resume(self) -> None:
        self._repair_paused.clear()

    def is_stopped(self) -> bool:
        return self._stop.is_set()

    def _wait_if_paused(self) -> bool:
        return not self._stop.is_set()

    def _cursor(self) -> int:
        try:
            return max(0, int(self.db.get_meta(self.CURSOR_META, "0") or 0))
        except (TypeError, ValueError):
            return 0

    def _ai_cursor(self) -> int:
        try:
            return max(0, int(self.db.get_meta(self.AI_CURSOR_META, "0") or 0))
        except (TypeError, ValueError):
            return 0

    def _verified_before(self) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.reverify_hours)
        return cutoff.isoformat()

    def _verify_row(self, row: dict[str, Any]) -> tuple[bool, bool]:
        return (
            local_artifact_exists(str(row.get("thumbnail_path") or "")),
            local_artifact_exists(str(row.get("feature_preview_path") or "")),
        )

    @staticmethod
    def _needs_isolated_repair(path: str) -> bool:
        ext = Path(str(path or "")).suffix.lower()
        return str(path or "").startswith("\\\\") or ext in {
            ".tif",
            ".tiff",
            ".psd",
            ".pdf",
            ".ai",
            ".svg",
        }

    def _repair_row_direct(self, row: dict[str, Any]) -> tuple[bool, bool, str]:
        path = str(row.get("path") or "")
        if not path:
            return False, False, "source_path_empty"
        try:
            result = extract_light_assets_once(
                path,
                thumbnailer=self.thumbnailer,
                feature_preview=self.feature_preview,
            )
        except Exception as exc:
            return False, False, str(exc)
        if not result.success:
            return False, False, str(result.error or "preview_repair_failed")
        self.db.upsert_file(
            {
                "path": path,
                "filename": str(row.get("filename") or Path(path).name),
                "thumbnail_path": str(result.thumbnail_path or ""),
                "feature_preview_path": str(result.feature_preview_path or ""),
                "thumbnail_status": str(
                    result.thumbnail_status or "thumbnail_ok"
                ),
                "preview_status": "preview_ok",
                "width": int(result.thumbnail_width or 0),
                "height": int(result.thumbnail_height or 0),
                "error_msg": "",
            }
        )
        thumb_ready = local_artifact_exists(str(result.thumbnail_path or ""))
        preview_ready = local_artifact_exists(
            str(result.feature_preview_path or "")
        )
        return thumb_ready, preview_ready, ""

    def _repair_row(self, row: dict[str, Any]) -> tuple[bool, bool, str]:
        path = str(row.get("path") or "")
        if not self.isolate_risky_repairs or not self._needs_isolated_repair(path):
            return self._repair_row_direct(row)

        thumb_path = self.thumbnailer.thumbnail_path_for(path)
        preview_path = self.feature_preview.preview_path_for(path)
        thumb_existed = thumb_path.exists()
        preview_existed = preview_path.exists()
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_isolated_light_repair_worker,
            args=(asdict(self.settings), dict(row), result_queue),
            daemon=True,
        )
        try:
            process.start()
            process.join(self.repair_timeout_sec)
            if process.is_alive():
                process.terminate()
                process.join(2.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(2.0)
                if not thumb_existed:
                    thumb_path.unlink(missing_ok=True)
                if not preview_existed:
                    preview_path.unlink(missing_ok=True)
                return False, False, f"repair timeout {self.repair_timeout_sec:.0f}s"
            try:
                result = result_queue.get(timeout=0.5)
            except Empty:
                return False, False, f"repair subprocess exit={process.exitcode}"
            return bool(result[0]), bool(result[1]), str(result[2] or "")
        except Exception as exc:
            return False, False, f"repair subprocess error: {exc}"
        finally:
            if process.is_alive():
                process.terminate()
                process.join(2.0)
            result_queue.close()
            result_queue.join_thread()

    def _repair_ai_row(self, row: dict[str, Any]) -> bool:
        if self._stop.is_set() or self._repair_paused.is_set():
            return False
        try:
            if self._ai_indexer is None:
                from core.indexer import Indexer

                self._ai_indexer = Indexer(self.settings)
            stats: dict[str, Any] = defaultdict(int)
            ok = self._ai_indexer._process_file(
                int(row["id"]),
                str(row.get("path") or ""),
                str(row.get("filename") or ""),
                stats,
                deep_scan=True,
                light_pass=False,
            )
            return bool(ok and self.db.is_ai_final_ready(int(row["id"])))
        except Exception:
            logger.exception("Background AI repair failed file_id=%s", row["id"])
            return False

    def run_batch(
        self,
        *,
        repair_missing: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bir batch çalıştırır; UI thread'inden çağrılmamalıdır."""
        if not self._wait_if_paused():
            return {"stopped": True}
        repairs_allowed = repair_missing and not self._repair_paused.is_set()
        cursor = self._cursor()
        rows = self.db.list_physical_verification_batch(
            after_id=cursor,
            limit=self.batch_size,
            verified_before=self._verified_before(),
        )
        wrapped = False
        if not rows and cursor > 0:
            cursor = 0
            self.db.set_meta(self.CURSOR_META, "0")
            rows = self.db.list_physical_verification_batch(
                after_id=0,
                limit=self.batch_size,
                verified_before=self._verified_before(),
            )
            wrapped = True

        stats: dict[str, Any] = {
            "checked": 0,
            "ready": 0,
            "missing": 0,
            "repaired": 0,
            "repair_failed": 0,
            "ai_requeued": 0,
            "ai_repaired": 0,
            "ai_missing": 0,
            "ai_integrity_candidates": 0,
            "ai_missing_dino": 0,
            "ai_missing_clip": 0,
            "ai_missing_texture": 0,
            "ai_missing_semantic": 0,
            "ai_missing_dna": 0,
            "ai_missing_patch": 0,
            "ai_cursor": self._ai_cursor(),
            "ai_wrapped": False,
            "locked": 0,
            "cursor": cursor,
            "wrapped": wrapped,
            "stopped": False,
            "repair_paused": not repairs_allowed and repair_missing,
        }
        for row in rows:
            if not self._wait_if_paused():
                stats["stopped"] = True
                break
            worker = "cache_reconcile"
            if not self.db.try_acquire_processing_lock(int(row["id"]), worker):
                stats["locked"] += 1
                stats["cursor"] = int(row["id"])
                self.db.set_meta(self.CURSOR_META, str(row["id"]))
                continue
            try:
                thumb_ready, preview_ready = self._verify_row(row)
                if repairs_allowed and not (thumb_ready and preview_ready):
                    if int(row.get("repair_retryable", 1) or 0) == 0:
                        error = str(row.get("repair_last_error") or "repair_blocked")
                        stats["repair_blocked"] = int(
                            stats.get("repair_blocked", 0)
                        ) + 1
                    else:
                        thumb_ready, preview_ready, error = self._repair_row(row)
                        if thumb_ready and preview_ready:
                            self.db.clear_light_repair_failure(int(row["id"]))
                            stats["repaired"] += 1
                        else:
                            reason, retryable = classify_light_repair_failure(
                                str(row.get("path") or ""),
                                error,
                            )
                            retry = self.db.record_light_repair_failure(
                                int(row["id"]),
                                reason=reason,
                                error_msg=error,
                                retryable=retryable,
                                max_retries=self.max_repair_retries,
                            )
                            stats["repair_failed"] += 1
                            stats[f"repair_failed_{reason}"] = int(
                                stats.get(f"repair_failed_{reason}", 0)
                            ) + 1
                            if not retry["retryable"]:
                                stats["repair_quarantined"] = int(
                                    stats.get("repair_quarantined", 0)
                                ) + 1
                            logger.info(
                                "cache_repair_failed file_id=%s reason=%s "
                                "retry=%s/%s error=%s",
                                row["id"],
                                reason,
                                retry["retry_count"],
                                self.max_repair_retries,
                                error,
                            )
                self.db.update_physical_readiness(
                    int(row["id"]),
                    thumbnail_ready=thumb_ready,
                    preview_ready=preview_ready,
                    requeue_missing=True,
                )
            finally:
                self.db.release_processing_lock(int(row["id"]), worker)
            stats["checked"] += 1
            stats["ready" if thumb_ready and preview_ready else "missing"] += 1
            stats["cursor"] = int(row["id"])
            self.db.set_meta(self.CURSOR_META, str(row["id"]))
            if progress_callback and (
                stats["checked"] % 10 == 0 or row is rows[-1]
            ):
                progress_callback(dict(stats))

        ai_cursor = self._ai_cursor()
        ai_rows = self.db.list_ai_integrity_candidates(
            after_id=ai_cursor,
            limit=self.batch_size,
        )
        if not ai_rows and ai_cursor > 0:
            ai_cursor = 0
            self.db.set_meta(self.AI_CURSOR_META, "0")
            ai_rows = self.db.list_ai_integrity_candidates(
                after_id=0,
                limit=self.batch_size,
            )
            stats["ai_wrapped"] = True
        stats["ai_integrity_candidates"] = len(ai_rows)
        stats["ai_missing"] = len(ai_rows)
        for row in ai_rows:
            if not self._wait_if_paused():
                stats["stopped"] = True
                break
            fid = int(row["id"])
            for key in (
                "dino",
                "clip",
                "texture",
                "semantic",
                "dna",
                "patch",
            ):
                stats[f"ai_missing_{key}"] += int(row.get(f"missing_{key}") or 0)
            worker = "ai_integrity_reconcile"
            if not self.db.try_acquire_processing_lock(fid, worker):
                stats["locked"] += 1
            else:
                try:
                    current = self.db.get_file_by_id(fid) or {}
                    if (
                        repairs_allowed
                        and str(current.get("heavy_status") or "") == "done"
                    ):
                        self.db.mark_heavy_incomplete(fid)
                        stats["ai_requeued"] += 1
                        if self.repair_ai and self._repair_ai_row(row):
                            stats["ai_repaired"] += 1
                finally:
                    self.db.release_processing_lock(fid, worker)
            stats["ai_cursor"] = fid
            self.db.set_meta(self.AI_CURSOR_META, str(fid))

        now = datetime.now(timezone.utc).isoformat()
        stats["physical"] = self.db.count_physical_readiness()
        if wrapped:
            stats["orphans"] = self.write_orphan_report(remove=repairs_allowed)
        self.db.set_meta(self.LAST_RUN_META, now)
        self.db.set_meta(
            self.LAST_STATS_META, json.dumps(stats, ensure_ascii=False)
        )
        if progress_callback:
            progress_callback(dict(stats))
        return stats

    def run_until_stopped(
        self,
        *,
        repair_missing: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        idle_sleep_sec: float = 0.5,
    ) -> dict[str, Any]:
        totals = {"checked": 0, "repaired": 0, "missing": 0}
        while not self._stop.is_set():
            batch = self.run_batch(
                repair_missing=repair_missing,
                progress_callback=progress_callback,
            )
            for key in totals:
                totals[key] += int(batch.get(key, 0) or 0)
            if batch.get("stopped"):
                break
            if not self._stop.wait(max(0.05, float(idle_sleep_sec))):
                continue
        totals["stopped"] = True
        return totals

    def write_orphan_report(
        self, *, sample_limit: int = 200, remove: bool = False
    ) -> dict[str, Any]:
        """Yalnız cache root içindeki DB referansı olmayan artifact'ları işler."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT thumbnail_path, feature_preview_path FROM files"
            ).fetchall()
        cache_root = Path(self.settings.cache_dir)
        referenced: set[str] = set()
        for row in rows:
            for raw in (row["thumbnail_path"], row["feature_preview_path"]):
                value = str(raw or "").strip()
                if not value:
                    continue
                path = Path(value)
                candidates = (
                    [path]
                    if path.is_absolute()
                    else [Path.cwd() / path, cache_root.parent / path]
                )
                referenced.update(
                    str(candidate.resolve()).lower() for candidate in candidates
                )
        samples: list[str] = []
        orphan_count = 0
        removed_count = 0
        for directory in (
            cache_root / "thumbnails",
            cache_root / "feature_previews",
        ):
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                if str(path.resolve()).lower() in referenced:
                    continue
                orphan_count += 1
                if len(samples) < max(0, int(sample_limit)):
                    samples.append(str(path))
                if remove:
                    try:
                        path.unlink()
                        removed_count += 1
                    except OSError:
                        logger.warning("Orphan cache silinemedi: %s", path)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "orphan_count": orphan_count,
            "removed_count": removed_count,
            "samples": samples,
            "action": "delete" if remove else "report_only",
        }
        report_dir = Path("data") / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "cache_orphans.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {**report, "report_path": str(report_path)}
