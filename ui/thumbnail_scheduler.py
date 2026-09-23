"""Thumbnail Preview Worker — loglu, kurtarmalı, sessiz hata yok."""

from __future__ import annotations

import heapq
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage

from core.logger import setup_logger
from core.thumbnail_cache import get_thumbnail_cache
from core.thumb_resolve import resolve_thumb_path, source_file_exists

logger = setup_logger(__name__)

MAX_CONCURRENT = 4

# Fail reason codes (UI + health)
FAIL_NO_PATH = "Cache yolu boş"
FAIL_CACHE_MISS = "Cache bulunamadı"
FAIL_SOURCE_MISSING = "Dosya okunamadı"
FAIL_DECODE = "Decode başarısız"
FAIL_CREATE = "Thumbnail oluşturulamadı"
FAIL_UI_SIGNAL = "UI signal ulaşmadı"
FAIL_WIDGET = "Widget güncellenmedi"
FAIL_CANCELLED = "İstek iptal edildi"


def _decode_thumb(path: str, size: int) -> QImage | None:
    """Sol kart thumbnail — geriye uyumlu (gerekirse büyütür)."""
    return _decode_image(path, size, allow_upscale=True)


@dataclass
class ThumbHealthSnapshot:
    total: int = 0
    loaded: int = 0
    missing: int = 0
    decode_error: int = 0
    cache_miss: int = 0
    file_missing: int = 0
    pending: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


@dataclass(order=True)
class _ThumbTask:
    priority: int
    seq: int
    file_id: int = field(compare=False)
    path: str = field(compare=False)
    size: int = field(compare=False)
    source_path: str = field(compare=False, default="")
    filename: str = field(compare=False, default="")
    mode: str = field(compare=False, default="thumb")  # thumb | detail
    feature_preview_path: str = field(compare=False, default="")


def _decode_image(path: str, size: int, *, allow_upscale: bool) -> QImage | None:
    """Disk önizlemesini decode et. Detail modunda upscale yok."""
    if not path or not os.path.isfile(path):
        return None
    try:
        img = QImage(path)
        if img.isNull():
            from io import BytesIO

            from PIL import Image

            with Image.open(path) as pil:
                pil = pil.convert("RGBA")
                buf = BytesIO()
                pil.save(buf, format="PNG")
                img = QImage.fromData(buf.getvalue(), "PNG")
        if img.isNull():
            return None
        long_edge = max(int(img.width()), int(img.height()))
        target = max(1, int(size))
        if long_edge <= target and not allow_upscale:
            return img
        if long_edge <= target and allow_upscale:
            # Kart thumbnail: mevcut davranış (küçük WEBP → istenen size)
            pass
        return img.scaled(
            target,
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
            if not allow_upscale
            else Qt.TransformationMode.FastTransformation,
        )
    except Exception as exc:
        logger.warning("THUMB decode error %s: %s", path, exc)
        return None


class _LoadRunnable(QRunnable):
    def __init__(
        self,
        file_id: int,
        path: str,
        size: int,
        source_path: str,
        filename: str,
        cache_dir: str,
        thumbnailer: Any,
        done: Callable,
        *,
        mode: str = "thumb",
        feature_preview_path: str = "",
        feature_previewer: Any = None,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self._file_id = file_id
        self._path = path
        self._size = size
        self._source_path = source_path
        self._filename = filename or os.path.basename(source_path or path or "")
        self._cache_dir = cache_dir
        self._thumbnailer = thumbnailer
        self._done = done
        self._mode = str(mode or "thumb")
        self._feature_preview_path = str(feature_preview_path or "").strip()
        self._feature_previewer = feature_previewer

    def run(self) -> None:
        if self._mode == "detail":
            self._run_detail()
            return
        self._run_thumb()

    def _run_detail(self) -> None:
        """Result Detail: feature/medium preview öncelik; NAS UI thread'de açılmaz."""
        fid = self._file_id
        name = self._filename
        try:
            resolved = self._resolve_detail_preview()
            if not resolved:
                # Son çare: küçük thumbnail (upscale yok — native px)
                resolved, status = resolve_thumb_path(
                    self._path,
                    cache_dir=self._cache_dir,
                    feature_preview_path=self._feature_preview_path,
                    source_path=self._source_path,
                )
                if not resolved:
                    if self._source_path and not source_file_exists(self._source_path):
                        self._fail(FAIL_SOURCE_MISSING)
                    else:
                        self._fail(FAIL_CACHE_MISS)
                    return
                logger.info(
                    "DETAIL file_id=%s FALLBACK thumb cache=%s path=%s",
                    fid,
                    status,
                    resolved,
                )
            img = _decode_image(resolved, self._size, allow_upscale=False)
            if img is None:
                self._fail(FAIL_DECODE)
                return
            logger.info(
                "DETAIL file_id=%s OK pixels=%sx%s path=%s size_req=%s",
                fid,
                img.width(),
                img.height(),
                resolved,
                self._size,
            )
            self._done(fid, img, self._size, "", name, resolved)
        except Exception as exc:
            logger.error("DETAIL file_id=%s exception: %s", fid, exc)
            self._fail(f"{FAIL_DECODE}: {exc}")

    def _resolve_detail_preview(self) -> str:
        """feature_preview_path → cache hit → create (worker thread)."""
        candidates: list[str] = []
        if self._feature_preview_path:
            candidates.append(self._feature_preview_path)
        fp_cache = self._feature_previewer
        if fp_cache is None and self._cache_dir:
            from core.preview_cache import FeaturePreviewCache

            fp_cache = FeaturePreviewCache(
                self._cache_dir,
                max_edge=max(1024, int(self._size or 1024)),
            )
            self._feature_previewer = fp_cache
        if fp_cache is not None and self._source_path:
            try:
                existing = fp_cache.get_existing(self._source_path)
                if existing.success and existing.preview_path:
                    candidates.append(str(existing.preview_path))
                else:
                    # Deterministik yol (DB path boş olsa bile)
                    candidates.append(str(fp_cache.preview_path_for(self._source_path)))
            except Exception as exc:
                logger.debug("DETAIL get_existing: %s", exc)

        for path in candidates:
            if path and os.path.isfile(path) and os.path.getsize(path) > 0:
                logger.info("DETAIL file_id=%s CACHE HIT path=%s", self._file_id, path)
                return path

        # Cache miss → medium preview üret (NAS burada, detail pool; GIL/NAS UI'ı kilitleyebilir —
        # kısa yield ile heartbeat'e fırsat ver, sonra create).
        if fp_cache is not None and self._source_path and source_file_exists(
            self._source_path
        ):
            try:
                import time as _time

                _time.sleep(0.001)
                logger.info(
                    "DETAIL file_id=%s CACHE MISS → FeaturePreviewCache.create",
                    self._file_id,
                )
                created = fp_cache.create(self._source_path)
                if created.success and created.preview_path:
                    logger.info(
                        "DETAIL file_id=%s CREATED %sx%s path=%s",
                        self._file_id,
                        created.width,
                        created.height,
                        created.preview_path,
                    )
                    return str(created.preview_path)
                logger.warning(
                    "DETAIL file_id=%s create failed: %s",
                    self._file_id,
                    created.error,
                )
            except Exception as exc:
                logger.error(
                    "DETAIL create exception file_id=%s: %s", self._file_id, exc
                )
        return ""

    def _run_thumb(self) -> None:
        fid = self._file_id
        name = self._filename
        try:
            resolved, status = resolve_thumb_path(
                self._path,
                cache_dir=self._cache_dir,
                feature_preview_path=self._feature_preview_path,
                source_path=self._source_path,
            )
            logger.info(
                "THUMB file_id=%s name=%s cache=%s path=%s",
                fid,
                name,
                status.upper(),
                self._path,
            )

            if status == "empty" and not self._source_path:
                self._fail(FAIL_NO_PATH)
                return

            if status == "miss" or not resolved:
                logger.info(
                    "THUMB file_id=%s CACHE MISS — kaynak=%s exists=%s",
                    fid,
                    self._source_path,
                    source_file_exists(self._source_path),
                )
                created, create_err = self._try_create()
                if not created:
                    if self._source_path and not source_file_exists(self._source_path):
                        self._fail(FAIL_SOURCE_MISSING)
                    elif self._source_path:
                        detail = create_err or FAIL_CREATE
                        self._fail(
                            detail
                            if detail.startswith(FAIL_CREATE)
                            or "dependency" in detail.lower()
                            or "timeout" in detail.lower()
                            or "ghostscript" in detail.lower()
                            or "pymupdf" in detail.lower()
                            else f"{FAIL_CREATE}: {detail}"
                        )
                    else:
                        self._fail(FAIL_CACHE_MISS)
                    return
                resolved = created
                logger.info(
                    "THUMB file_id=%s regenerated → %s", fid, resolved
                )

            img = self._decode(resolved)
            if img is None:
                # One retry: regenerate then decode again
                logger.warning(
                    "THUMB file_id=%s decode fail → retry regenerate", fid
                )
                created, _cerr = self._try_create(force=True)
                if created:
                    img = self._decode(created)
                    resolved = created
                if img is None:
                    self._fail(FAIL_DECODE)
                    return

            logger.info(
                "THUMB file_id=%s OK decode+scale size=%s → signal",
                fid,
                self._size,
            )
            self._done(fid, img, self._size, "", name, resolved)
        except Exception as exc:
            logger.error("THUMB file_id=%s exception: %s", fid, exc)
            self._fail(f"{FAIL_DECODE}: {exc}")

    def _decode(self, path: str) -> QImage | None:
        return _decode_thumb(path, self._size)

    def _try_create(self, *, force: bool = False) -> tuple[str, str]:
        """Return (thumbnail_path_or_empty, error_detail). Never permanently blocks retry."""
        if not self._thumbnailer:
            return "", FAIL_NO_PATH
        try:
            if force and self._source_path:
                out = self._thumbnailer.thumbnail_path_for(self._source_path)
                if out.exists():
                    try:
                        out.unlink()
                    except OSError:
                        pass
            # Prefer existing feature preview (no NAS re-read) — even if source offline
            try:
                from core.preview_cache import FeaturePreviewCache
                from core.thumb_resolve import lookup_valid_preview

                preview = ""
                if self._feature_preview_path and os.path.isfile(
                    self._feature_preview_path
                ):
                    preview = self._feature_preview_path
                elif self._cache_dir and self._source_path:
                    preview = lookup_valid_preview(
                        self._source_path, self._cache_dir
                    )
                    if not preview:
                        fp = (
                            self._feature_previewer
                            or FeaturePreviewCache(self._cache_dir)
                        )
                        existing = fp.get_existing(self._source_path)
                        if existing.success and existing.preview_path:
                            preview = str(existing.preview_path)
                if preview:
                    result = self._thumbnailer.create_from_existing_preview(
                        self._source_path or preview, preview
                    )
                    if result.success and result.thumbnail_path:
                        logger.info(
                            "THUMB file_id=%s SSOT preview→thumb → %s",
                            self._file_id,
                            result.thumbnail_path,
                        )
                        return str(result.thumbnail_path), ""
            except Exception as exc:
                logger.debug("THUMB feature_preview reuse: %s", exc)

            if not source_file_exists(self._source_path):
                return "", FAIL_SOURCE_MISSING if self._source_path else FAIL_NO_PATH

            result = self._thumbnailer.create(self._source_path)
            if result.success and result.thumbnail_path:
                logger.info(
                    "THUMB file_id=%s created cache=%s",
                    self._file_id,
                    result.thumbnail_path,
                )
                return str(result.thumbnail_path), ""
            err = str(result.error or FAIL_CREATE)
            logger.warning(
                "THUMB file_id=%s create failed: %s",
                self._file_id,
                err,
            )
            return "", err
        except Exception as exc:
            logger.error("THUMB create exception file_id=%s: %s", self._file_id, exc)
            return "", str(exc)

    def _fail(self, reason: str) -> None:
        logger.error(
            "THUMB FAIL file_id=%s name=%s reason=%s",
            self._file_id,
            self._filename,
            reason,
        )
        self._done(self._file_id, None, self._size, reason, self._filename, "")


class ThumbnailScheduler(QObject):
    """Öncelikli kuyruk + otomatik kurtarma + sağlık metrikleri."""

    thumbnail_ready = Signal(int, object, int)  # file_id, QImage|None, size
    thumbnail_failed = Signal(int, str, str)  # file_id, filename, reason
    health_changed = Signal(object)  # ThumbHealthSnapshot

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(MAX_CONCURRENT)
        # Detail/FeaturePreviewCache.create can hold the GIL (PIL/NAS). Keep it off the
        # thumb pool and serialize so list scrolling stays responsive during cache miss.
        self._detail_pool = QThreadPool(self)
        self._detail_pool.setMaxThreadCount(1)
        # Default True so unit tests / non-MainWindow hosts are unaffected.
        # MainWindow sets False until first interactive frame, then mark_ui_interactive().
        self._detail_ui_ready = True
        self._detail_deferred: list[tuple] = []
        self._heap: list[_ThumbTask] = []
        self._queued: set[int] = set()
        self._in_flight: set[int] = set()
        self._loaded: set[int] = set()
        self._image_cache: dict[int, object] = {}
        self._detail_cache: dict[int, object] = {}
        self._detail_meta: dict[int, dict[str, Any]] = {}
        self._detail_in_flight: set[int] = set()
        self._failed: dict[int, str] = {}
        self._fail_names: dict[int, str] = {}
        self._expected: set[int] = set()
        self._seq = 0
        self._lock = threading.Lock()
        self._generation = 0
        self._thumbnailer = None
        self._feature_previewer = None
        self._cache_dir = ""
        self._lru = get_thumbnail_cache(800)

    def configure(self, *, cache_dir: str = "", thumbnailer=None) -> None:
        self._cache_dir = str(cache_dir or "")
        self._thumbnailer = thumbnailer
        self._feature_previewer = None
        if self._cache_dir:
            try:
                from core.preview_cache import FeaturePreviewCache

                self._feature_previewer = FeaturePreviewCache(
                    self._cache_dir,
                    max_edge=1024,
                )
            except Exception as exc:
                logger.warning("FeaturePreviewCache init failed: %s", exc)

    def begin_search_batch(self, expected_ids: list[int] | None = None) -> None:
        """Yeni arama sonuç seti için sağlık sayaçlarını sıfırla (iptal etmeden)."""
        with self._lock:
            self._expected = {int(i) for i in (expected_ids or [])}
            # Keep loaded/failed for visible continuity; reset fail list for new batch report
            self._failed.clear()
            self._fail_names.clear()
        self._emit_health()

    def cancel_all(self) -> None:
        with self._lock:
            self._generation += 1
            self._heap.clear()
            self._queued.clear()
            self._in_flight.clear()
            self._detail_in_flight.clear()
            self._detail_deferred.clear()
            self._loaded.clear()
            self._image_cache.clear()
            self._detail_cache.clear()
            self._detail_meta.clear()
            self._failed.clear()
            self._fail_names.clear()
            self._expected.clear()
        self._lru.clear()
        self._emit_health()

    def peek_detail_meta(self, file_id: int) -> dict[str, Any]:
        with self._lock:
            return dict(self._detail_meta.get(int(file_id), {}))

    def mark_ui_interactive(self) -> None:
        """Call after MainWindow is shown — flush deferred detail creates."""
        self._detail_ui_ready = True
        pending = list(self._detail_deferred)
        self._detail_deferred.clear()
        for args in pending:
            self._start_detail_runnable(*args)

    def _start_detail_runnable(
        self,
        fid: int,
        thumb: str,
        size: int,
        source_path: str,
        filename: str,
        fp: str,
        gen: int,
    ) -> None:
        runnable = _LoadRunnable(
            fid,
            thumb,
            size,
            source_path,
            filename,
            self._cache_dir,
            self._thumbnailer,
            lambda f, img, sz, reason, name, resolved, g=gen: self._finish_detail(
                f, img, sz, reason, name, resolved, g
            ),
            mode="detail",
            feature_preview_path=fp,
            feature_previewer=self._feature_previewer,
        )
        self._detail_pool.start(runnable)

    def request_detail_preview(
        self,
        file_id: int,
        thumbnail_path: str = "",
        *,
        source_path: str = "",
        filename: str = "",
        feature_preview_path: str = "",
        size: int = 1024,
        priority: int = 0,
    ) -> None:
        """Sağ Result Detail — feature/medium preview; thumbnail yalnızca fallback."""
        fid = int(file_id)
        size = max(768, min(1024, int(size or 1024)))
        thumb = str(thumbnail_path or "").strip()
        source_path = str(source_path or "").strip()
        filename = str(filename or "").strip()
        fp = str(feature_preview_path or "").strip()

        with self._lock:
            cached = self._detail_cache.get(fid)
            busy = fid in self._detail_in_flight

        if cached is not None and not cached.isNull():
            self.thumbnail_ready.emit(fid, cached, size)
            return
        if busy:
            return
        if not thumb and not source_path and not fp:
            self.thumbnail_failed.emit(fid, filename or str(fid), FAIL_NO_PATH)
            self.thumbnail_ready.emit(fid, None, size)
            return

        with self._lock:
            self._detail_in_flight.add(fid)
            gen = self._generation

        args = (fid, thumb, size, source_path, filename, fp, gen)
        # Startup: keep FeaturePreviewCache.create off the critical path until UI is usable.
        if not self._detail_ui_ready:
            self._detail_deferred = [
                a for a in self._detail_deferred if a[0] != fid
            ]
            self._detail_deferred.append(args)
            return
        self._start_detail_runnable(*args)

    def _finish_detail(
        self,
        file_id: int,
        image: QImage | None,
        size: int,
        reason: str,
        filename: str,
        resolved_path: str,
        gen: int,
    ) -> None:
        with self._lock:
            if gen != self._generation:
                self._detail_in_flight.discard(file_id)
                return
            self._detail_in_flight.discard(file_id)
            if image is not None:
                self._detail_cache[file_id] = image
                self._detail_meta[file_id] = {
                    "path": resolved_path,
                    "width": int(image.width()),
                    "height": int(image.height()),
                    "size_req": int(size),
                }
            else:
                self._detail_cache.pop(file_id, None)
                self._detail_meta.pop(file_id, None)
        if image is not None:
            logger.info(
                "DETAIL file_id=%s UI signal OK %sx%s path=%s",
                file_id,
                image.width(),
                image.height(),
                resolved_path,
            )
            self.thumbnail_ready.emit(file_id, image, size)
        else:
            self.thumbnail_failed.emit(file_id, filename or str(file_id), reason)
            self.thumbnail_ready.emit(file_id, None, size)
        self._emit_health()

    def peek_image(self, file_id: int) -> QImage | None:
        """Yuklenmis thumbnail — widget henuz gorunur olmasa bile RAM'de."""
        fid = int(file_id)
        with self._lock:
            img = self._image_cache.get(fid)
        if img is not None and not img.isNull():
            return img
        for sz in (120, 88, 72, 52):
            lru_img, _pix = self._lru.get(fid, sz)
            if lru_img is not None and not lru_img.isNull():
                return lru_img
        return None

    def peek_detail_image(self, file_id: int) -> QImage | None:
        """Detail/preview RAM cache — UI thread'de disk yok."""
        fid = int(file_id)
        with self._lock:
            img = self._detail_cache.get(fid)
        if img is not None and not img.isNull():
            return img
        return self.peek_image(fid)

    def request(
        self,
        file_id: int,
        path: str,
        size: int,
        *,
        priority: int = 10,
        source_path: str = "",
        filename: str = "",
        feature_preview_path: str = "",
    ) -> None:
        fid = int(file_id)
        path = str(path or "").strip()
        source_path = str(source_path or "").strip()
        filename = str(filename or "").strip()
        feature_preview_path = str(feature_preview_path or "").strip()
        size = int(size)

        with self._lock:
            self._expected.add(fid)
            if fid in self._loaded and fid in self._image_cache:
                cached = self._image_cache[fid]
            else:
                cached = None
            busy = fid in self._queued or fid in self._in_flight

        # LRU v2 — RAM'den aninda (UI thread decode yok)
        lru_img, _lru_pix = self._lru.get(fid, size)
        if lru_img is not None and not lru_img.isNull():
            with self._lock:
                self._loaded.add(fid)
                self._image_cache[fid] = lru_img
            self.thumbnail_ready.emit(fid, lru_img, size)
            self._emit_health()
            return

        if cached is not None:
            self.thumbnail_ready.emit(fid, cached, size)
            return

        if not path and not source_path and not feature_preview_path:
            with self._lock:
                self._failed[fid] = FAIL_NO_PATH
                self._fail_names[fid] = filename or str(fid)
            logger.error(
                "THUMB FAIL file_id=%s name=%s reason=%s",
                fid,
                filename,
                FAIL_NO_PATH,
            )
            self.thumbnail_failed.emit(fid, filename or str(fid), FAIL_NO_PATH)
            self.thumbnail_ready.emit(fid, None, size)
            self._emit_health()
            return

        if busy:
            # Fast scroll: if still only queued (not in flight), boost priority.
            with self._lock:
                if fid in self._queued and priority < 10:
                    new_heap: list[_ThumbTask] = []
                    boosted = False
                    for task in self._heap:
                        if task.file_id == fid:
                            self._seq += 1
                            new_heap.append(
                                _ThumbTask(
                                    int(priority),
                                    self._seq,
                                    task.file_id,
                                    task.path,
                                    task.size,
                                    task.source_path,
                                    task.filename,
                                    feature_preview_path=(
                                        feature_preview_path
                                        or task.feature_preview_path
                                    ),
                                )
                            )
                            boosted = True
                        else:
                            new_heap.append(task)
                    if boosted:
                        self._heap = new_heap
                        heapq.heapify(self._heap)
            self._pump()
            self._emit_health()
            return

        # Zaten yuklendi ama UI kacirdiysa tekrar signal gonder
        with self._lock:
            if fid in self._loaded:
                img = self._image_cache.get(fid)
                if img is not None and not img.isNull():
                    self.thumbnail_ready.emit(fid, img, size)
                    self._emit_health()
                    return

        # Tum decode islemleri arka planda — UI thread'de sync decode YOK
        with self._lock:
            if fid in self._queued or fid in self._in_flight or fid in self._loaded:
                pass
            else:
                self._seq += 1
                heapq.heappush(
                    self._heap,
                    _ThumbTask(
                        priority,
                        self._seq,
                        fid,
                        path,
                        size,
                        source_path,
                        filename,
                        feature_preview_path=feature_preview_path,
                    ),
                )
                self._queued.add(fid)
                self._failed.pop(fid, None)
        self._pump()
        self._emit_health()

    def request_visible(self, items: list[tuple]) -> None:
        """items: (file_id, path, size[, source, filename[, feature_preview_path]]).

        Visible cards get priority 0; previously queued off-screen work is dropped
        (in-flight cannot cancel) so viewport is not starved by a demoted pile-up.
        """
        visible_set = {int(item[0]) for item in items if item}
        with self._lock:
            if self._heap and visible_set:
                kept: list[_ThumbTask] = []
                for task in self._heap:
                    if task.file_id in visible_set:
                        # Drop so request() re-queues at priority 0 with fresh seq.
                        self._queued.discard(task.file_id)
                        continue
                    # Off-screen: drop from queue (do not demote/retain).
                    self._queued.discard(task.file_id)
                self._heap = kept
                heapq.heapify(self._heap)
        for item in items:
            if len(item) >= 6:
                fid, path, size, source, name, fp = item[:6]
                self.request(
                    int(fid),
                    str(path),
                    int(size),
                    priority=0,
                    source_path=str(source or ""),
                    filename=str(name or ""),
                    feature_preview_path=str(fp or ""),
                )
            elif len(item) >= 5:
                fid, path, size, source, name = item[:5]
                self.request(
                    int(fid),
                    str(path),
                    int(size),
                    priority=0,
                    source_path=str(source or ""),
                    filename=str(name or ""),
                )
            else:
                fid, path, size = item[:3]
                self.request(int(fid), str(path), int(size), priority=0)

    def mark_widget_updated(self, file_id: int, ok: bool) -> None:
        """Kart görünür sonuç setindeyken signal geldi ama widget yoksa FAIL."""
        fid = int(file_id)
        if ok:
            return
        with self._lock:
            if fid not in self._expected:
                return
            if fid not in self._loaded:
                return
            self._failed[fid] = FAIL_WIDGET
            self._fail_names.setdefault(fid, str(fid))
        logger.error("THUMB FAIL file_id=%s reason=%s", fid, FAIL_WIDGET)
        self.thumbnail_failed.emit(fid, str(fid), FAIL_WIDGET)
        self._emit_health()

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._heap) + len(self._in_flight)

    def health_snapshot(self) -> ThumbHealthSnapshot:
        with self._lock:
            expected = set(self._expected) | set(self._loaded) | set(self._failed)
            total = len(expected) if expected else (
                len(self._loaded) + len(self._failed) + len(self._queued) + len(self._in_flight)
            )
            loaded = len(self._loaded)
            failed = dict(self._failed)
            names = dict(self._fail_names)
            pending = len(self._queued) + len(self._in_flight)
        decode_n = sum(1 for r in failed.values() if FAIL_DECODE in r)
        cache_n = sum(
            1
            for r in failed.values()
            if r in (FAIL_CACHE_MISS, FAIL_NO_PATH) or FAIL_CREATE in r
        )
        file_n = sum(1 for r in failed.values() if FAIL_SOURCE_MISSING in r)
        missing = len(failed)
        failures = [
            {"file_id": str(fid), "filename": names.get(fid, str(fid)), "reason": reason}
            for fid, reason in list(failed.items())[:50]
        ]
        return ThumbHealthSnapshot(
            total=total,
            loaded=loaded,
            missing=missing,
            decode_error=decode_n,
            cache_miss=cache_n,
            file_missing=file_n,
            pending=pending,
            failures=failures,
        )

    def _emit_health(self) -> None:
        self.health_changed.emit(self.health_snapshot())

    def _pump(self) -> None:
        while True:
            with self._lock:
                if len(self._in_flight) >= MAX_CONCURRENT or not self._heap:
                    return
                task = heapq.heappop(self._heap)
                self._queued.discard(task.file_id)
                self._in_flight.add(task.file_id)
                gen = self._generation
            runnable = _LoadRunnable(
                task.file_id,
                task.path,
                task.size,
                task.source_path,
                task.filename,
                self._cache_dir,
                self._thumbnailer,
                lambda fid, img, sz, reason, name, resolved, g=gen: self._finish(
                    fid, img, sz, reason, name, resolved, g
                ),
                feature_preview_path=task.feature_preview_path,
                feature_previewer=self._feature_previewer,
            )
            self._pool.start(runnable)

    def _finish(
        self,
        file_id: int,
        image: QImage | None,
        size: int,
        reason: str,
        filename: str,
        resolved_path: str,
        gen: int,
    ) -> None:
        with self._lock:
            if gen != self._generation:
                self._in_flight.discard(file_id)
                logger.warning(
                    "THUMB file_id=%s dropped (generation mismatch) — %s",
                    file_id,
                    FAIL_CANCELLED,
                )
                return
            self._in_flight.discard(file_id)
            if image is not None:
                self._loaded.add(file_id)
                self._image_cache[file_id] = image
                self._failed.pop(file_id, None)
                try:
                    self._lru.put(file_id, size, image)
                except Exception:
                    pass
            else:
                self._failed[file_id] = reason or FAIL_DECODE
                self._fail_names[file_id] = filename or str(file_id)
                self._image_cache.pop(file_id, None)
        if image is not None:
            logger.info(
                "THUMB file_id=%s UI signal OK path=%s",
                file_id,
                resolved_path,
            )
            self.thumbnail_ready.emit(file_id, image, size)
        else:
            self.thumbnail_failed.emit(file_id, filename or str(file_id), reason)
            self.thumbnail_ready.emit(file_id, None, size)
        self._emit_health()
        self._pump()


_scheduler: ThumbnailScheduler | None = None


def get_thumbnail_scheduler(parent=None) -> ThumbnailScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ThumbnailScheduler(parent)
    return _scheduler
