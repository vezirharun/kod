"""Index Engine V3 — gerçek extractor bağları (Thumbnailer/VIPS/DINO/CLIP/…).

Legacy Indexer / dual-pipeline / event counter KULLANMAZ.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as py_queue
import json
import time
import threading
from collections import OrderedDict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from core.feature_extractor import FeatureExtractor
from core.index_analysis_guards import local_artifact_exists
from core.index_v3.safe_decode import is_risky_path, run_with_timeout
from core.index_v3.faiss_sync import sync_file_embeddings
from core.index_v3.types import Artifact, Job
from core.index_v3.worker import WorkerStats
from core.light_asset_extractor import LightAssetResult, extract_light_assets_once
from core.manual_label_guard import parse_texture_map
from core.metadata_sanitize import metadata_to_json
from core.ocr_engine import OCREngine
from core.preview_cache import FEATURE_PREVIEW_VERSION, FeaturePreviewCache
from core.settings import AppSettings, DEFAULT_CACHE_DIR
from core.thumbnailer import Thumbnailer
from core.utils import normalize_path

_file_read_tls = threading.local()


def _note_file_read_ms(elapsed_ms: float) -> None:
    stats = getattr(_file_read_tls, "stats", None)
    if stats is None:
        return
    stats.file_read_ms = float(getattr(stats, "file_read_ms", 0) or 0) + float(
        elapsed_ms
    )


def _isolated_light_entry(
    settings_data: dict[str, Any],
    path: str,
    mtime: float,
    result_queue: Any,
) -> None:
    """Child process: libvips/Pillow light extract (parent applies hard timeout)."""
    try:
        allowed = {f.name for f in fields(AppSettings)}
        settings = AppSettings(
            **{k: v for k, v in settings_data.items() if k in allowed}
        )
        cache = str(settings.cache_dir)
        thumbnailer = Thumbnailer(
            cache,
            max_edge=int(settings.thumbnail_max_edge or 512),
            fmt=str(settings.thumbnail_format or "webp"),
        )
        feature_preview = FeaturePreviewCache(
            cache,
            int(settings.feature_preview_max_edge or 1024),
            str(settings.thumbnail_format or "webp"),
        )
        result = extract_light_assets_once(
            path,
            thumbnailer=thumbnailer,
            feature_preview=feature_preview,
            source_mtime=mtime,
        )
        result_queue.put(
            (
                "ok",
                {
                    "success": bool(result.success),
                    "error": str(result.error or ""),
                    "thumbnail_path": str(result.thumbnail_path or ""),
                    "thumbnail_width": int(result.thumbnail_width or 0),
                    "thumbnail_height": int(result.thumbnail_height or 0),
                    "thumbnail_status": str(result.thumbnail_status or ""),
                    "feature_preview_path": str(result.feature_preview_path or ""),
                    "metadata": dict(result.metadata or {}),
                    "metadata_status": str(result.metadata_status or ""),
                    "visual_status": str(result.visual_status or ""),
                    "visual_renderer": str(result.visual_renderer or ""),
                },
            )
        )
    except BaseException as exc:
        result_queue.put(("err", repr(exc)))


def _run_light_isolated(
    settings: AppSettings,
    path: str,
    mtime: float,
    *,
    timeout_sec: float,
) -> LightAssetResult:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_isolated_light_entry,
        args=(asdict(settings), path, float(mtime), q),
        daemon=True,
    )
    try:
        proc.start()
        proc.join(max(0.5, float(timeout_sec)))
        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(2.0)
            raise TimeoutError(f"v3_decode_timeout:{timeout_sec}s")
        # mp.Queue.empty() is racy on Windows: the child may have exited while
        # its feeder thread is still flushing the message. That made successful
        # isolated decodes look like v3_decode_subprocess_exit and, after two
        # attempts, the job became failed_permanent. Wait for the actual result.
        try:
            kind, payload = q.get(timeout=2.0)
        except py_queue.Empty as exc:
            raise RuntimeError(
                f"v3_decode_subprocess_exit:{proc.exitcode}"
            ) from exc
        if kind != "ok":
            raise RuntimeError(str(payload))
        return LightAssetResult(
            success=bool(payload.get("success")),
            error=str(payload.get("error") or ""),
            thumbnail_path=str(payload.get("thumbnail_path") or ""),
            thumbnail_width=int(payload.get("thumbnail_width") or 0),
            thumbnail_height=int(payload.get("thumbnail_height") or 0),
            thumbnail_status=str(payload.get("thumbnail_status") or ""),
            feature_preview_path=str(payload.get("feature_preview_path") or ""),
            metadata=dict(payload.get("metadata") or {}),
            metadata_status=str(payload.get("metadata_status") or ""),
            visual_status=str(payload.get("visual_status") or ""),
            visual_renderer=str(payload.get("visual_renderer") or ""),
        )
    finally:
        # Queue feeder thread + Process handle sızıntısı: NAS'ta her dosya
        # için Process açılır; close/join yoksa threads 100→1300+ şişer.
        try:
            q.close()
        except Exception:
            pass
        try:
            q.join_thread()
        except Exception:
            pass
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
        except Exception:
            pass
        try:
            proc.close()
        except Exception:
            pass


class RealArtifactProcessor:
    """Artifact-first: yalnız istenen artifact üretilir; VERIFY caller'da."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        timeout_sec: float | None = None,
        retries: int = 2,
        enable_ai: bool | None = None,
    ) -> None:
        self.settings = settings
        self.timeout_sec = float(
            timeout_sec
            if timeout_sec is not None
            else getattr(settings, "network_timeout_sec", 30) or 30
        )
        self.retries = max(0, int(retries))
        self.enable_ai = (
            bool(settings.ai_embedding_enabled)
            if enable_ai is None
            else bool(enable_ai)
        )
        cache = str(settings.cache_dir)
        self.thumbnailer = Thumbnailer(
            cache,
            max_edge=int(getattr(settings, "thumbnail_max_edge", 512) or 512),
            fmt=str(getattr(settings, "thumbnail_format", "webp") or "webp"),
        )
        self.feature_preview = FeaturePreviewCache(
            cache,
            int(getattr(settings, "feature_preview_max_edge", 1024) or 1024),
            str(getattr(settings, "thumbnail_format", "webp") or "webp"),
        )
        extra_cache = str(DEFAULT_CACHE_DIR)
        self._preview_caches = [self.feature_preview]
        if extra_cache and extra_cache != cache:
            self._preview_caches.append(
                FeaturePreviewCache(
                    extra_cache,
                    int(getattr(settings, "feature_preview_max_edge", 1024) or 1024),
                    str(getattr(settings, "thumbnail_format", "webp") or "webp"),
                )
            )
        self._extractor: FeatureExtractor | None = None
        self._hash_extractor: FeatureExtractor | None = None
        self._ocr: OCREngine | None = None
        self._arr_lru: OrderedDict[str, Any] = OrderedDict()
        self._arr_lru_max = 8

    def _get_extractor(self, *, need_ai: bool = True) -> FeatureExtractor:
        if not need_ai:
            if self._hash_extractor is None:
                self._hash_extractor = FeatureExtractor(
                    use_ai=False, use_gpu=False, fast_hash_only=False
                )
            return self._hash_extractor
        if self._extractor is None:
            self._extractor = FeatureExtractor(
                use_ai=self.enable_ai,
                use_gpu=bool(getattr(self.settings, "use_gpu", False)),
                fast_hash_only=False,
            )
        return self._extractor

    def _cached_feature_array(self, src: str) -> Any:
        """Reuse decoded feature-preview pixels across HASH/DINO/CLIP/TEXTURE/PATCH."""
        hit = self._arr_lru.get(src)
        if hit is not None:
            self._arr_lru.move_to_end(src)
            return hit
        t0 = time.perf_counter()
        image = Thumbnailer.load_image(src)
        _note_file_read_ms((time.perf_counter() - t0) * 1000.0)
        if image is None:
            return None
        self._arr_lru[src] = image
        while len(self._arr_lru) > self._arr_lru_max:
            self._arr_lru.popitem(last=False)
        return image

    def process(self, db: Any, job: Job, stats: WorkerStats) -> bool:
        art = job.artifact
        stats.bump_model(art.value)
        row = db.get_file_by_id(job.file_id) or {}
        path = normalize_path(str(job.path or row.get("path") or ""))
        if not path:
            raise RuntimeError("missing_source_path")
        _file_read_tls.stats = stats
        try:
            return self._process_dispatch(db, job, stats, row, path, art)
        finally:
            _file_read_tls.stats = None

    def _process_dispatch(
        self, db: Any, job: Job, stats: WorkerStats, row: dict[str, Any], path: str, art: Artifact
    ) -> bool:
        if art == Artifact.PREVIEW:
            return self._process_preview(db, job, row, path)

        if art == Artifact.THUMBNAIL:
            # Prefer existing FeaturePreview → 256 thumb (no NAS). Else source → 256.
            # Never forces FeaturePreview creation for the whole archive.
            preview_candidates = [
                str(row.get("feature_preview_path") or "").strip(),
            ]
            for cache in getattr(self, "_preview_caches", [self.feature_preview]):
                preview_candidates.append(str(cache.preview_path_for(path)).strip())
            for preview_path in preview_candidates:
                if not preview_path or not local_artifact_exists(preview_path):
                    continue
                result = self.thumbnailer.create_from_existing_preview(path, preview_path)
                if not result.success:
                    continue
                db.upsert_file({
                    "path": path,
                    "thumbnail_path": result.thumbnail_path,
                    "width": int(result.width or 0),
                    "height": int(result.height or 0),
                    "thumbnail_status": "from_preview",
                })
                thumb_ok = local_artifact_exists(result.thumbnail_path)
                prev_ok = local_artifact_exists(preview_path)
                db.update_physical_readiness(
                    job.file_id, thumbnail_ready=thumb_ok,
                    preview_ready=prev_ok, requeue_missing=False,
                )
                if thumb_ok:
                    return True
            # Independent 256px from source (cache HIT inside Thumbnailer.create).
            result = self.thumbnailer.create(path)
            if result.success and result.thumbnail_path:
                db.upsert_file({
                    "path": path,
                    "thumbnail_path": result.thumbnail_path,
                    "width": int(result.width or 0),
                    "height": int(result.height or 0),
                    "thumbnail_status": "from_source",
                })
                thumb_ok = local_artifact_exists(result.thumbnail_path)
                # Do not clobber preview readiness when thumb is source-derived.
                prev_path = str(row.get("feature_preview_path") or "").strip()
                prev_ok = (
                    int(row.get("physical_preview_ready") or 0) == 1
                    or (bool(prev_path) and local_artifact_exists(prev_path))
                )
                db.update_physical_readiness(
                    job.file_id, thumbnail_ready=thumb_ok,
                    preview_ready=prev_ok, requeue_missing=False,
                )
                if thumb_ok:
                    return True
            err = str(getattr(result, "error", "") or "thumbnail_create_failed")
            soft = any(
                t in err.lower()
                for t in (
                    "timeout",
                    "network",
                    "nas",
                    "errno 22",
                    "winerror",
                    "temporarily",
                    "unavailable",
                    "permission",
                )
            )
            if soft:
                raise RuntimeError(f"thumbnail_deferred:{err}")
            raise RuntimeError(f"thumbnail_create_failed:{err}")

        if art == Artifact.HASH:
            return self._process_hash(db, job, row, path)
        if art == Artifact.METADATA:
            return self._process_metadata(db, job, row, path)

        if art in (
            Artifact.DINO,
            Artifact.CLIP,
            Artifact.TEXTURE,
            Artifact.PATCH,
        ):
            return self._process_vision(db, job, row, path, art)

        if art == Artifact.SEMANTIC:
            return self._process_semantic(db, job)
        if art == Artifact.DNA:
            return self._process_dna(db, job)
        if art == Artifact.OBJECT_CONCEPT:
            return self._process_object_concept(db, job, row)
        if art == Artifact.OWLV2:
            return self._process_owlv2(db, job, row)
        if art == Artifact.OCR:
            return self._process_ocr(db, job, row, path)
        raise RuntimeError(f"unknown_artifact:{art.value}")

    def _process_preview(
        self, db: Any, job: Job, row: dict[str, Any], path: str
    ) -> bool:
        """Yalnız Feature Preview üretir; Thumbnail sonraki ayrı iştir."""
        existing = None
        for cache in getattr(self, "_preview_caches", [self.feature_preview]):
            hit = cache.get_existing(path)
            if hit.success:
                existing = hit
                break
        if existing and existing.success:
            # Mode C light gate — do not serve corrupt/misrepresenting as ready.
            try:
                from core.preview_self_heal.hooks import gate_new_preview
                from core.preview_self_heal.validate import Verdict

                g = gate_new_preview(
                    str(existing.preview_path),
                    path,
                    compare_path=None,
                    allow_source_compare=True,
                )
                if g.verdict == Verdict.INVALID:
                    existing = None
                elif g.verdict.value == "suspicious":
                    # Soft mark; still skip recreate this pass if artifact exists
                    try:
                        db.upsert_file(
                            {
                                "path": path,
                                "preview_status": "preview_suspicious",
                            }
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        if existing and existing.success:
            db.upsert_file({
                "path": path,
                "feature_preview_path": existing.preview_path,
                "feature_preview_version": FEATURE_PREVIEW_VERSION,
                "preview_status": "preview_ok",
            })
            db.update_physical_readiness(
                job.file_id,
                thumbnail_ready=bool(local_artifact_exists(str(row.get("thumbnail_path") or ""))),
                preview_ready=True, requeue_missing=False,
            )
            return True

        def _work():
            work_path = path
            try:
                from core.global_preview import materialize_visual
                vis = materialize_visual(path, cache_dir=str(self.thumbnailer.cache_dir.parent))
                if not vis.ok:
                    raise RuntimeError(f"{vis.status}:{vis.error}"[:300])
                work_path = vis.raster_path or path
            except RuntimeError:
                raise
            except Exception:
                work_path = path
            result = self.feature_preview.create(
                path, render_path=work_path if work_path != path else None
            )
            if not result.success:
                err = result.error or "preview_extract_failed"
                low = err.lower()
                if (
                    "not a known file format" in low
                    or "vipsforeignload" in low
                    or "cannot identify image" in low
                    or "truncated" in low
                ):
                    err = f"corrupt_file:{err}"
                raise RuntimeError(err)
            # Mode C — validate new artifact; compare vs local render_path (no NAS).
            try:
                from core.preview_self_heal.hooks import apply_gate_or_repair

                def _recreate():
                    return self.feature_preview.create(
                        path, render_path=work_path if work_path != path else None
                    )

                ok, new_path, reason = apply_gate_or_repair(
                    source_path=path,
                    artifact_path=str(result.preview_path or ""),
                    create_fn=_recreate,
                    compare_path=work_path if work_path != path else None,
                    settings=getattr(self, "settings", None),
                )
                if not ok:
                    raise RuntimeError(f"preview_self_heal:{reason}"[:300])
                if new_path and new_path != result.preview_path:
                    result.preview_path = new_path
                if reason.startswith("suspicious:"):
                    result._heal_status = "preview_suspicious"  # type: ignore[attr-defined]
            except RuntimeError:
                raise
            except Exception:
                pass
            return result

        # Jumbo/UNC TIFF: hard timeout zaten worker'da permanent.
        # retries=2 → 3×45s (~135s) worker kilidi; timeout'u tekrarlama.
        preview_timeout = max(float(self.timeout_sec), 45.0)
        risky = is_risky_path(path)
        try:
            result = run_with_timeout(
                _work,
                timeout_sec=preview_timeout,
                retries=0 if risky else self.retries,
                use_process=False,
                retry_on_timeout=False,
            )
        except TimeoutError:
            raise
        except MemoryError as exc:
            raise RuntimeError(f"v3_decode_memory:{exc}") from exc
        except OSError as exc:
            low = str(exc).lower()
            if any(
                t in low
                for t in (
                    "network",
                    "semaphore",
                    "winerror 53",
                    "winerror 64",
                    "winerror 59",
                    "errno 121",
                    "unreachable",
                )
            ):
                raise RuntimeError(f"v3_network_read:{exc}") from exc
            raise RuntimeError(f"v3_unreadable:{exc}") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"v3_decode_exception:{type(exc).__name__}:{exc}") from exc
        if not result.success or not local_artifact_exists(result.preview_path):
            raise RuntimeError("preview_verify_failed")
        heal_status = str(getattr(result, "_heal_status", "") or "") or "preview_ok"
        db.upsert_file({
            "path": path,
            "feature_preview_path": result.preview_path,
            "feature_preview_version": FEATURE_PREVIEW_VERSION,
            "preview_status": heal_status,
        })
        db.update_physical_readiness(
            job.file_id,
            thumbnail_ready=bool(local_artifact_exists(str(row.get("thumbnail_path") or ""))),
            preview_ready=True, requeue_missing=False,
        )
        return True

    def _process_light_bundle(
        self,
        db: Any,
        job: Job,
        row: dict[str, Any],
        path: str,
        art: Artifact,
    ) -> bool:
        mtime = float(row.get("mtime") or 0)

        # Ağ/UNC dosyaları ayrı process'te izole edilir.
        # ÖNEMLİ: Önceki stabil sürümde yerel TIFF normal extractor yolundan
        # geçiyordu ve 30 sn ağ zaman aşımı kullanıyordu. TIFF'leri topluca
        # "risky" sayıp 8 sn'ye düşürmek NAS'taki normal 3-7 sn okumaları
        # hataya çevirdi ve binlerce retry üretti. Bu davranışa geri dönüyoruz.
        try:
            from core.network_index_throttle import is_network_path
            use_isolated = bool(is_network_path(path))
        except Exception:
            use_isolated = path.startswith(("\\\\", "//"))

        light_timeout = float(
            getattr(
                self.settings,
                "light_decode_timeout_sec",
                getattr(self.settings, "network_timeout_sec", 30) or 30,
            )
            or getattr(self.settings, "network_timeout_sec", 30)
            or 30
        )
        light_timeout = max(5.0, light_timeout)

        if use_isolated:
            last_exc: BaseException | None = None
            # NAS jumbo TIFF: identical hard-timeout retry → N×light_timeout lock.
            # Keep one transient retry for non-timeout isolate failures only.
            max_attempts = max(1, self.retries + 1)
            for attempt in range(max_attempts):
                try:
                    result = _run_light_isolated(
                        self.settings,
                        path,
                        mtime,
                        timeout_sec=light_timeout,
                    )
                    last_exc = None
                    break
                except TimeoutError as exc:
                    last_exc = exc
                    break
                except Exception as exc:
                    last_exc = exc
                    if "timeout" in str(exc).lower():
                        break
                    if attempt + 1 >= max_attempts:
                        raise
            if last_exc is not None:
                raise last_exc
        else:

            def _work():
                return extract_light_assets_once(
                    path,
                    thumbnailer=self.thumbnailer,
                    feature_preview=self.feature_preview,
                    source_mtime=mtime,
                )

            result = run_with_timeout(
                _work,
                timeout_sec=self.timeout_sec,
                retries=self.retries,
                use_process=False,
                retry_on_timeout=False,
            )
        if not result.success:
            vs = str(getattr(result, "visual_status", "") or "")
            if vs in (
                "unsupported_visual",
                "renderer_missing",
                "empty_file",
                "corrupt_file",
            ):
                try:
                    db.upsert_file(
                        {
                            "path": path,
                            "preview_status": vs,
                            "unsupported_preview": 1
                            if vs
                            in ("unsupported_visual", "renderer_missing")
                            else 0,
                        }
                    )
                except Exception:
                    pass
            raise RuntimeError(result.error or "light_extract_failed")

        payload: dict[str, Any] = {"path": path}
        if result.thumbnail_path:
            payload["thumbnail_path"] = result.thumbnail_path
            payload["width"] = int(result.thumbnail_width or 0)
            payload["height"] = int(result.thumbnail_height or 0)
            payload["thumbnail_status"] = result.thumbnail_status
        if result.feature_preview_path:
            payload["feature_preview_path"] = result.feature_preview_path
            payload["feature_preview_version"] = FEATURE_PREVIEW_VERSION
            payload["preview_status"] = str(
                getattr(result, "visual_status", "") or "preview_ok"
            )
            if payload["preview_status"] == "visual_renderable":
                payload["preview_status"] = "preview_ok"
        if getattr(result, "visual_status", "") in (
            "unsupported_visual",
            "renderer_missing",
        ):
            payload["unsupported_preview"] = 1
        if result.metadata:
            payload["format_metadata"] = metadata_to_json(
                result.metadata, default="{}"
            )
            payload["metadata_status"] = result.metadata_status
            if not payload.get("width"):
                payload["width"] = int(result.metadata.get("width") or 0)
                payload["height"] = int(result.metadata.get("height") or 0)

        db.upsert_file(payload)

        # Fiziksel doğrulama — READY iddiası yalnız diskte varsa
        thumb_path = str(
            payload.get("thumbnail_path") or row.get("thumbnail_path") or ""
        ).strip()
        prev_path = str(
            payload.get("feature_preview_path")
            or row.get("feature_preview_path")
            or ""
        ).strip()
        thumb_ok = bool(thumb_path) and local_artifact_exists(thumb_path)
        prev_ok = bool(prev_path) and local_artifact_exists(prev_path)
        if art == Artifact.THUMBNAIL:
            if not thumb_ok:
                raise RuntimeError("thumb_verify_failed")
        elif art == Artifact.PREVIEW:
            if not prev_ok:
                raise RuntimeError("preview_verify_failed")
        elif art == Artifact.METADATA:
            if not (
                payload.get("format_metadata")
                or payload.get("width")
            ):
                raise RuntimeError("metadata_verify_failed")

        # SSOT: path yetmez — physical_*_ready=1 olmadan Tamamlanan artmaz
        db.update_physical_readiness(
            job.file_id,
            thumbnail_ready=thumb_ok,
            preview_ready=prev_ok,
            requeue_missing=False,
        )
        return True

    def _process_hash(
        self, db: Any, job: Job, row: dict[str, Any], path: str
    ) -> bool:
        feat = db.get_features(job.file_id) or {}
        # Mevcut phash: tekrar hesaplama yok. İçerik değişimi discovery'de
        # features satırını siler; burada kaynak TIFF/JPG okunmaz.
        if str(feat.get("phash") or "").strip():
            return True
        # Preview/thumb zorunlu — büyük TIFF full Pillow decode yok
        src = self._feature_source(row, path, require_cache=True)
        image = self._cached_feature_array(src)
        if image is None:
            raise RuntimeError("preview_required")

        def _work():
            ext = self._get_extractor(need_ai=False)
            extract_arr = getattr(ext, "extract_from_array", None)
            if extract_arr is not None:
                return extract_arr(
                    image,
                    include_patches=False,
                    filename=Path(path).name,
                    path=path,
                    deep_analysis=False,
                    compute={"hash", "color"},
                )
            return ext.extract_from_path(
                src,
                source_path=path,
                include_patches=False,
                deep_analysis=False,
                compute={"hash", "color"},
            )

        features = run_with_timeout(
            _work,
            timeout_sec=self.timeout_sec,
            retries=self.retries,
            use_process=False,
        )
        if not features.phash:
            # force hash-only path
            ext = self._get_extractor(need_ai=False)
            prev = ext.fast_hash_only
            ext.fast_hash_only = True
            try:
                extract_arr = getattr(ext, "extract_from_array", None)
                if extract_arr is not None:
                    features = extract_arr(
                        image,
                        filename=Path(path).name,
                        path=path,
                        include_patches=False,
                        deep_analysis=False,
                    )
                else:
                    features = ext.extract_from_path(
                        src,
                        source_path=path,
                        include_patches=False,
                        deep_analysis=False,
                    )
            finally:
                ext.fast_hash_only = prev
        if not features.phash:
            raise RuntimeError("hash_empty")

        db.upsert_features(
            job.file_id,
            {
                "phash": features.phash,
                "dhash": features.dhash,
                "whash": features.whash,
                "color_hist": features.color_hist or feat.get("color_hist"),
                "dominant_colors": features.dominant_colors
                or feat.get("dominant_colors")
                or [],
                "texture_features": feat.get("texture_features") or [],
                "dino_embedding": feat.get("dino_embedding"),
                "clip_embedding": feat.get("clip_embedding"),
                "patch_embeddings_meta": feat.get("patch_embeddings_meta") or [],
                "texture_map": feat.get("texture_map") or {},
            },
        )
        return True

    def _process_metadata(
        self, db: Any, job: Job, row: dict[str, Any], path: str
    ) -> bool:
        """Genel AI metadata: hazır Preview'dan okunur; kaynağa geri dönmez."""
        src = self._feature_source(row, path, require_cache=True)
        try:
            from PIL import Image
            t0 = time.perf_counter()
            with Image.open(src) as img:
                _note_file_read_ms((time.perf_counter() - t0) * 1000.0)
                meta = {
                    "width": int(img.width),
                    "height": int(img.height),
                    "mode": str(img.mode),
                    "image_format": str(img.format or ""),
                    "frames": int(getattr(img, "n_frames", 1) or 1),
                }
                dpi = img.info.get("dpi")
                if dpi:
                    meta["dpi"] = [float(x) for x in dpi]
        except Exception as exc:
            raise RuntimeError(f"metadata_preview_failed:{exc}") from exc
        db.upsert_file({
            "path": path,
            "format_metadata": json.dumps(meta, ensure_ascii=False),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "metadata_status": "preview_ok",
        })
        return True

    def _feature_source(
        self,
        row: dict[str, Any],
        path: str,
        *,
        require_cache: bool = False,
    ) -> str:
        """Return only a local artifact for analysis; never silently fall back to source."""
        prev = str(row.get("feature_preview_path") or "").strip()
        if prev and local_artifact_exists(prev):
            return prev
        for cache in getattr(self, "_preview_caches", [self.feature_preview]):
            hit = cache.get_existing(path)
            if hit.success and local_artifact_exists(hit.preview_path):
                return hit.preview_path
        thumb = str(row.get("thumbnail_path") or "").strip()
        if thumb and local_artifact_exists(thumb):
            return thumb
        if require_cache:
            raise RuntimeError("preview_required")
        return path

    def _process_vision(
        self,
        db: Any,
        job: Job,
        row: dict[str, Any],
        path: str,
        art: Artifact,
    ) -> bool:
        feat = db.get_features(job.file_id) or {}
        # Deep analiz hiçbir koşulda kaynak/orijinal dosyaya düşmez.
        # Preview fiziksel olarak hazır değilse planner/worker tekrar kuyruğa alır.
        src = self._feature_source(row, path, require_cache=True)
        if art == Artifact.TEXTURE:
            tex = feat.get("texture_features")
            if (
                isinstance(tex, list)
                and len(tex) > 0
                and str(feat.get("phash") or "").strip()
            ):
                return True
        if art == Artifact.PATCH:
            meta = feat.get("patch_embeddings_meta")
            if isinstance(meta, list) and len(meta) > 0:
                return True
        compute = {art.value}
        seed_dom = None
        seed_hist: bytes | None = None
        if art == Artifact.TEXTURE:
            if not str(feat.get("phash") or "").strip():
                compute.add("hash")
            seed_dom = feat.get("dominant_colors") or []
            raw_hist = feat.get("color_hist")
            if isinstance(raw_hist, (bytes, bytearray, memoryview)):
                seed_hist = bytes(raw_hist)
            if not seed_dom:
                compute.add("color")
                seed_dom = None
        if art == Artifact.PATCH:
            compute.add("patch")

        image = self._cached_feature_array(src)
        if image is None:
            raise RuntimeError("preview_required")

        def _work():
            ext = self._get_extractor(need_ai=True)
            extract_arr = getattr(ext, "extract_from_array", None)
            if extract_arr is not None:
                return extract_arr(
                    image,
                    filename=Path(path).name,
                    path=path,
                    include_patches=(art == Artifact.PATCH),
                    deep_analysis=True,
                    compute=compute,
                    seed_dominant=seed_dom,
                    seed_color_hist=seed_hist,
                )
            return ext.extract_from_path(
                src,
                source_path=path,
                include_patches=(art == Artifact.PATCH),
                deep_analysis=True,
                compute=compute,
            )

        features = run_with_timeout(
            _work,
            timeout_sec=max(self.timeout_sec, 60.0) if self.enable_ai else self.timeout_sec,
            retries=self.retries,
            use_process=False,
        )

        tm = parse_texture_map(feat.get("texture_map"))
        if features.texture_map:
            tm = {**tm, **dict(features.texture_map)}

        payload = {
            "phash": features.phash or feat.get("phash") or "",
            "dhash": features.dhash or feat.get("dhash") or "",
            "whash": features.whash or feat.get("whash") or "",
            "color_hist": features.color_hist or feat.get("color_hist"),
            "dominant_colors": features.dominant_colors
            or feat.get("dominant_colors")
            or [],
            "texture_features": features.texture_features
            or feat.get("texture_features")
            or [],
            "dino_embedding": feat.get("dino_embedding"),
            "clip_embedding": feat.get("clip_embedding"),
            "patch_embeddings_meta": feat.get("patch_embeddings_meta") or [],
            "texture_map": tm,
        }
        if art == Artifact.DINO:
            if not features.dino_embedding:
                raise RuntimeError("dino_empty")
            payload["dino_embedding"] = features.dino_embedding
        elif art == Artifact.CLIP:
            if not features.clip_embedding:
                raise RuntimeError("clip_empty")
            payload["clip_embedding"] = features.clip_embedding
            try:
                from core.index_object_evidence import attach_preview_object_evidence

                prev = str(row.get("feature_preview_path") or "").strip()
                if prev:
                    goi = attach_preview_object_evidence(
                        file_id=int(job.file_id),
                        preview_path=prev,
                        object_db_path=str(getattr(self.settings, "object_db_path", "") or ""),
                        min_confidence=float(
                            getattr(self.settings, "global_object_detection_min_confidence", 0.45) or 0.45
                        ),
                    )
                    if goi:
                        tm["global_object_intelligence"] = goi
                        if goi.get("visual_concept_dna"):
                            tm["visual_concept_dna"] = goi["visual_concept_dna"]
                        payload["texture_map"] = tm
            except Exception:
                pass
        elif art == Artifact.TEXTURE:
            if not payload["texture_features"] or not payload["phash"]:
                raise RuntimeError("texture_empty")
        elif art == Artifact.PATCH:
            meta = features.patch_embeddings_meta or []
            if not meta:
                raise RuntimeError("patch_empty")
            payload["patch_embeddings_meta"] = meta

        db.upsert_features(job.file_id, payload)
        # FAISS sync AI Final'i etkilemez; hata job FAIL etmez.
        if art in (Artifact.DINO, Artifact.CLIP):
            try:
                sync_file_embeddings(
                    db,
                    self.settings,
                    int(job.file_id),
                    kinds=(art.value,),
                )
            except Exception:
                pass
        return True

    def _get_ocr(self) -> OCREngine:
        enabled = bool(getattr(self.settings, "ocr_enabled", False))
        if self._ocr is None or bool(self._ocr.enabled) != enabled:
            langs = getattr(self.settings, "ocr_languages", None)
            self._ocr = OCREngine(
                enabled=enabled,
                languages=list(langs) if langs else None,
            )
        return self._ocr

    def _process_ocr(
        self,
        db: Any,
        job: Job,
        row: dict[str, Any],
        path: str,
    ) -> bool:
        """Preview/thumb üzerinden OCR. Boş metin = completed empty (processed=1)."""
        if not bool(getattr(self.settings, "ocr_enabled", False)):
            # Disabled iken job gelirse no-op complete (gap üretme).
            with db.connect() as conn:
                conn.execute(
                    "UPDATE files SET ocr_processed=1, updated_at=datetime('now') "
                    "WHERE id=?",
                    (int(job.file_id),),
                )
            return True

        try:
            src = self._feature_source(row, path, require_cache=True)
            engine = self._get_ocr()
            if not engine.enabled:
                # Motor yok — empty completed (failed değil)
                text = ""
            else:
                text = str(engine.extract_text(src) or "").strip()
        except Exception as exc:
            err = str(exc)
            if not err.startswith("ocr_failed:"):
                err = f"ocr_failed:{err}"
            # OCR hatası yalnız OCR state; Genel AI / files.error_msg dokunulmaz.
            with db.connect() as conn:
                conn.execute(
                    "UPDATE files SET ocr_processed=0, ocr_error=?, "
                    "updated_at=datetime('now') WHERE id=?",
                    (err[:500], int(job.file_id)),
                )
            raise RuntimeError(err) from exc

        with db.connect() as conn:
            conn.execute(
                "UPDATE files SET ocr_text=?, ocr_processed=1, ocr_error='', "
                "updated_at=datetime('now') WHERE id=?",
                (text, int(job.file_id)),
            )
        return True

    def _process_semantic(self, db: Any, job: Job) -> bool:
        from core.semantic_tags import build_semantic_tags

        row = db.get_file_by_id(job.file_id) or {}
        feat = db.get_features(job.file_id) or {}
        tm = parse_texture_map(feat.get("texture_map"))
        tm["semantic_tags"] = build_semantic_tags(
            tm,
            category_path=str(row.get("category_path") or ""),
            ocr_text=str(row.get("ocr_text") or ""),
            filename=str(row.get("filename") or ""),
        )
        db.upsert_features(
            job.file_id,
            {
                "phash": feat.get("phash") or "",
                "dhash": feat.get("dhash") or "",
                "whash": feat.get("whash") or "",
                "color_hist": feat.get("color_hist"),
                "dominant_colors": feat.get("dominant_colors") or [],
                "texture_features": feat.get("texture_features") or [],
                "dino_embedding": feat.get("dino_embedding"),
                "clip_embedding": feat.get("clip_embedding"),
                "patch_embeddings_meta": feat.get("patch_embeddings_meta") or [],
                "texture_map": tm,
            },
        )
        return True

    def _process_dna(self, db: Any, job: Job) -> bool:
        from core.pattern_dna import apply_dna_to_texture_map, build_pattern_dna

        row = db.get_file_by_id(job.file_id) or {}
        feat = db.get_features(job.file_id) or {}
        tm = parse_texture_map(feat.get("texture_map"))
        dna = build_pattern_dna(
            tm,
            semantic_tags=tm.get("semantic_tags"),
            category_path=str(row.get("category_path") or ""),
            source="index_v3",
        )
        tm = apply_dna_to_texture_map(tm, dna)
        db.upsert_features(
            job.file_id,
            {
                "phash": feat.get("phash") or "",
                "dhash": feat.get("dhash") or "",
                "whash": feat.get("whash") or "",
                "color_hist": feat.get("color_hist"),
                "dominant_colors": feat.get("dominant_colors") or [],
                "texture_features": feat.get("texture_features") or [],
                "dino_embedding": feat.get("dino_embedding"),
                "clip_embedding": feat.get("clip_embedding"),
                "patch_embeddings_meta": feat.get("patch_embeddings_meta") or [],
                "texture_map": tm,
            },
        )
        return True

    def _process_object_concept(self, db: Any, job: Job, row: dict[str, Any]) -> bool:
        """Mevcut Feature Preview → RT-DETR. Preview üretmez, NAS/TIFF okumaz."""
        prev = str(row.get("feature_preview_path") or "").strip()
        if not prev or not local_artifact_exists(prev):
            raise RuntimeError("preview_required")
        from core.index_object_evidence import attach_preview_object_evidence
        from core.index_v3.artifact_state import _visual_concept_dna_ok
        from core.object_index import ObjectIndexStore
        from core.visual_concept_dna_backfill import persist_goi_texture_map

        feat = db.get_features(int(job.file_id)) or {}
        tm = parse_texture_map(feat.get("texture_map"))
        if _visual_concept_dna_ok(tm):
            return True
        obj_path = str(getattr(self.settings, "object_db_path", "") or "")
        force = False
        try:
            force = bool(obj_path) and ObjectIndexStore(obj_path, readonly=True).has_scan(
                int(job.file_id)
            )
        except Exception:
            force = False
        goi = attach_preview_object_evidence(
            file_id=int(job.file_id),
            preview_path=prev,
            object_db_path=obj_path,
            min_confidence=float(
                getattr(self.settings, "global_object_detection_min_confidence", 0.45)
                or 0.45
            ),
            force=force,
        )
        if not goi:
            raise RuntimeError("preview_required")
        persist_goi_texture_map(db, int(job.file_id), goi)
        return True

    def _owlv2_clip_fn(self) -> Any:
        cached = getattr(self, "_owlv2_clip", None)
        if cached is not None or getattr(self, "_owlv2_clip_tried", False):
            return getattr(self, "_owlv2_clip", None)
        self._owlv2_clip_tried = True
        if not self.enable_ai:
            self._owlv2_clip = None
            return None
        try:
            from core.owlv2_index_probe import default_clip_fn
            from core.search_evidence_gate import _ZS_PROMPT

            fe = self._get_extractor(need_ai=True)
            prompts = {
                **_ZS_PROMPT,
                "floral": "a repeating floral textile print, not a real fruit",
                "animal_print": "an animal print textile, not a real animal",
                "textile": "a repeating decorative fabric pattern, not a real object",
                "necklace": _ZS_PROMPT["jewelry"],
            }
            text_emb = {k: fe.embed_text(p) for k, p in prompts.items()}
            self._owlv2_clip = default_clip_fn(fe, text_emb)
        except Exception:
            self._owlv2_clip = None
        return self._owlv2_clip

    def _process_owlv2(self, db: Any, job: Job, row: dict[str, Any]) -> bool:
        """OWLv2 on existing feature Preview. Does not rasterize the original file."""
        from core.index_analysis_guards import local_artifact_exists
        from core.index_freeze import (
            INDEX_FROZEN,
            allow_index_writes,
            process_search_active,
        )
        from core.index_v3.artifact_state import object_index_path_for_patterns_db
        from core.object_index import ObjectIndexStore
        from core.ovd_index import index_open_vocab_image

        if INDEX_FROZEN and process_search_active():
            raise RuntimeError("search_session")
        preview = str(row.get("feature_preview_path") or "").strip()
        if not preview or not local_artifact_exists(preview):
            raise RuntimeError("preview_required")
        feat = db.get_features(int(job.file_id)) or {}
        tm = parse_texture_map(feat.get("texture_map"))
        family = str(row.get("pattern_family") or "")
        if not family and isinstance(tm, dict):
            family = str((tm.get("pattern_dna") or {}).get("motif_family") or "")
        obj_path = object_index_path_for_patterns_db(db, self.settings)
        if not obj_path:
            return True
        store = ObjectIndexStore(obj_path)
        backend = getattr(self, "_owlv2_backend", None)
        clip_fn = getattr(self, "_owlv2_clip_fn_override", None)
        if clip_fn is None:
            clip_fn = self._owlv2_clip_fn()
        with allow_index_writes():
            out = index_open_vocab_image(
                file_id=int(job.file_id),
                image_path=preview,
                store=store,
                backend=backend,
                pattern_family=family,
                texture_map=tm,
                clip_fn=clip_fn,
                preview_version=str(row.get("feature_preview_version") or ""),
                archive_path=str(row.get("path") or job.path or ""),
            )
        if out.get("skipped") == "frozen_or_search":
            raise RuntimeError("search_session")
        if out.get("skipped") == "preview_required":
            raise RuntimeError("preview_required")
        return True
