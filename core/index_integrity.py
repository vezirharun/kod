"""Index Integrity Engine — indexed dosyaların arama için gerçekten hazır olup olmadığını doğrular.

Search Engine / Ranking / AI skor algoritmalarına dokunmaz.
Sadece index bütünlüğü, queue DONE doğrulaması ve force-reindex.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.db import Database
from core.index_stages import PENDING_FULL, PENDING_LIGHT
from core.logger import setup_logger
from core.settings import AppSettings

logger = setup_logger(__name__)

STATUS_INCOMPLETE = "incomplete"
STATUS_INDEXED = "indexed"
STATUS_PROCESSING = "processing"

# Görev tipi listesi (queue manager ile uyumlu; circular import önlemek için burada)
TASK_TYPES = (
    "thumbnail",
    "feature_preview",
    "hash",
    "patch",
    "texture_map",
    "taxonomy",
    "ai_embedding",
    "ocr",
)

# Görev tipi → bütünlük alanı eşlemesi
TASK_TO_CHECKS: dict[str, tuple[str, ...]] = {
    "thumbnail": ("thumbnail",),
    "feature_preview": ("thumbnail",),  # preview opsiyonel; thumb zorunlu
    "hash": ("phash", "dhash", "features_row"),
    "patch": ("patches", "features_row"),
    "texture_map": ("texture", "features_row"),
    "taxonomy": ("pattern_dna",),
    "ai_embedding": ("dino", "clip"),
    "ocr": (),  # OCR opsiyonel — settings'e bağlı
}


@dataclass
class IntegrityFlags:
    features_row: bool = False
    thumbnail: bool = False
    phash: bool = False
    dhash: bool = False
    texture: bool = False
    dino: bool = False
    clip: bool = False
    pattern_dna: bool = False
    patches: bool = False

    def missing(self, *, require_ai: bool = True) -> list[str]:
        needed = [
            ("features_row", self.features_row),
            ("thumbnail", self.thumbnail),
            ("phash", self.phash),
            ("dhash", self.dhash),
            ("texture", self.texture),
            ("pattern_dna", self.pattern_dna),
            ("patches", self.patches),
        ]
        if require_ai:
            needed.extend(
                [
                    ("dino", self.dino),
                    ("clip", self.clip),
                ]
            )
        return [name for name, ok in needed if not ok]

    def is_search_ready(self, *, require_ai: bool = True) -> bool:
        return not self.missing(require_ai=require_ai)


@dataclass
class FileIntegrityReport:
    file_id: int
    filename: str = ""
    path: str = ""
    status: str = ""
    flags: IntegrityFlags = field(default_factory=IntegrityFlags)
    missing: list[str] = field(default_factory=list)
    false_done_stages: list[str] = field(default_factory=list)
    search_ready: bool = False


@dataclass
class DoctorReport:
    timestamp: str
    total_indexed: int = 0
    total_incomplete: int = 0
    feature_missing: int = 0
    thumbnail_missing: int = 0
    hash_missing: int = 0
    embedding_missing: int = 0
    dna_missing: int = 0
    false_done: int = 0
    auto_fixed: int = 0
    requeued: int = 0
    marked_incomplete: int = 0
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _thumb_exists(path: str, cache_dir: str = "") -> bool:
    raw = str(path or "").strip()
    if not raw:
        return False
    p = Path(raw)
    if p.is_file():
        return True
    if cache_dir and not p.is_absolute():
        cand = Path(cache_dir) / raw
        if cand.is_file():
            return True
        # relative like cache\thumbnails\x.webp from project root
        root = Path(cache_dir).resolve().parent
        cand2 = root / raw
        if cand2.is_file():
            return True
    return os.path.isfile(raw)


def _parse_tm(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def evaluate_row(
    row: dict[str, Any],
    *,
    require_ai: bool = True,
    cache_dir: str = "",
) -> IntegrityFlags:
    """Tek dosya satırından IntegrityFlags üret."""
    flags = IntegrityFlags()
    has_feat = row.get("feat_id") is not None or row.get("features_present")
    # JOIN ile fe.file_id veya explicit
    if row.get("feat_file_id") is not None:
        has_feat = True
    if "phash" in row or "dino_len" in row or "feat_file_id" in row:
        has_feat = row.get("feat_file_id") is not None or bool(
            str(row.get("phash") or "").strip()
        ) or int(row.get("dino_len") or 0) > 0 or int(row.get("clip_len") or 0) > 0 or int(
            row.get("tmap_len") or 0
        ) > 0
    # More reliable: feat_file_id from LEFT JOIN
    if "feat_file_id" in row:
        has_feat = row.get("feat_file_id") is not None

    flags.features_row = bool(has_feat)
    flags.thumbnail = _thumb_exists(str(row.get("thumbnail_path") or ""), cache_dir)
    flags.phash = bool(str(row.get("phash") or "").strip())
    flags.dhash = bool(str(row.get("dhash") or "").strip())
    tm = _parse_tm(row.get("texture_map"))
    flags.texture = bool(tm) and (
        int(row.get("texture_version") or 0) > 0
        or bool(tm.get("pattern_family"))
        or bool(tm.get("color_family"))
        or bool(tm.get("pattern_dna"))
        or len(tm) > 2
    )
    flags.dino = int(row.get("dino_len") or 0) > 0 or bool(row.get("dino_embedding"))
    flags.clip = int(row.get("clip_len") or 0) > 0 or bool(row.get("clip_embedding"))
    dna = tm.get("pattern_dna") if isinstance(tm, dict) else None
    flags.pattern_dna = isinstance(dna, dict) and bool(dna)
    # patch_embeddings kolonu JSON meta tutar; "[]" / boş = eksik
    patch_raw = row.get("patch_embeddings")
    patch_len = int(row.get("patch_len") or 0)
    if patch_len > 0:
        flags.patches = patch_len > 2  # "[]" length 2
    elif isinstance(patch_raw, list):
        flags.patches = len(patch_raw) > 0
    elif isinstance(patch_raw, str):
        s = patch_raw.strip()
        flags.patches = bool(s) and s not in ("[]", "{}", "null", "None")
    else:
        flags.patches = False
    return flags


def _fetch_file_integrity_row(db: Database, file_id: int) -> dict[str, Any] | None:
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT f.id, f.filename, f.path, f.status, f.source_id,
                   f.thumbnail_path, f.feature_version, f.texture_version,
                   fe.file_id AS feat_file_id,
                   COALESCE(fe.phash, '') AS phash,
                   COALESCE(fe.dhash, '') AS dhash,
                   fe.texture_map AS texture_map,
                   fe.patch_embeddings AS patch_embeddings,
                   length(COALESCE(fe.patch_embeddings, '')) AS patch_len,
                   length(COALESCE(fe.dino_embedding, X'')) AS dino_len,
                   length(COALESCE(fe.clip_embedding, X'')) AS clip_len,
                   length(COALESCE(fe.texture_map, '')) AS tmap_len
            FROM files f
            LEFT JOIN features fe ON fe.file_id = f.id
            WHERE f.id = ?
            """,
            (int(file_id),),
        ).fetchone()
    return dict(row) if row else None


def check_file(
    db: Database,
    file_id: int,
    *,
    settings: AppSettings | None = None,
) -> FileIntegrityReport:
    settings = settings or AppSettings.load()
    require_ai = bool(getattr(settings, "ai_embedding_enabled", True))
    cache_dir = str(getattr(settings, "cache_dir", "") or "")
    row = _fetch_file_integrity_row(db, file_id)
    if not row:
        return FileIntegrityReport(
            file_id=int(file_id),
            missing=["file_not_found"],
            search_ready=False,
        )
    flags = evaluate_row(row, require_ai=require_ai, cache_dir=cache_dir)
    missing = flags.missing(require_ai=require_ai)
    false_done = list_false_done_stages(db, file_id, flags, require_ai=require_ai)
    return FileIntegrityReport(
        file_id=int(file_id),
        filename=str(row.get("filename") or ""),
        path=str(row.get("path") or ""),
        status=str(row.get("status") or ""),
        flags=flags,
        missing=missing,
        false_done_stages=false_done,
        search_ready=flags.is_search_ready(require_ai=require_ai),
    )


def list_false_done_stages(
    db: Database,
    file_id: int,
    flags: IntegrityFlags,
    *,
    require_ai: bool = True,
) -> list[str]:
    """Queue'da done yazılmış ama artifact yok olan stage'ler."""
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT COALESCE(task_type, stage) AS task
            FROM index_queue
            WHERE file_id=? AND status='done'
            """,
            (int(file_id),),
        ).fetchall()
    false: list[str] = []
    flag_map = {
        "thumbnail": flags.thumbnail,
        "phash": flags.phash,
        "dhash": flags.dhash,
        "features_row": flags.features_row,
        "texture": flags.texture,
        "pattern_dna": flags.pattern_dna,
        "dino": flags.dino,
        "clip": flags.clip,
        "patches": flags.patches,
    }
    for row in rows:
        task = str(row["task"] or "")
        checks = TASK_TO_CHECKS.get(task, ())
        if task == "ai_embedding" and not require_ai:
            continue
        for c in checks:
            if c in flag_map and not flag_map[c]:
                false.append(task)
                break
    return sorted(set(false))


def task_artifact_ok(
    db: Database,
    file_id: int,
    task_type: str,
    *,
    settings: AppSettings | None = None,
) -> bool:
    """Tek bir queue stage'inin DONE olabilmesi için artifact kontrolü."""
    settings = settings or AppSettings.load()
    require_ai = bool(getattr(settings, "ai_embedding_enabled", True))
    if task_type == "ocr":
        if not getattr(settings, "ocr_enabled", False):
            return True
        with db.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(ocr_text,'') AS ocr FROM files WHERE id=?",
                (int(file_id),),
            ).fetchone()
        return bool(str(row["ocr"] if row else "").strip())
    if task_type == "ai_embedding" and not require_ai:
        return True

    report = check_file(db, file_id, settings=settings)
    checks = TASK_TO_CHECKS.get(task_type, ())
    if not checks:
        return True
    flag_map = asdict(report.flags)
    return all(flag_map.get(c, False) for c in checks)


def mark_incomplete(db: Database, file_id: int, *, reason: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE files SET status=?, error_msg=?, updated_at=?,
                   index_stage=?
            WHERE id=?
            """,
            (
                STATUS_INCOMPLETE,
                (reason or "index_integrity_incomplete")[:500],
                now,
                PENDING_LIGHT,
                int(file_id),
            ),
        )


def invalidate_false_done_queue(
    db: Database,
    file_id: int,
    stages: list[str] | None = None,
) -> int:
    """False-done queue satırlarını pending'e çek."""
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    with db.connect() as conn:
        if stages:
            for stage in stages:
                cur = conn.execute(
                    """
                    UPDATE index_queue
                    SET status='pending', error_message=?, last_error=?,
                        updated_at=?, attempts=0, retry_count=0
                    WHERE file_id=? AND status='done'
                      AND (task_type=? OR stage=?)
                    """,
                    (
                        "false_done_reopened",
                        "false_done_reopened",
                        now,
                        int(file_id),
                        stage,
                        stage,
                    ),
                )
                n += cur.rowcount
        else:
            cur = conn.execute(
                """
                UPDATE index_queue
                SET status='pending', error_message=?, last_error=?,
                    updated_at=?, attempts=0, retry_count=0
                WHERE file_id=? AND status='done'
                """,
                ("false_done_reopened", "false_done_reopened", now, int(file_id)),
            )
            n += cur.rowcount
    return int(n)


def force_reindex_file(
    db: Database,
    file_id: int,
    *,
    settings: AppSettings | None = None,
    include_ai: bool | None = None,
    run_indexer: bool = False,
) -> dict[str, Any]:
    """Eksik feature'ları kuyruğa al; false-done olsa bile yeniden üret.

    run_indexer=True ise büyük TIFF defer'ı geçici kapatılarak light+deep process denenir.
    """
    settings = settings or AppSettings.load()
    report = check_file(db, file_id, settings=settings)
    if not report.path:
        return {"ok": False, "reason": "file_not_found", "file_id": file_id}

    require_ai = bool(getattr(settings, "ai_embedding_enabled", True))
    if include_ai is None:
        include_ai = require_ai

    # Hangi görevler gerekli?
    missing = set(report.missing)
    tasks: list[str] = []
    if "thumbnail" in missing:
        tasks.extend(["thumbnail", "feature_preview"])
    if "phash" in missing or "dhash" in missing or "features_row" in missing:
        tasks.extend(["hash", "patch"])
    if "patches" in missing:
        tasks.append("patch")
    if "texture" in missing or "features_row" in missing:
        tasks.append("texture_map")
    if "pattern_dna" in missing:
        tasks.append("taxonomy")
    if include_ai and ("dino" in missing or "clip" in missing):
        tasks.append("ai_embedding")
    if not tasks:
        # Hiçbir şey eksik değilse ama false-done varsa veya kullanıcı force istedi
        tasks = ["thumbnail", "feature_preview", "hash", "patch", "texture_map", "taxonomy"]
        if include_ai:
            tasks.append("ai_embedding")

    # unique preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tasks:
        if t not in seen and t in TASK_TYPES:
            seen.add(t)
            ordered.append(t)

    # done satırları da varsa yeniden üretmek için reopen + pending insert
    stages_to_reopen = list(set(ordered) | set(report.false_done_stages))
    reopened = invalidate_false_done_queue(db, file_id, stages_to_reopen)

    row = _fetch_file_integrity_row(db, file_id) or {}
    source_id = int(row.get("source_id") or 0)
    path = str(row.get("path") or report.path)

    mark_incomplete(
        db,
        file_id,
        reason=f"force_reindex missing={','.join(report.missing) or 'none'}",
    )
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE files SET status=?, index_stage=?, updated_at=?
            WHERE id=?
            """,
            (
                STATUS_PROCESSING,
                PENDING_FULL if include_ai else PENDING_LIGHT,
                datetime.now(timezone.utc).isoformat(),
                int(file_id),
            ),
        )

    # index_queue yazılmaz — tek kuyruk SSOT (light/heavy_status)
    _ = (source_id, path, ordered)

    result: dict[str, Any] = {
        "ok": True,
        "file_id": file_id,
        "filename": report.filename,
        "missing_before": report.missing,
        "false_done_reopened": reopened,
        "tasks_queued": [],
        "status": STATUS_PROCESSING,
    }

    if run_indexer and path and os.path.isfile(path):
        result["indexer"] = _run_integrity_indexer(db, file_id, path, settings)

    return result


def _run_integrity_indexer(
    db: Database,
    file_id: int,
    path: str,
    settings: AppSettings,
) -> dict[str, Any]:
    """Büyük TIFF defer'ını geçici kapatıp light (+ gerekirse deep) çalıştır."""
    from core.indexer import Indexer

    # Ayarları kopyalama: AppSettings dataclass olabilir — attribute override
    prev_defer = getattr(settings, "fast_tif_defer_mb", 64)
    prev_skip = getattr(settings, "index_skip_heavy_formats", True)
    try:
        settings.fast_tif_defer_mb = 10**9
        settings.index_skip_heavy_formats = False
        ix = Indexer(settings)
        stats_light: dict[str, Any] = {"processed": 0, "errors": 0}
        ok_light = ix._process_file(
            file_id,
            path,
            Path(path).name,
            stats_light,
            deep_scan=False,
            light_pass=True,
        )
        after_light = check_file(db, file_id, settings=settings)
        if not after_light.flags.patches:
            patch_res = ix.backfill_patches_for_file(file_id)
            after_light = check_file(db, file_id, settings=settings)
        else:
            patch_res = {"skipped": True}
        ok_deep = False
        stats_deep: dict[str, Any] = {"processed": 0, "errors": 0}
        if after_light.flags.thumbnail and not after_light.search_ready:
            ok_deep = ix._process_file(
                file_id,
                path,
                Path(path).name,
                stats_deep,
                deep_scan=True,
                light_pass=False,
            )
        final = check_file(db, file_id, settings=settings)
        if final.search_ready:
            finalize_file_status(db, file_id, settings=settings)
        return {
            "light_ok": bool(ok_light),
            "deep_ok": bool(ok_deep),
            "patch_backfill": patch_res,
            "search_ready": final.search_ready,
            "missing": final.missing,
            "status": final.status,
        }
    except Exception as exc:
        logger.exception("integrity indexer failed file_id=%s", file_id)
        return {"error": str(exc)}
    finally:
        settings.fast_tif_defer_mb = prev_defer
        settings.index_skip_heavy_formats = prev_skip


def force_reindex_missing_patches(
    db: Database,
    *,
    settings: AppSettings | None = None,
    limit: int = 500,
    file_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Patch meta'sı boş indexed dosyaları otomatik backfill / requeue.

    Önce thumbnail üzerinden patch üretir (orijinal açmaz).
    Üretemezse patch görevini kuyruğa alır.
    """
    settings = settings or AppSettings.load()
    from core.indexer import Indexer

    targets: list[int] = []
    if file_ids:
        targets = [int(x) for x in file_ids]
    else:
        with db.connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id
                FROM files f
                LEFT JOIN features fe ON fe.file_id = f.id
                WHERE f.status IN ('indexed', 'incomplete', 'processing')
                  AND f.thumbnail_path IS NOT NULL AND f.thumbnail_path != ''
                  AND (
                    fe.file_id IS NULL
                    OR fe.patch_embeddings IS NULL
                    OR fe.patch_embeddings = ''
                    OR fe.patch_embeddings = '[]'
                  )
                ORDER BY f.id
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        targets = [int(r["id"]) for r in rows]

    ix = Indexer(settings)
    fixed = 0
    queued = 0
    failed = 0
    details: list[dict[str, Any]] = []
    for fid in targets:
        result = ix.backfill_patches_for_file(fid)
        if result.get("ok") and int(result.get("patch_count") or 0) > 0:
            fixed += 1
            # patch task done
            now = datetime.now(timezone.utc).isoformat()
            with db.connect() as conn:
                conn.execute(
                    """
                    UPDATE index_queue
                    SET status='done', error_message='', last_error='', updated_at=?
                    WHERE file_id=? AND (task_type='patch' OR stage='patch')
                    """,
                    (now, int(fid)),
                )
            finalize_file_status(db, fid, settings=settings)
            details.append({"file_id": fid, "status": "patched", **result})
            continue
        # backfill olmadı → force reindex patch
        fr = force_reindex_file(db, fid, settings=settings, include_ai=False)
        if fr.get("ok"):
            queued += 1
            details.append({"file_id": fid, "status": "queued", **fr})
        else:
            failed += 1
            details.append({"file_id": fid, "status": "failed", **result, **fr})

    return {
        "ok": True,
        "scanned": len(targets),
        "fixed": fixed,
        "queued": queued,
        "failed": failed,
        "details": details[:50],
    }


def scan_integrity(
    db: Database,
    *,
    settings: AppSettings | None = None,
    limit: int = 0,
    statuses: tuple[str, ...] = (STATUS_INDEXED, STATUS_INCOMPLETE),
) -> list[FileIntegrityReport]:
    settings = settings or AppSettings.load()
    require_ai = bool(getattr(settings, "ai_embedding_enabled", True))
    cache_dir = str(getattr(settings, "cache_dir", "") or "")
    placeholders = ",".join("?" for _ in statuses)
    sql = f"""
        SELECT f.id, f.filename, f.path, f.status, f.source_id,
               f.thumbnail_path, f.feature_version, f.texture_version,
               fe.file_id AS feat_file_id,
               COALESCE(fe.phash, '') AS phash,
               COALESCE(fe.dhash, '') AS dhash,
               fe.texture_map AS texture_map,
               fe.patch_embeddings AS patch_embeddings,
               length(COALESCE(fe.patch_embeddings, '')) AS patch_len,
               length(COALESCE(fe.dino_embedding, X'')) AS dino_len,
               length(COALESCE(fe.clip_embedding, X'')) AS clip_len,
               length(COALESCE(fe.texture_map, '')) AS tmap_len
        FROM files f
        LEFT JOIN features fe ON fe.file_id = f.id
        WHERE f.status IN ({placeholders})
        ORDER BY f.id
    """
    params: list[Any] = list(statuses)
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    out: list[FileIntegrityReport] = []
    for row in rows:
        flags = evaluate_row(row, require_ai=require_ai, cache_dir=cache_dir)
        missing = flags.missing(require_ai=require_ai)
        out.append(
            FileIntegrityReport(
                file_id=int(row["id"]),
                filename=str(row.get("filename") or ""),
                path=str(row.get("path") or ""),
                status=str(row.get("status") or ""),
                flags=flags,
                missing=missing,
                search_ready=flags.is_search_ready(require_ai=require_ai),
            )
        )
    return out


def run_doctor(
    settings: AppSettings | None = None,
    *,
    limit: int = 0,
    repair: bool = False,
    repair_limit: int = 500,
) -> DoctorReport:
    """Tüm DB'yi tara; rapor üret; isteğe bağlı otomatik düzelt."""
    settings = settings or AppSettings.load()
    db = Database(settings.db_path)
    require_ai = bool(getattr(settings, "ai_embedding_enabled", True))

    reports = scan_integrity(db, settings=settings, limit=limit)
    # false-done sayımı (örneklem — her incomplete için)
    false_done = 0
    samples: dict[str, list[dict[str, Any]]] = {
        "feature_missing": [],
        "thumbnail_missing": [],
        "hash_missing": [],
        "embedding_missing": [],
        "dna_missing": [],
        "false_done": [],
    }

    feature_missing = thumbnail_missing = hash_missing = embedding_missing = dna_missing = 0
    incomplete_ids: list[int] = []

    for r in reports:
        if r.status == STATUS_INDEXED and not r.search_ready:
            incomplete_ids.append(r.file_id)
        if "features_row" in r.missing:
            feature_missing += 1
            if len(samples["feature_missing"]) < 8:
                samples["feature_missing"].append(
                    {"id": r.file_id, "filename": r.filename}
                )
        if "thumbnail" in r.missing:
            thumbnail_missing += 1
            if len(samples["thumbnail_missing"]) < 8:
                samples["thumbnail_missing"].append(
                    {"id": r.file_id, "filename": r.filename}
                )
        if "phash" in r.missing or "dhash" in r.missing:
            hash_missing += 1
            if len(samples["hash_missing"]) < 8:
                samples["hash_missing"].append(
                    {"id": r.file_id, "filename": r.filename}
                )
        if require_ai and ("dino" in r.missing or "clip" in r.missing):
            embedding_missing += 1
            if len(samples["embedding_missing"]) < 8:
                samples["embedding_missing"].append(
                    {"id": r.file_id, "filename": r.filename}
                )
        if "pattern_dna" in r.missing:
            dna_missing += 1
            if len(samples["dna_missing"]) < 8:
                samples["dna_missing"].append(
                    {"id": r.file_id, "filename": r.filename}
                )

    # False done: indexed ama search_ready değil olanlardan örnek kontrol
    for fid in incomplete_ids[:2000]:
        full = check_file(db, fid, settings=settings)
        if full.false_done_stages:
            false_done += 1
            if len(samples["false_done"]) < 8:
                samples["false_done"].append(
                    {
                        "id": fid,
                        "filename": full.filename,
                        "stages": full.false_done_stages,
                    }
                )

    marked = 0
    requeued = 0
    if repair:
        for fid in incomplete_ids[: max(0, int(repair_limit))]:
            mark_incomplete(db, fid, reason="index_doctor_incomplete")
            marked += 1
            result = force_reindex_file(db, fid, settings=settings)
            if result.get("ok"):
                requeued += 1

    with db.connect() as conn:
        total_indexed = conn.execute(
            "SELECT COUNT(*) c FROM files WHERE status='indexed'"
        ).fetchone()["c"]
        total_incomplete = conn.execute(
            "SELECT COUNT(*) c FROM files WHERE status=?",
            (STATUS_INCOMPLETE,),
        ).fetchone()["c"]

    return DoctorReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_indexed=int(total_indexed),
        total_incomplete=int(total_incomplete),
        feature_missing=feature_missing,
        thumbnail_missing=thumbnail_missing,
        hash_missing=hash_missing,
        embedding_missing=embedding_missing,
        dna_missing=dna_missing,
        false_done=false_done,
        auto_fixed=requeued,
        requeued=requeued,
        marked_incomplete=marked,
        samples=samples,
    )


def validate_before_queue_done(
    db: Database,
    file_id: int,
    task_type: str,
    *,
    settings: AppSettings | None = None,
) -> tuple[bool, str]:
    """Queue DONE yazılmadan önce çağrılır."""
    ok = task_artifact_ok(db, file_id, task_type, settings=settings)
    if ok:
        return True, ""
    return False, f"artifact_missing_for_task:{task_type}"


def finalize_file_status(
    db: Database,
    file_id: int,
    *,
    settings: AppSettings | None = None,
) -> str:
    """Tüm queue işleri bitince indexed veya incomplete yaz."""
    settings = settings or AppSettings.load()
    report = check_file(db, file_id, settings=settings)
    now = datetime.now(timezone.utc).isoformat()
    if report.search_ready:
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE files SET status=?, indexed_at=?, error_msg='', updated_at=?
                WHERE id=?
                """,
                (STATUS_INDEXED, now, now, int(file_id)),
            )
        return STATUS_INDEXED
    mark_incomplete(
        db,
        file_id,
        reason=f"incomplete:{','.join(report.missing)}",
    )
    return STATUS_INCOMPLETE
