"""Index SSOT — tek doğruluk kaynağı: files.light_status / files.heavy_status.

Hızlı Index:  light_status='pending'  → bitince 'done'
Genel AI:     light='done' AND heavy='pending' → bitince heavy='done'

Sayaçlar ve job listeleri yalnızca bu alanlardan. index_queue / index_stage
ana ilerlemeyi belirlemez.
"""

from __future__ import annotations

from typing import Any

from core.index_file_status import WORKER_HEAVY, WORKER_LIGHT
from core.index_stages import LIGHT_DONE

Job = tuple[int, str, str]

_ACTIVE_FILE = "status NOT IN ('excluded_internal','missing')"


def count_lanes(
    db: Any, source_ids: list[int] | None = None
) -> dict[str, int]:
    """Ana kart / resume için kalıcı lane sayıları.

    Kimlik:
      light_done + light_pending + light_processing + light_failed == total
      heavy_queue_display = heavy_ready_pending + heavy_processing
        (WAITING_PREVIEW dahil değil — executable General AI kuyruğu)
      waiting_preview = light_done ∧ heavy pending ∧ needs_medium_preview=1

    source_ids verilirse yalnız o kaynak(lar) sayılır.
    """
    ids = [int(x) for x in (source_ids or []) if int(x) > 0]
    extra = ""
    params: tuple[int, ...] = ()
    if ids:
        ph = ",".join("?" * len(ids))
        extra = f" AND source_id IN ({ph})"
        params = tuple(ids)
    with db.connect() as conn:
        row = conn.execute(
            f"""
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN light_status='pending' THEN 1 ELSE 0 END) AS light_pending,
              SUM(CASE WHEN light_status='processing' THEN 1 ELSE 0 END) AS light_processing,
              SUM(CASE WHEN light_status='done' THEN 1 ELSE 0 END) AS light_done,
              SUM(CASE WHEN light_status='failed' THEN 1 ELSE 0 END) AS light_failed,
              SUM(CASE WHEN heavy_status='pending' THEN 1 ELSE 0 END) AS heavy_pending,
              SUM(CASE WHEN heavy_status='processing' THEN 1 ELSE 0 END) AS heavy_processing,
              SUM(CASE WHEN heavy_status='done' THEN 1 ELSE 0 END) AS heavy_done,
              SUM(CASE WHEN heavy_status='failed' THEN 1 ELSE 0 END) AS heavy_failed,
              SUM(CASE WHEN light_status='done'
                        AND coalesce(nullif(heavy_status,''),'pending')='pending'
                        AND coalesce(needs_medium_preview,0)=0
                   THEN 1 ELSE 0 END) AS heavy_ready_pending,
              SUM(CASE WHEN light_status='done'
                        AND coalesce(nullif(heavy_status,''),'pending')='pending'
                        AND coalesce(needs_medium_preview,0)=1
                   THEN 1 ELSE 0 END) AS waiting_preview
            FROM files
            WHERE {_ACTIVE_FILE}{extra}
            """,
            params,
        ).fetchone()
    out = {k: int(row[k] or 0) for k in row.keys()}
    out["light_queue_display"] = out["light_pending"] + out["light_processing"]
    # Executable General AI kuyruğu — residual (total-done) DEĞİL
    out["heavy_queue_display"] = (
        out["heavy_ready_pending"] + out["heavy_processing"]
    )
    return out


def archive_content_changed(
    existing: dict | None,
    file_size: int,
    mtime: float,
) -> bool:
    """Güncelle/envanter: yalnız yeni, missing, size/mtime — done'u status tuzağıyla bozmaz."""
    if existing is None:
        return True
    if str(existing.get("status") or "") == "missing":
        return True
    if int(existing.get("file_size") or 0) != int(file_size or 0):
        return True
    if abs(float(existing.get("mtime") or 0) - float(mtime or 0)) > 0.5:
        return True
    return False


def list_fast_jobs(
    db: Any, source_id: int = 0, limit: int = 50000
) -> list[Job]:
    """Hızlı Index + preview repair lane.

    Gerçek light pending önce; light_done fakat WAITING_PREVIEW olanlar sonra.
    Repair sırasında light_status done kalır.
    """
    sql = f"""
        SELECT id, path, filename
        FROM files
        WHERE {_ACTIVE_FILE}
          AND (
                light_status IN ('pending', 'processing')
                OR (
                    light_status='done'
                    AND coalesce(nullif(heavy_status,''),'pending')='pending'
                    AND coalesce(needs_medium_preview,0)=1
                )
              )
        ORDER BY CASE light_status
                   WHEN 'pending' THEN 0
                   WHEN 'processing' THEN 1
                   ELSE 2
                 END,
                 id
        LIMIT ?
    """
    with db.connect() as conn:
        if source_id:
            sql = sql.replace(
                f"WHERE {_ACTIVE_FILE}",
                f"WHERE source_id=? AND {_ACTIVE_FILE}",
                1,
            )
            rows = conn.execute(sql, (int(source_id), int(limit))).fetchall()
        else:
            rows = conn.execute(sql, (int(limit),)).fetchall()
    return [(int(r["id"]), str(r["path"] or ""), str(r["filename"] or "")) for r in rows]


def count_fast_jobs(db: Any, source_id: int = 0) -> int:
    """Light pending + WAITING_PREVIEW repair toplamı."""
    sql = f"""
        SELECT COUNT(*) AS n FROM files
        WHERE {_ACTIVE_FILE}
          AND (
                light_status IN ('pending', 'processing')
                OR (
                    light_status='done'
                    AND coalesce(nullif(heavy_status,''),'pending')='pending'
                    AND coalesce(needs_medium_preview,0)=1
                )
              )
    """
    with db.connect() as conn:
        if source_id:
            sql = sql.replace(
                f"WHERE {_ACTIVE_FILE}",
                f"WHERE source_id=? AND {_ACTIVE_FILE}",
                1,
            )
            return int(conn.execute(sql, (int(source_id),)).fetchone()["n"] or 0)
        return int(conn.execute(sql).fetchone()["n"] or 0)


def count_light_pending_jobs(db: Any, source_id: int = 0) -> int:
    """Gerçek Light pending; kaynak önceliğinde preview repair'den önde."""
    sql = f"""
        SELECT COUNT(*) AS n FROM files
        WHERE {_ACTIVE_FILE}
          AND light_status IN ('pending', 'processing')
    """
    with db.connect() as conn:
        if source_id:
            sql = sql.replace(
                f"WHERE {_ACTIVE_FILE}",
                f"WHERE source_id=? AND {_ACTIVE_FILE}",
                1,
            )
            return int(conn.execute(sql, (int(source_id),)).fetchone()["n"] or 0)
        return int(conn.execute(sql).fetchone()["n"] or 0)


def list_general_ai_jobs(
    db: Any, source_id: int = 0, limit: int = 50000
) -> list[Job]:
    """Genel AI READY lane — preview bekleyenler light repair'e gider.

    Diskte cache kaybı sonradan anlaşılırsa worker WAITING_PREVIEW'e taşır.
    """
    sql = f"""
        SELECT id, path, filename
        FROM files
        WHERE {_ACTIVE_FILE}
          AND light_status='done'
          AND coalesce(nullif(heavy_status,''),'pending')='pending'
          AND coalesce(needs_medium_preview,0)=0
        ORDER BY id
        LIMIT ?
    """
    with db.connect() as conn:
        if source_id:
            sql = sql.replace(
                f"WHERE {_ACTIVE_FILE}",
                f"WHERE source_id=? AND {_ACTIVE_FILE}",
                1,
            )
            rows = conn.execute(sql, (int(source_id), int(limit))).fetchall()
        else:
            rows = conn.execute(sql, (int(limit),)).fetchall()
    return [(int(r["id"]), str(r["path"] or ""), str(r["filename"] or "")) for r in rows]


def count_general_ai_jobs(db: Any, source_id: int = 0) -> int:
    """Backlog sıralama için — list çekmeden COUNT."""
    sql = f"""
        SELECT COUNT(*) AS n FROM files
        WHERE {_ACTIVE_FILE}
          AND light_status='done'
          AND coalesce(nullif(heavy_status,''),'pending')='pending'
          AND coalesce(needs_medium_preview,0)=0
    """
    with db.connect() as conn:
        if source_id:
            sql = sql.replace(
                f"WHERE {_ACTIVE_FILE}",
                f"WHERE source_id=? AND {_ACTIVE_FILE}",
                1,
            )
            return int(conn.execute(sql, (int(source_id),)).fetchone()["n"] or 0)
        return int(conn.execute(sql).fetchone()["n"] or 0)


def reset_stale_processing(db: Any) -> dict[str, int]:
    """Start/resume: yarım kalan processing → pending.

    Çalışan worker yokken (program açılışı / index start) tüm processing
    kayıtları güvenle pending'e döner. Done'a dokunulmaz.
    """
    with db.connect() as conn:
        n_l = conn.execute(
            """
            UPDATE files SET light_status='pending', processing_lock='',
                             updated_at=datetime('now')
            WHERE light_status='processing'
            """
        ).rowcount
        n_h = conn.execute(
            """
            UPDATE files SET heavy_status='pending', processing_lock='',
                             updated_at=datetime('now')
            WHERE heavy_status='processing'
            """
        ).rowcount
        # Boş kilit + pending dışı takılılar
        n_lock = conn.execute(
            """
            UPDATE files SET processing_lock='', updated_at=datetime('now')
            WHERE coalesce(processing_lock,'')!=''
              AND light_status NOT IN ('processing')
              AND coalesce(heavy_status,'') NOT IN ('processing')
            """
        ).rowcount
    return {
        "light_reset": int(n_l or 0),
        "heavy_reset": int(n_h or 0),
        "lock_cleared": int(n_lock or 0),
    }


def claim_light(db: Any, file_id: int) -> bool:
    if not db.try_acquire_processing_lock(file_id, WORKER_LIGHT):
        return False
    db.mark_light_processing(file_id)
    return True


def claim_heavy(db: Any, file_id: int) -> bool:
    if not db.try_acquire_processing_lock(file_id, WORKER_HEAVY):
        return False
    db.mark_heavy_processing(file_id)
    return True


def complete_light(db: Any, file_id: int) -> None:
    """Hızlı Index başarı — kalıcı light_status='done'."""
    db.mark_light_done(file_id)
    row = db.get_file_by_id(file_id) or {}
    stage = str(row.get("index_stage") or "")
    if stage in ("", "pending_light", "failed", "processing"):
        db.set_file_index_stage(file_id, LIGHT_DONE)


def complete_heavy(db: Any, file_id: int) -> bool:
    """Yalnız AI Final/Patch kesişimi tam ise heavy_status='done'."""
    return bool(db.mark_heavy_done(file_id))


def release_claim(db: Any, file_id: int, worker: str) -> None:
    db.release_processing_lock(file_id, worker)


def fail_orphan_light_processing(db: Any) -> int:
    """processing kalmış + error_msg dolu → failed (worker öldükten sonra)."""
    from core.quarantine import classify_quarantine_reason

    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, coalesce(error_msg,'') AS error_msg
            FROM files
            WHERE light_status='processing'
              AND coalesce(error_msg,'') != ''
            """
        ).fetchall()
    n = 0
    for row in rows:
        err = str(row["error_msg"] or "")
        reason = classify_quarantine_reason(err)
        db.mark_light_failed(int(row["id"]), reason=reason or "other", error_msg=err)
        n += 1
    return n


def ensure_light_terminal(db: Any, file_id: int, *, error_msg: str = "") -> None:
    """Claim sonrası dosya processing'de kalmasın — done değilse failed."""
    row = db.get_file_by_id(file_id) or {}
    status = str(row.get("light_status") or "")
    if status in ("done", "failed"):
        return
    msg = error_msg or str(row.get("error_msg") or "light worker incomplete")
    from core.quarantine import classify_quarantine_reason

    db.mark_light_failed(
        file_id,
        reason=classify_quarantine_reason(msg) or "other",
        error_msg=msg,
    )


def list_orphan_fast_jobs(db: Any, limit: int = 50000) -> list[Job]:
    """source_id=0 veya aktif kaynak dışı — kaynak döngüsünde kaçan light işler."""
    sql = f"""
        SELECT id, path, filename
        FROM files
        WHERE {_ACTIVE_FILE}
          AND light_status IN ('pending', 'processing')
          AND coalesce(source_id, 0) = 0
        ORDER BY id
        LIMIT ?
    """
    with db.connect() as conn:
        rows = conn.execute(sql, (int(limit),)).fetchall()
    return [(int(r["id"]), str(r["path"] or ""), str(r["filename"] or "")) for r in rows]


def fail_missing_light_pending(db: Any) -> int:
    """Pending ama diskte yok → failed (sonsuza pending kalmasın)."""
    import os

    from core.utils import fs_access_path

    n = 0
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, path FROM files
            WHERE light_status='pending'
            ORDER BY id
            """
        ).fetchall()
    for row in rows:
        path = str(row["path"] or "")
        try:
            access = fs_access_path(path) if path else ""
            ok = bool(access) and os.path.exists(access)
            if not ok and path:
                ok = os.path.exists(path)
        except OSError:
            ok = False
        if ok:
            continue
        db.mark_light_failed(
            int(row["id"]),
            reason="path_invalid",
            error_msg="missing_on_disk",
        )
        n += 1
    return n


def write_pending_lt100_report(db: Any, out_dir: str | None = None) -> str:
    """Pending < 100 iken her dosya için bekleme nedeni raporu."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    lanes = count_lanes(db)
    pending_n = int(lanes.get("light_pending", 0) or 0) + int(
        lanes.get("light_processing", 0) or 0
    )
    if pending_n >= 100:
        return ""
    root = Path(out_dir) if out_dir else Path("artifacts") / "rc1_final"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    import os

    from core.utils import fs_access_path

    with db.connect() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                """
                SELECT f.id, f.path, f.status, f.light_status, f.heavy_status,
                       f.quarantine_reason, f.error_msg, f.source_id,
                       f.needs_review, f.index_stage, f.updated_at, f.file_size,
                       coalesce(q.retry_count, q.attempts, 0) AS retry_count,
                       coalesce(q.status, '') AS queue_state
                FROM files f
                LEFT JOIN index_queue q ON q.file_id = f.id
                WHERE f.light_status IN ('pending','processing')
                ORDER BY f.id
                """
            )
        ]

    def _wait_reason(r: dict) -> str:
        if r.get("light_status") == "processing":
            return "processing"
        sid = int(r.get("source_id") or 0)
        if sid == 0:
            return "orphan_source_id_0"
        path = str(r.get("path") or "")
        try:
            access = fs_access_path(path) if path else ""
            exists = bool(access) and os.path.exists(access)
            if not exists and path:
                exists = os.path.exists(path)
        except OSError:
            exists = False
        if not exists:
            return "missing_on_disk"
        if str(r.get("status") or "") == "error":
            return "status_error_awaiting_heal"
        return "awaiting_worker"

    payload = {
        "timestamp": stamp,
        "pending_n": pending_n,
        "lanes": lanes,
        "files": [
            {
                "id": r["id"],
                "path": r["path"],
                "wait_reason": _wait_reason(r),
                "current_state": r["light_status"],
                "retry_count": int(r.get("retry_count") or 0),
                "queue_state": r.get("queue_state") or "",
                "status": r["status"],
                "light_status": r["light_status"],
                "source_id": r["source_id"],
                "last_error": (r["error_msg"] or "")[:200],
                "quarantine_reason": r["quarantine_reason"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ],
    }
    path = root / f"pending_lt100_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def heal_demoted_light_done(db: Any) -> int:
    """Yanlış demote: light_done_at dolu ama light_status=pending → done geri al.

    light_done_at, mark_light_done ile yazılır; content_changed reset onu siler.
    Demote yolu light_done_at'ı silmediği için güvenli geri alma.
    """
    with db.connect() as conn:
        cur = conn.execute(
            """
            UPDATE files SET
                light_status='done',
                processing_lock='',
                updated_at=datetime('now')
            WHERE light_status='pending'
              AND coalesce(light_done_at,'') != ''
            """
        )
        return int(cur.rowcount or 0)


def requeue_retryable_heavy_preview_failures(db: Any) -> int:
    """Preview önkoşulu gerçek fail değildir: heavy pending'e geri al."""
    with db.connect() as conn:
        cur = conn.execute(
            """
            UPDATE files SET
                heavy_status='pending',
                index_stage='needs_medium_preview',
                needs_medium_preview=1,
                processing_lock='',
                quarantine_reason='',
                error_msg='',
                updated_at=datetime('now')
            WHERE heavy_status='failed'
              AND (
                    quarantine_reason='missing_preview'
                    OR error_msg IN (
                        'missing_preview_artifact',
                        'needs_medium_preview'
                    )
                  )
            """
        )
        return int(cur.rowcount or 0)


def heal_pending_error_conflict(db: Any) -> dict[str, int]:
    """light_status=pending + status=error çakışmasını kaldır (spin kaynağı).

    - quarantine_reason dolu / sert karantina → light failed
    - retry adayı (q boş, stage pending_light) → status temizlenir, light pending kalır
    """
    from core.index_analysis_guards import is_quarantined_file
    from core.quarantine import classify_quarantine_reason

    failed_n = 0
    cleared_n = 0
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, path, status, light_status, quarantine_reason,
                   error_msg, needs_review, index_stage, thumbnail_path,
                   feature_preview_path
            FROM files
            WHERE light_status='pending' AND status='error'
            """
        ).fetchall()
    for row in rows:
        rec = dict(row)
        fid = int(rec["id"])
        q = str(rec.get("quarantine_reason") or "").strip()
        # Retry: q boş ve stage hâlâ pending_light → status çakışmasını temizle
        if not q and str(rec.get("index_stage") or "") in ("", "pending_light"):
            with db.connect() as conn:
                conn.execute(
                    """
                    UPDATE files SET
                        status='pending',
                        needs_review=0,
                        error_msg='',
                        updated_at=datetime('now')
                    WHERE id=? AND light_status='pending' AND status='error'
                    """,
                    (fid,),
                )
            cleared_n += 1
            continue
        # Gerçek karantina / error kalıntısı → failed (pending'de durmasın)
        if is_quarantined_file(rec) or q:
            err = str(rec.get("error_msg") or "") or f"pending_error_conflict:{q or 'error'}"
            reason = q or classify_quarantine_reason(err) or "quarantine"
            db.mark_light_failed(fid, reason=reason, error_msg=err)
            failed_n += 1
        else:
            with db.connect() as conn:
                conn.execute(
                    """
                    UPDATE files SET
                        status='pending',
                        needs_review=0,
                        updated_at=datetime('now')
                    WHERE id=? AND light_status='pending' AND status='error'
                    """,
                    (fid,),
                )
            cleared_n += 1
    return {"conflict_total": len(rows), "cleared_retry": cleared_n, "marked_failed": failed_n}
