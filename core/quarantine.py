"""Karantina nedenleri — preview-first index pipeline + RC1 failed rapor."""

from __future__ import annotations

from typing import Any

# DB'ye yazılan sabit reason kodları
PREVIEW_FAILED = "preview_failed"
FORMAT_UNSUPPORTED = "format_unsupported"
FILE_CORRUPT = "file_corrupt"
PERMISSION_DENIED = "permission_denied"
NETWORK_TIMEOUT = "network_timeout"
DEPENDENCY_MISSING = "dependency_missing"
FILE_TOO_LARGE = "file_too_large"
AI_ANALYSIS_ERROR = "ai_analysis_error"
DECODE_ERROR = "decode_error"
TRUNCATED = "truncated"
DB_TRANSACTION = "db_transaction"
PATH_INVALID = "path_invalid"
OTHER = "other"

# Arama dışı kalır — tekrar denenmez (preview yok)
HARD_QUARANTINE_REASONS = frozenset(
    {
        PREVIEW_FAILED,
        FORMAT_UNSUPPORTED,
        FILE_CORRUPT,
        PERMISSION_DENIED,
        NETWORK_TIMEOUT,
        DEPENDENCY_MISSING,
        FILE_TOO_LARGE,
        DECODE_ERROR,
        TRUNCATED,
    }
)

# Preview/embedding var — aranabilir kalır, yalnızca derin analiz eksik
SOFT_QUARANTINE_REASONS = frozenset({AI_ANALYSIS_ERROR})

REASON_LABELS_TR: dict[str, str] = {
    PREVIEW_FAILED: "Önizleme üretilemedi",
    FORMAT_UNSUPPORTED: "Format desteklenmiyor",
    FILE_CORRUPT: "Broken / bozuk dosya",
    PERMISSION_DENIED: "Permission / izin yok",
    NETWORK_TIMEOUT: "Network / zaman aşımı",
    DEPENDENCY_MISSING: "Bağımlılık eksik",
    FILE_TOO_LARGE: "Too Large / çok büyük",
    AI_ANALYSIS_ERROR: "AI analiz hatası",
    DECODE_ERROR: "Decode Error",
    TRUNCATED: "Truncated",
    DB_TRANSACTION: "DB Transaction",
    PATH_INVALID: "Invalid Path (Errno 22)",
    OTHER: "Other",
}

# Ürün kovaları (RC1 rapor)
PRODUCT_BUCKETS = (
    ("Broken TIFF", FILE_CORRUPT),
    ("Truncated", TRUNCATED),
    ("Permission", PERMISSION_DENIED),
    ("Unsupported", FORMAT_UNSUPPORTED),
    ("Too Large", FILE_TOO_LARGE),
    ("Decode Error", DECODE_ERROR),
    ("Network", NETWORK_TIMEOUT),
    ("Invalid Path", PATH_INVALID),
    ("Other", OTHER),
)


def classify_quarantine_reason(
    message: str,
    *,
    preview_status: str = "",
    unsupported: bool = False,
) -> str:
    """Hata metninden karantina reason kodu çıkar."""
    msg = (message or "").lower()
    ps = (preview_status or "").lower()
    if unsupported or ps == "unsupported_preview":
        return FORMAT_UNSUPPORTED
    if ps == "preview_missing_dependency":
        return DEPENDENCY_MISSING
    if "cannot start a transaction" in msg or "within a transaction" in msg:
        return DB_TRANSACTION
    # Guard mesajı — gerçek timeout değil
    if "attempted to open network original" in msg:
        return PREVIEW_FAILED
    if "errno 22" in msg or "invalid argument" in msg:
        return PATH_INVALID
    if "timeout" in msg or "timed out" in msg or "zaman aşım" in msg:
        return NETWORK_TIMEOUT
    if (
        "permission" in msg
        or "erişim" in msg
        or "access denied" in msg
        or "winerror 5" in msg
    ):
        return PERMISSION_DENIED
    if (
        "decompression bomb" in msg
        or "exceeds limit" in msg
        or "too large" in msg
        or "çok büyük" in msg
    ):
        return FILE_TOO_LARGE
    if "truncated" in msg:
        return TRUNCATED
    if (
        "broken data stream" in msg
        or "corrupt" in msg
        or "bozuk" in msg
        or "cannot identify image" in msg
    ):
        return FILE_CORRUPT
    if (
        "decode" in msg
        or "decoder" in msg
        or "oserror" in msg
        and ("tif" in msg or "tiff" in msg)
    ):
        return DECODE_ERROR
    if "preview" in msg or "thumbnail" in msg or "önizleme" in msg:
        return PREVIEW_FAILED
    if "ai" in msg or "embedding" in msg or "dino" in msg or "clip" in msg:
        return AI_ANALYSIS_ERROR
    if "network" in msg or "smb" in msg or "winerror 53" in msg or "winerror 64" in msg:
        return NETWORK_TIMEOUT
    if not msg.strip():
        return OTHER
    return OTHER


def reason_label(reason: str) -> str:
    return REASON_LABELS_TR.get(reason, reason or "Bilinmeyen")


def is_hard_quarantine(reason: str | None) -> bool:
    return bool(reason) and reason in HARD_QUARANTINE_REASONS


def is_soft_quarantine(reason: str | None) -> bool:
    return bool(reason) and reason in SOFT_QUARANTINE_REASONS


def backfill_failed_quarantine_reasons(
    db: Any, *, limit: int = 20000, force: bool = False
) -> int:
    """light_status=failed → error_msg'den quarantine_reason doldur/yenile."""
    updated = 0
    with db.connect() as conn:
        if force:
            rows = conn.execute(
                """
                SELECT id, coalesce(error_msg,'') AS error_msg
                FROM files
                WHERE light_status='failed'
                ORDER BY id
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, coalesce(error_msg,'') AS error_msg
                FROM files
                WHERE light_status='failed'
                  AND coalesce(quarantine_reason,'')=''
                ORDER BY id
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        for row in rows:
            reason = classify_quarantine_reason(str(row["error_msg"] or ""))
            conn.execute(
                "UPDATE files SET quarantine_reason=? WHERE id=?",
                (reason, int(row["id"])),
            )
            updated += 1
    return updated


def failed_reason_report(db: Any, *, limit: int = 10) -> dict[str, Any]:
    """Top N neden + ürün kovaları (light_failed)."""
    with db.connect() as conn:
        total = int(
            conn.execute(
                """
                SELECT COUNT(*) AS c FROM files
                WHERE status NOT IN ('excluded_internal','missing')
                  AND light_status='failed'
                """
            ).fetchone()["c"]
        )
        rows = conn.execute(
            """
            SELECT coalesce(nullif(quarantine_reason,''), 'other') AS reason,
                   COUNT(*) AS n
            FROM files
            WHERE status NOT IN ('excluded_internal','missing')
              AND light_status='failed'
            GROUP BY 1
            ORDER BY n DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    top = [
        {
            "reason": str(r["reason"]),
            "label": reason_label(str(r["reason"])),
            "count": int(r["n"]),
        }
        for r in rows
    ]
    # Kova özeti
    by_reason = {t["reason"]: t["count"] for t in top}
    # full counts for buckets
    with db.connect() as conn:
        all_rows = conn.execute(
            """
            SELECT coalesce(nullif(quarantine_reason,''), 'other') AS reason,
                   COUNT(*) AS n
            FROM files
            WHERE status NOT IN ('excluded_internal','missing')
              AND light_status='failed'
            GROUP BY 1
            """
        ).fetchall()
    full = {str(r["reason"]): int(r["n"]) for r in all_rows}
    buckets = []
    for label, code in PRODUCT_BUCKETS:
        buckets.append({"bucket": label, "reason": code, "count": int(full.get(code, 0))})
    # other codes not in product list
    known = {code for _, code in PRODUCT_BUCKETS}
    extra = sum(n for r, n in full.items() if r not in known)
    if extra:
        for b in buckets:
            if b["reason"] == OTHER:
                b["count"] += extra
    return {"total_failed": total, "top": top, "buckets": buckets}
