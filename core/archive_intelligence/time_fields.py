"""Labeled time sources for archive queries.

File system dates are NOT usage dates. Every field carries an explicit source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class LabeledTimeField:
    key: str
    value: str
    source: str
    is_usage_date: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "is_usage_date": self.is_usage_date,
            "note": self.note,
        }


def _fmt_epoch(raw: Any) -> str:
    try:
        ts = float(raw or 0)
    except (TypeError, ValueError):
        return ""
    if ts <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _fmt_text(raw: Any) -> str:
    text = str(raw or "").strip()
    return text


def labeled_time_fields(file_row: dict[str, Any]) -> list[LabeledTimeField]:
    """Return reliable date fields with explicit source labeling."""
    out: list[LabeledTimeField] = []
    mtime = _fmt_epoch(file_row.get("mtime"))
    if mtime:
        out.append(
            LabeledTimeField(
                key="mtime",
                value=mtime,
                source="filesystem_mtime",
                is_usage_date=False,
                note="Dosya sistemi değişim zamanı — kullanım tarihi değil",
            )
        )
    indexed = _fmt_text(file_row.get("indexed_at"))
    if indexed:
        out.append(
            LabeledTimeField(
                key="indexed_at",
                value=indexed,
                source="index_clock",
                is_usage_date=False,
                note="Vezir indeksleme zamanı — dosya kullanım tarihi değil",
            )
        )
    created = _fmt_text(file_row.get("created_at"))
    if created:
        out.append(
            LabeledTimeField(
                key="created_at",
                value=created,
                source="db_created_at",
                is_usage_date=False,
                note="Kayıt oluşturma zamanı (DB) — kullanım tarihi değil",
            )
        )
    preview_mt = _fmt_epoch(file_row.get("feature_preview_mtime"))
    if preview_mt:
        out.append(
            LabeledTimeField(
                key="feature_preview_mtime",
                value=preview_mt,
                source="preview_cache_mtime",
                is_usage_date=False,
                note="Önizleme önbelleği mtime — orijinal kullanım tarihi değil",
            )
        )
    # Explicit absence: no usage/event date in current index schema.
    if not any(f.is_usage_date for f in out):
        out.append(
            LabeledTimeField(
                key="usage_date",
                value="",
                source="unavailable",
                is_usage_date=True,
                note="Kullanım/olay tarihi indekslenmiyor — sorgu için mtime/indexed_at kullanın ve kaynağı belirtin",
            )
        )
    return out
