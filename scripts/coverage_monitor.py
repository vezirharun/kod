"""Dataset Coverage Monitor — her dosya için eksik özellik kontrolü.

Çalıştır:
    python scripts/coverage_monitor.py              # özet
    python scripts/coverage_monitor.py --queue      # eksikleri kuyruğa alır
    python scripts/coverage_monitor.py --save       # JSON raporu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import Database
from core.settings import AppSettings


CHECKS = [
    "thumbnail",
    "preview",
    "embedding",
    "pattern_dna",
    "semantic",
    "ocr",
    "knowledge",
    "fts",
]

_STATUS_ICON = {"ok": "OK", "missing": "!!", "partial": "~~"}


@dataclass
class FileCoverage:
    file_id: int
    filename: str
    checks: dict[str, str]   # check_name → "ok" | "missing"

    @property
    def ready(self) -> bool:
        return all(v == "ok" for v in self.checks.values())

    @property
    def missing_list(self) -> list[str]:
        return [k for k, v in self.checks.items() if v != "ok"]


@dataclass
class CoverageReport:
    timestamp: str
    total_indexed: int
    fully_ready: int
    partial: int
    check_totals: dict[str, int]   # check_name → count_ok
    check_missing: dict[str, int]  # check_name → count_missing
    sample_incomplete: list[dict] = field(default_factory=list)

    def print_summary(self) -> None:
        w = 60
        print("=" * w)
        print(f"  DATASET COVERAGE MONITOR  —  {self.timestamp}")
        print(f"  Toplam indexed: {self.total_indexed:,}")
        pct_ready = self.fully_ready / max(self.total_indexed, 1) * 100
        print(f"  Tam hazır (READY): {self.fully_ready:,} ({pct_ready:.1f}%)")
        print(f"  Eksik var:         {self.partial:,}")
        print("-" * w)
        for chk in CHECKS:
            ok  = self.check_totals.get(chk, 0)
            mis = self.check_missing.get(chk, 0)
            pct = ok / max(self.total_indexed, 1) * 100
            icon = "OK" if mis == 0 else ("~" if mis < self.total_indexed * 0.1 else "!!")
            print(f"  {icon} {chk:15} {ok:>7,} / {self.total_indexed:,}  ({pct:.1f}%)")
        if self.sample_incomplete:
            print("-" * w)
            print("  Eksik örnek (ilk 5):")
            for s in self.sample_incomplete[:5]:
                missing = ", ".join(s.get("missing", []))
                print(f"    id={s['id']}  {s['filename'][:40]}  eksik=[{missing}]")
        print("=" * w)


def _check_file(row: dict) -> dict[str, str]:
    checks: dict[str, str] = {}
    checks["thumbnail"] = "ok" if row.get("thumbnail_path") else "missing"
    checks["preview"]   = "ok" if (row.get("thumbnail_path") or row.get("path")) else "missing"

    tm = row.get("texture_map") or {}
    if isinstance(tm, str):
        try:
            tm = json.loads(tm)
        except Exception:
            tm = {}

    # Embedding: DINO veya CLIP
    has_emb = bool(
        row.get("dino_embedding") or
        row.get("clip_embedding") or
        tm.get("embedding_version")
    )
    checks["embedding"] = "ok" if has_emb else "missing"

    # Pattern DNA
    dna = tm.get("pattern_dna") or {}
    checks["pattern_dna"] = "ok" if (isinstance(dna, dict) and dna) else "missing"

    # Semantic tags
    sem = tm.get("semantic_tags") or {}
    checks["semantic"] = "ok" if (isinstance(sem, dict) and sem) else "missing"

    # OCR
    ocr = row.get("ocr_text") or ""
    checks["ocr"] = "ok" if ocr else "missing"

    # Knowledge — proxy: pattern_family set and not "unknown"
    pf = row.get("pattern_family") or tm.get("pattern_family") or ""
    checks["knowledge"] = "ok" if (pf and pf not in ("", "unknown")) else "missing"

    # FTS — proxy: text_search_blob exists
    blob = row.get("text_search_blob") or ""
    checks["fts"] = "ok" if blob else "missing"

    return checks


def build_report(db: Database, *, limit: int = 0) -> CoverageReport:
    sql = """
        SELECT f.id, f.filename, f.path, f.thumbnail_path, f.ocr_text,
               f.pattern_family, f.text_search_blob,
               fe.texture_map, fe.dino_embedding, fe.clip_embedding
        FROM files f
        LEFT JOIN features fe ON fe.file_id = f.id
        WHERE f.status = 'indexed'
    """
    if limit > 0:
        sql += f" LIMIT {int(limit)}"

    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]

    total = len(rows)
    check_ok:  dict[str, int] = {c: 0 for c in CHECKS}
    check_mis: dict[str, int] = {c: 0 for c in CHECKS}
    fully_ready = 0
    sample_incomplete: list[dict] = []

    for row in rows:
        chks = _check_file(row)
        all_ok = True
        for c in CHECKS:
            if chks.get(c) == "ok":
                check_ok[c] += 1
            else:
                check_mis[c] += 1
                all_ok = False
        if all_ok:
            fully_ready += 1
        elif len(sample_incomplete) < 10:
            sample_incomplete.append({
                "id": row["id"],
                "filename": row["filename"],
                "missing": [c for c, v in chks.items() if v != "ok"],
            })

    return CoverageReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_indexed=total,
        fully_ready=fully_ready,
        partial=total - fully_ready,
        check_totals=check_ok,
        check_missing=check_mis,
        sample_incomplete=sample_incomplete,
    )


def queue_missing(db: Database, report: CoverageReport) -> dict[str, int]:
    """Eksik özelliklere sahip dosyaları index kuyruğuna ekle."""
    queued: dict[str, int] = {}
    if report.check_missing.get("thumbnail", 0) > 0:
        n = db.requeue_files_missing_thumbnails()
        queued["thumbnail"] = n.get("queued", 0) if isinstance(n, dict) else int(n or 0)
    if report.check_missing.get("embedding", 0) > 0:
        n = db.queue_files_missing_ai_embeddings()
        queued["embedding"] = n.get("queued", 0) if isinstance(n, dict) else int(n or 0)
    if report.check_missing.get("ocr", 0) > 0:
        n = db.queue_files_missing_ocr()
        queued["ocr"] = n.get("queued", 0) if isinstance(n, dict) else int(n or 0)
    return queued


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Coverage Monitor")
    parser.add_argument("--queue", action="store_true", help="Eksikleri kuyruğa al")
    parser.add_argument("--save",  action="store_true", help="JSON raporu kaydet")
    parser.add_argument("--limit", type=int, default=0, help="Satır limiti (0=hepsi)")
    args = parser.parse_args()

    settings = AppSettings.load()
    db       = Database(settings.db_path)

    print("Coverage analizi yapılıyor…")
    report = build_report(db, limit=args.limit)
    report.print_summary()

    if args.queue:
        print("Eksikler kuyruğa alınıyor…")
        q = queue_missing(db, report)
        for k, n in q.items():
            print(f"  {k}: {n} dosya kuyruğa alındı")

    if args.save:
        out = Path("data/reports/coverage_report.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        import dataclasses
        out.write_text(
            json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Rapor: {out}")


if __name__ == "__main__":
    main()
