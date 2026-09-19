"""Index Doctor — index bütünlüğü tarama / onarım / rapor.

Calistir:
    python scripts/index_doctor.py
    python scripts/index_doctor.py --repair --repair-limit 200
    python scripts/index_doctor.py --file-id 130165 --force-reindex
    python scripts/index_doctor.py --file-id 130165 --check-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import Database
from core.index_integrity import (
    STATUS_INCOMPLETE,
    check_file,
    force_reindex_file,
    run_doctor,
)
from core.settings import AppSettings


def print_doctor(report) -> None:
    w = 64
    print("=" * w)
    print("  INDEX DOCTOR")
    print(f"  {report.timestamp}")
    print("=" * w)
    print(f"  Toplam indexed........ {report.total_indexed:,}")
    print(f"  Incomplete............ {report.total_incomplete:,}")
    print(f"  Feature eksik......... {report.feature_missing:,}")
    print(f"  Thumbnail eksik....... {report.thumbnail_missing:,}")
    print(f"  Hash eksik............ {report.hash_missing:,}")
    print(f"  Embedding eksik....... {report.embedding_missing:,}")
    print(f"  DNA eksik............. {report.dna_missing:,}")
    print(f"  False Done............ {report.false_done:,}")
    print(f"  Marked incomplete..... {report.marked_incomplete:,}")
    print(f"  Requeued / auto-fixed. {report.auto_fixed:,}")
    print("-" * w)
    for key, rows in (report.samples or {}).items():
        if not rows:
            continue
        print(f"  [{key}]")
        for row in rows[:5]:
            print(f"    - id={row.get('id')}  {row.get('filename', '')[:50]}")
    print("=" * w)


def print_file_report(rep) -> None:
    print("=" * 64)
    print(f"  FILE INTEGRITY  id={rep.file_id}")
    print("=" * 64)
    print(f"  filename...... {rep.filename}")
    print(f"  status........ {rep.status}")
    print(f"  search_ready.. {rep.search_ready}")
    print(f"  missing....... {', '.join(rep.missing) or '-'}")
    print(f"  false_done.... {', '.join(rep.false_done_stages) or '-'}")
    flags = asdict(rep.flags)
    for k, v in flags.items():
        print(f"    {k:14} {'OK' if v else 'MISSING'}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Doctor / Integrity Engine")
    parser.add_argument("--limit", type=int, default=0, help="Tarama limiti (0=hepsi)")
    parser.add_argument("--repair", action="store_true", help="Eksikleri incomplete+requeue")
    parser.add_argument("--repair-limit", type=int, default=500)
    parser.add_argument("--file-id", type=int, default=0)
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument(
        "--run-indexer",
        action="store_true",
        help="Force reindex sonrası Indexer'ı hemen çalıştır (büyük TIFF defer kapalı)",
    )
    parser.add_argument(
        "--backfill-patches",
        action="store_true",
        help="Patch meta'sı boş dosyaları thumbnail üzerinden doldur",
    )
    parser.add_argument("--backfill-limit", type=int, default=200)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    settings = AppSettings.load()
    db = Database(settings.db_path)

    if args.backfill_patches and not args.file_id:
        from core.index_integrity import force_reindex_missing_patches

        result = force_reindex_missing_patches(
            db, settings=settings, limit=int(args.backfill_limit)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.save:
            out = ROOT / "data" / "reports" / "patch_backfill_report.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("Saved:", out)
        return

    if args.file_id:
        before = check_file(db, args.file_id, settings=settings)
        print_file_report(before)
        if args.check_only and not args.force_reindex and not args.backfill_patches:
            return
        if args.backfill_patches and not args.force_reindex:
            from core.index_integrity import force_reindex_missing_patches

            patch_result = force_reindex_missing_patches(
                db, settings=settings, file_ids=[args.file_id], limit=1
            )
            print("\nPatch backfill:")
            print(json.dumps(patch_result, ensure_ascii=False, indent=2))
            after = check_file(db, args.file_id, settings=settings)
            print_file_report(after)
            if args.save:
                out = ROOT / "data" / "reports" / f"patch_backfill_{args.file_id}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(
                        {
                            "before": asdict(before),
                            "patch_backfill": patch_result,
                            "after": asdict(after),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print("Saved:", out)
            return
        if args.force_reindex:
            result = force_reindex_file(
                db,
                args.file_id,
                settings=settings,
                run_indexer=bool(args.run_indexer),
            )
            print("\nForce reindex:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.backfill_patches or args.run_indexer:
                from core.index_integrity import force_reindex_missing_patches

                patch_result = force_reindex_missing_patches(
                    db, settings=settings, file_ids=[args.file_id], limit=1
                )
                print("\nPatch backfill:")
                print(json.dumps(patch_result, ensure_ascii=False, indent=2))
            after = check_file(db, args.file_id, settings=settings)
            print("\nAfter queue update (artifacts may still be pending until indexer runs):")
            print_file_report(after)
            if args.save:
                out = ROOT / "data" / "reports" / f"integrity_file_{args.file_id}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(
                        {
                            "before": asdict(before),
                            "force_reindex": result,
                            "after_queue": asdict(after),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print("Saved:", out)
        return

    report = run_doctor(
        settings,
        limit=args.limit,
        repair=args.repair,
        repair_limit=args.repair_limit,
    )
    print_doctor(report)
    if args.save:
        out = ROOT / "data" / "reports" / "index_doctor_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("Saved:", out)

    # Hizli ozet: incomplete ornegi
    with db.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) c FROM files WHERE status=?",
            (STATUS_INCOMPLETE,),
        ).fetchone()["c"]
    print(f"DB incomplete status count: {n:,}")


if __name__ == "__main__":
    main()
