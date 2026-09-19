#!/usr/bin/env python
"""Vezir Pattern Foundation Dataset Builder CLI.

Örnek:
  python scripts/build_foundation_dataset.py
  python scripts/build_foundation_dataset.py --limit 5000 --max-triplets 100000
  python scripts/build_foundation_dataset.py --min-confidence 0.95 --materialize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Build Vezir Pattern Foundation Dataset v1.0")
    p.add_argument(
        "--out",
        default=str(ROOT / "data" / "foundation_dataset"),
        help="Çıktı klasörü (default: data/foundation_dataset)",
    )
    p.add_argument("--min-confidence", type=float, default=0.80)
    p.add_argument("--high-confidence", type=float, default=0.95)
    p.add_argument("--positives", type=int, default=4)
    p.add_argument("--hard-positives", type=int, default=2)
    p.add_argument("--hard-negatives", type=int, default=2)
    p.add_argument("--negatives", type=int, default=3)
    p.add_argument("--max-triplets", type=int, default=2_000_000)
    p.add_argument("--max-anchors", type=int, default=0, help="0 = tümü")
    p.add_argument("--limit", type=int, default=0, help="DB'den okunacak max satır (0=tümü)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--materialize",
        action="store_true",
        help="Thumbnail'leri anchor/positive/negative klasörlerine kopyala",
    )
    p.add_argument(
        "--allow-mid-confidence",
        action="store_true",
        help="0.80–0.95 arası aile bilinen kayıtları da al",
    )
    args = p.parse_args()

    from core.db import Database
    from core.foundation_dataset import FoundationDatasetBuilder
    from core.settings import AppSettings

    settings = AppSettings.load()
    db = Database(settings.db_path)
    out = Path(args.out)
    builder = FoundationDatasetBuilder(
        db,
        output_dir=out,
        min_confidence=args.min_confidence,
        high_confidence=args.high_confidence,
        positives_per_anchor=args.positives,
        hard_positives=args.hard_positives,
        hard_negatives=args.hard_negatives,
        negatives=args.negatives,
        max_triplets=args.max_triplets,
        max_anchors=args.max_anchors or None,
        seed=args.seed,
        materialize_thumbs=args.materialize,
        prefer_high_confidence=not args.allow_mid_confidence,
    )
    load_limit = args.limit or None
    print(f"DB: {settings.db_path}")
    print(f"OUT: {out}")
    print(f"Loading records (limit={load_limit or 'ALL'})…")
    n = builder.load_records(limit=load_limit)
    print(f"Kept records: {n} (skipped low-conf: {getattr(builder, '_skipped_low', 0)})")
    clusters = builder.build_clusters()
    print(f"Clusters: {len(clusters)}")
    print("Exporting triplets / pairs / metadata…")
    bench = builder.export()
    print()
    print("=== Foundation Dataset Benchmark ===")
    print(json.dumps(bench.to_dict(), indent=2, ensure_ascii=False))
    print()
    print(f"Wrote: {out / 'triplets.jsonl'}")
    print(f"Wrote: {out / 'pairs.jsonl'}")
    print(f"Wrote: {out / 'clusters.jsonl'}")
    print(f"Wrote: {out / 'metadata.json'}")
    print(f"Wrote: {out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
