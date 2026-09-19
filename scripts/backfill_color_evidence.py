"""Safe Stage-6/7 color evidence backfill (missing first, optional thumbnails).

Does not touch concept_examples / learning_events / index_v3 producers.
No ThreadPoolExecutor — sequential batches only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.color_evidence import backfill_color_evidence_db  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(ROOT / "data" / "patterns.db"))
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--no-thumbs", action="store_true")
    p.add_argument("--no-upgrade-ratios", action="store_true")
    p.add_argument("--all-missing", action="store_true")
    args = p.parse_args()
    limit = 50000 if args.all_missing else max(1, args.limit)
    stats = backfill_color_evidence_db(
        args.db,
        limit=limit,
        only_missing=True,
        use_thumbnails=not args.no_thumbs,
        batch_size=max(1, args.batch_size),
        upgrade_ratios=not args.no_upgrade_ratios,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
