"""DINO-gated family split — shadow only, no persist.

python scripts/visual_memory_phase2_split.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.settings import AppSettings
from core.visual_memory.phase2_analyze import run_shadow_split


def main() -> None:
    settings = AppSettings.load()
    report = run_shadow_split(settings.db_path, per_source=20, source_limit=5)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    dest = Path(settings.db_path).parent / "reports" / "visual_memory_phase2_split.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    b, a = report["before"], report["after"]
    print("before", json.dumps(b, ensure_ascii=False))
    print("after ", json.dumps(a, ensure_ascii=False))
    print("splits", len(report["splits"]))
    for s in report["splits"]:
        print(
            f"  {s['parent_cluster_id']} -> {s['child_cluster_id']} "
            f"family={s['family']} n={s['member_count']} "
            f"clip={s['clip_cohesion']} dino_gain={s['split_gain']} "
            f"status={s['status']} class={s['claimed_class']!r}"
        )
    print("skipped", report.get("skipped"))
    print("success", report["success"])
    print("wrote", dest)


if __name__ == "__main__":
    main()
