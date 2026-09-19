"""Phase 2 Visual Memory shadow analysis. Production read-only.

python scripts/visual_memory_phase2_analyze.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.settings import AppSettings
from core.visual_memory.phase2_analyze import run_shadow_analysis


def main() -> None:
    settings = AppSettings.load()
    report = run_shadow_analysis(settings.db_path, per_source=20, source_limit=5)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    dest = Path(settings.db_path).parent / "reports" / "visual_memory_phase2_shadow.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("cluster_count", report["cluster_count"])
    print("largest", report["largest_cluster"])
    print("mean_cohesion", report["mean_cohesion"])
    print("inter_cluster_separation", report["inter_cluster_separation"])
    print("mixed_family", len(report["mixed_family_clusters"]))
    print("mixed_object", len(report["mixed_object_clusters"]))
    print("candidate_splits", len(report["candidate_splits"]))
    for c in report["candidate_splits"]:
        sc = c.get("scores") or {}
        print(
            "  ",
            c["cluster_id"],
            c["reason"],
            "n=",
            c["member_count"],
            "best=",
            sc.get("best_channel"),
            "gain=",
            sc.get("best_gain"),
            "split?",
            sc.get("recommend_split"),
        )
    print("prototype", report["synthetic_style_prototype_on_shadow"])
    print("wrote", dest)


if __name__ == "__main__":
    main()
