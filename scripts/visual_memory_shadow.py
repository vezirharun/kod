"""Shadow Visual Memory discovery — production DB read-only.

Writes only data/visual_memory/_shadow.db (or --customer).
Does not rebuild FAISS, does not change ranking.

python scripts/visual_memory_shadow.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.visual_memory.engine import VisualMemoryEngine
from core.settings import AppSettings


def main() -> None:
    settings = AppSettings.load()
    memory_root = Path(settings.db_path).parent / "visual_memory"
    engine = VisualMemoryEngine(memory_root)
    report = engine.shadow_from_production_db(
        settings.db_path,
        customer="_shadow",
        per_source=20,
        source_limit=9,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["production_db"] = "read_only"
    report["ranking_changed"] = False
    report["faiss_rebuild"] = False
    report["auto_class_claim_forbidden"] = report["claimed_class_n"] == 0
    dest = Path(settings.db_path).parent / "reports" / "visual_memory_shadow.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "items", "clusters", "discovered", "unknown",
        "claimed_class_n", "illegal_auto_status_n", "auto_class_claim_forbidden",
    )}, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
