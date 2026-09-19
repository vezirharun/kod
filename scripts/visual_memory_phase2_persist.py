"""Persist Phase 2 split snapshot to an isolated shadow DB.

Does not write patterns.db, Phase 1 VM, FAISS, or ranking.

python scripts/visual_memory_phase2_persist.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.settings import AppSettings
from core.visual_memory.phase2_analyze import persist_phase2_shadow
from core.visual_memory.phase2_shadow import phase2_shadow_path


def _fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {"path": str(path), "exists": True, "size": st.st_size, "mtime": st.st_mtime}


def main() -> None:
    settings = AppSettings.load()
    data_dir = Path(settings.db_path).parent
    before = {
        "patterns_db": _fingerprint(Path(settings.db_path)),
        "faiss_clip": _fingerprint(Path(settings.faiss_clip_path)),
        "faiss_dino": _fingerprint(Path(settings.faiss_dino_path)),
        "phase1_vm": _fingerprint(data_dir / "visual_memory" / "_shadow.db"),
    }
    shadow = phase2_shadow_path(data_dir)
    report = persist_phase2_shadow(
        settings.db_path, shadow, per_source=20, source_limit=5
    )
    after = {
        "patterns_db": _fingerprint(Path(settings.db_path)),
        "faiss_clip": _fingerprint(Path(settings.faiss_clip_path)),
        "faiss_dino": _fingerprint(Path(settings.faiss_dino_path)),
        "phase1_vm": _fingerprint(data_dir / "visual_memory" / "_shadow.db"),
    }
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["fingerprints_before"] = before
    report["fingerprints_after"] = after
    report["production_unchanged"] = before == after
    dest = data_dir / "reports" / "visual_memory_phase2_persist.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("phase1", json.dumps(report["phase1"], ensure_ascii=False))
    print("phase2", json.dumps(report["phase2"], ensure_ascii=False))
    print("idempotent", report["idempotent"], "clusters", report["shadow"]["clusters"])
    print("claimed_class_n", report["claimed_class_n"])
    print("production_unchanged", report["production_unchanged"])
    print("shadow_db", report["shadow_db"])
    print("wrote", dest)


if __name__ == "__main__":
    main()
