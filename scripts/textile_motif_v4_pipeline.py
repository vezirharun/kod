"""Textile motif V4 pipeline: dry-run train + acceptance. INDEX FROZEN.

Does not write Pattern Index / FAISS / patterns.db. Does not scan 116K.
Does not activate production. Dummy metrics are not textile P/R.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.index_freeze import freeze_fingerprint, snapshot_index_artifacts
from core.settings import DEFAULT_DATA_DIR
from core.textile_motif_v4 import FIXTURE_DIR, MOTIF_DATA_DIR, run_acceptance


def _snap() -> dict:
    return snapshot_index_artifacts(
        db_path=DEFAULT_DATA_DIR / "patterns.db",
        faiss_dino_path=DEFAULT_DATA_DIR / "faiss_dino.index",
        faiss_clip_path=DEFAULT_DATA_DIR / "faiss_clip.index",
        cache_dir="",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Motif V4 isolated train/acceptance pipeline")
    p.add_argument("--dry-run", action="store_true", help="tiny dummy fixture compile/train")
    p.add_argument("--acceptance", action="store_true", help="run P/R/F1/mAP harness")
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "reports" / "textile_motif_v4.json",
    )
    args = p.parse_args()
    if not args.dry_run and not args.acceptance:
        args.dry_run = True
        args.acceptance = True

    ds = args.dataset
    if ds is None:
        human = MOTIF_DATA_DIR / "annotations.json"
        ds = human if human.is_file() else FIXTURE_DIR

    before = freeze_fingerprint(_snap())
    with tempfile.TemporaryDirectory(prefix="motif_v4_") as tmp:
        payload = run_acceptance(ds, work=Path(tmp), pred_mode="none")
    after = freeze_fingerprint(_snap())
    payload["index_freeze"]["fingerprint_changed"] = after != before
    payload["index_freeze"]["faiss_changed"] = any(
        after.get(k) != before.get(k)
        for k in set(before) | set(after)
        if "faiss" in k.replace("\\", "/").lower()
    )
    payload["index_freeze"]["fingerprint_stable"] = after == before

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    v2 = ROOT / "data" / "reports" / "textile_motif_v2.json"
    v2.parent.mkdir(parents=True, exist_ok=True)
    v2.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    gate = payload["production_gate"]["status"]
    print("MOTIF_V4", gate, "dummy=", payload["dataset"]["dummy"])
    print("INDEX fingerprint changed?", payload["index_freeze"]["fingerprint_changed"])
    print("FAISS changed?", payload["index_freeze"]["faiss_changed"])
    print("116K scanned?", payload["index_freeze"]["archive_116k_scanned"])
    print("Reindex?", payload["index_freeze"]["reindex"])
    print("report", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
