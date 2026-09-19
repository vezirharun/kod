"""Safely recover faiss_clip.map.npy from stored DB embeddings.

Does not rewrite faiss_clip.index, does not recompute CLIP, does not write DB.
Zero / unmatched slots stay -1 and are skipped at search time.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.faiss_store import (
    FaissStore,
    audit_clip_id_map,
    inspect_faiss_id_map,
    load_faiss_id_map,
    recover_clip_id_map_from_stored_embeddings,
    restore_clip_id_map_atomic,
)
from core.settings import AppSettings


def main() -> None:
    settings = AppSettings.load()
    clip_path = Path(settings.faiss_clip_path)
    map_path = clip_path.with_suffix(".map.npy")
    index_before = {
        "exists": clip_path.exists(),
        "size": int(clip_path.stat().st_size) if clip_path.exists() else 0,
        "mtime": clip_path.stat().st_mtime if clip_path.exists() else 0,
    }
    inspection = inspect_faiss_id_map(map_path)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clip_index": str(clip_path),
        "map_path": str(map_path),
        "index_before": index_before,
        "map_before": inspection,
        "wrote_map": False,
        "skipped": False,
    }

    existing_ids: list[int] | None = None
    try:
        existing_ids = load_faiss_id_map(map_path)
    except Exception as exc:
        report["load_before_error"] = str(exc)

    if existing_ids and inspection.get("is_npy"):
        mapped = sum(1 for x in existing_ids if int(x) > 0)
        if mapped >= 1000:
            report["skipped"] = True
            report["skip_reason"] = f"map already valid npy with {mapped} positive ids"
            ids = existing_ids
            rec_stats = {"unique_ok": mapped, "reused_existing": True}
        else:
            ids = None
            rec_stats = {}
    else:
        ids = None
        rec_stats = {}

    if ids is None:
        print("recovering CLIP id map from stored embeddings…")
        ids, rec_stats = recover_clip_id_map_from_stored_embeddings(
            clip_path, settings.db_path
        )
        print("recovery", rec_stats)
        report["recovery"] = rec_stats
        pre = audit_clip_id_map(
            ids, clip_index_path=clip_path, db_path=settings.db_path, probe_n=5
        )
        report["audit_before_write"] = {
            k: v for k, v in pre.items() if k != "probes"
        }
        report["probes_before_write"] = pre.get("probes")
        if not pre.get("ok"):
            report["refused_write"] = True
            report["refuse_reason"] = "pre-write audit failed"
            out = Path(settings.db_path).parent / "reports" / "clip_map_restore.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("REFUSED", out)
            raise SystemExit(2)
        dest = restore_clip_id_map_atomic(clip_path, ids, backup=True)
        report["wrote_map"] = True
        report["dest"] = str(dest)
        report["backup"] = str(dest.with_name(dest.name + ".zeroed.bak"))

    index_after = {
        "exists": clip_path.exists(),
        "size": int(clip_path.stat().st_size) if clip_path.exists() else 0,
        "mtime": clip_path.stat().st_mtime if clip_path.exists() else 0,
    }
    report["index_after"] = index_after
    report["index_unchanged"] = index_before == index_after

    store = FaissStore(settings.faiss_dino_path, str(clip_path))
    report["reopen"] = {
        "clip_map_unavailable": store.clip_map_unavailable,
        "clip_map_error": store.clip_map_error,
        "clip_count": store.clip_count,
        "dino_count": store.dino_count,
        "map_len": len(store.clip_id_map),
        "needs_rebuild": store.needs_rebuild,
    }
    post = audit_clip_id_map(
        store.clip_id_map,
        clip_index_path=clip_path,
        db_path=settings.db_path,
        probe_n=5,
    )
    report["audit_after_reopen"] = {k: v for k, v in post.items() if k != "probes"}
    report["probes"] = post.get("probes")
    report["ok"] = bool(
        report["index_unchanged"]
        and not store.clip_map_unavailable
        and store.clip_count > 0
        and post.get("ok")
    )
    out = Path(settings.db_path).parent / "reports" / "clip_map_restore.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ok", "wrote_map", "skipped", "reopen") if k in report}, indent=2))
    print("audit", report["audit_after_reopen"])
    print("wrote", out)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
