"""Read-only leopard chain audit. No DB/FAISS/ranking writes.

python scripts/leopard_chain_audit.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERY = "leopar"
DISPLAY_N = 24


def _connect_ro(db_path: str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con


def _searchable() -> str:
    return (
        "(f.status='indexed' OR COALESCE(f.physical_preview_ready,0)=1) "
        "AND f.status NOT IN ('missing','excluded_internal')"
    )


def _nt_has(blob: str, needles: tuple[str, ...]) -> bool:
    from core.textile_terms import normalize_turkish

    b = normalize_turkish(blob or "")
    return any(normalize_turkish(n) in b for n in needles if n)


def load_index_gt(db_path: str) -> dict:
    con = _connect_ro(db_path)
    where = _searchable()
    total = con.execute(f"SELECT COUNT(*) FROM files f WHERE {where}").fetchone()[0]
    fname_sql = (
        f"SELECT f.id, f.filename, f.ocr_text, f.pattern_family, f.pattern_type, "
        f"f.pattern_subtype, fe.texture_map, "
        f"(fe.clip_embedding IS NOT NULL AND length(fe.clip_embedding)>0) AS has_clip "
        f"FROM files f LEFT JOIN features fe ON fe.file_id=f.id "
        f"WHERE {where} AND ("
        f" lower(f.filename) LIKE '%leopard%' OR lower(f.filename) LIKE '%leopar%'"
        f" OR lower(COALESCE(f.ocr_text,'')) LIKE '%leopard%'"
        f" OR lower(COALESCE(f.ocr_text,'')) LIKE '%leopar%'"
        f" OR lower(COALESCE(f.pattern_type,'')) LIKE '%leopard%'"
        f" OR lower(COALESCE(f.pattern_subtype,'')) LIKE '%leopard%'"
        f" OR lower(COALESCE(fe.texture_map,'')) LIKE '%\"animal_print_type\": \"leopard\"%'"
        f" OR lower(COALESCE(fe.texture_map,'')) LIKE '%\"animal_print_type\":\"leopard\"%'"
        f")"
    )
    rows = [dict(r) for r in con.execute(fname_sql).fetchall()]
    con.close()

    filename: set[int] = set()
    ocr: set[int] = set()
    ai: set[int] = set()
    family_animal: set[int] = set()
    has_clip: set[int] = set()
    meta: dict[int, dict] = {}
    for r in rows:
        fid = int(r["id"])
        fn = str(r.get("filename") or "")
        oc = str(r.get("ocr_text") or "")
        pt = str(r.get("pattern_type") or "") + " " + str(r.get("pattern_subtype") or "")
        tm_raw = r.get("texture_map") or ""
        animal = ""
        pf = str(r.get("pattern_family") or "")
        if tm_raw:
            try:
                tm = json.loads(tm_raw) if isinstance(tm_raw, str) else tm_raw
            except Exception:
                tm = {}
            if isinstance(tm, dict):
                animal = str(tm.get("animal_print_type") or "")
                pf = pf or str(tm.get("pattern_family") or "")
        if _nt_has(fn, ("leopard", "leopar")):
            filename.add(fid)
        if _nt_has(oc, ("leopard", "leopar")):
            ocr.add(fid)
        if animal.lower() == "leopard" or _nt_has(pt, ("leopard",)):
            ai.add(fid)
        if pf == "animal_print":
            family_animal.add(fid)
        if r.get("has_clip"):
            has_clip.add(fid)
        meta[fid] = {
            "filename": fn,
            "animal": animal,
            "pattern_family": pf,
            "pattern_type": str(r.get("pattern_type") or ""),
            "has_clip_db": bool(r.get("has_clip")),
        }
    union = filename | ocr | ai
    return {
        "indexed_searchable": total,
        "filename": filename,
        "ocr": ocr,
        "ai": ai,
        "family_animal_print": family_animal,
        "union_gt": union,
        "filename_and_ai": filename & ai,
        "filename_not_ai": filename - ai,
        "ai_not_filename": ai - filename,
        "has_clip_db": has_clip,
        "meta": meta,
    }


def _ids(rows) -> set[int]:
    out = set()
    for r in rows or []:
        fid = getattr(r, "file_id", None)
        if fid is None and isinstance(r, dict):
            fid = r.get("id") or r.get("file_id")
        if fid is not None:
            out.add(int(fid))
    return out


def run_search_audit(engine, gt: dict, *, limit: int) -> dict:
    from core.semantic_pattern_intel import cap_confidence_padding, parse_pattern_intent

    stages: dict[str, set[int]] = {}
    drop_reasons: Counter[str] = Counter()
    gate_drop: dict[int, str] = {}

    orig_cands = engine.db.search_text_candidates
    orig_clip = engine._clip_text_hits
    orig_merge = engine._merge_semantic_intel
    orig_cap = cap_confidence_padding

    def cands(*a, **k):
        rows = orig_cands(*a, **k)
        stages["text_candidates"] = {int(r["id"]) for r in rows}
        return rows

    def clip(*a, **k):
        hits, meta = orig_clip(*a, **k)
        stages["clip_hits"] = set(int(i) for i in hits)
        stages["_clip_meta"] = meta  # type: ignore
        return hits, meta

    def merge(text, intent, results, clip_by_id, customer, plan=None):
        stages["merge_in_text"] = _ids(results)
        stages["merge_in_clip"] = set(int(i) for i in (clip_by_id or {}))
        # Probe gate reasons only for GT ids that enter merge.
        from core.semantic_pattern_intel import collect_evidence, gate_result

        pool = stages["merge_in_text"] | stages["merge_in_clip"]
        need = [fid for fid in (pool & gt["union_gt"]) if fid]
        recs = {}
        if need:
            recs = {
                int(r["id"]): r
                for r in engine.db.get_indexed_files_by_ids(
                    need, include_processing_ready=True, lightweight=True
                )
            }
        for fid in need:
            rec = recs.get(fid) or {}
            clip = float((clip_by_id or {}).get(fid) or 0)
            rivals = dict((getattr(engine, "_last_clip_rivals", {}) or {}).get(fid) or {})
            if intent and intent.motif and clip and intent.motif not in rivals:
                rivals[str(intent.motif)] = clip
            ev = collect_evidence(intent, rec, clip_score=clip, rival_clip=rivals)
            ok, tier, _ = gate_result(intent, ev, text_score=0.5)
            if not ok:
                gate_drop[fid] = tier or "gate_drop"
        out = orig_merge(text, intent, results, clip_by_id, customer, plan=plan)
        stages["after_merge"] = _ids(out)
        return out

    import core.semantic_pattern_intel as spi

    def cap(intent, rows, *, tier_attr="semantic_tier"):
        stages["before_cap"] = _ids(rows)
        out = orig_cap(intent, rows, tier_attr=tier_attr)
        stages["after_cap"] = _ids(out)
        return out

    engine.db.search_text_candidates = cands
    engine._clip_text_hits = clip
    engine._merge_semantic_intel = merge
    spi.cap_confidence_padding = cap

    try:
        rows = engine.search_by_text(QUERY, limit=limit, threshold=0.55)
    finally:
        engine.db.search_text_candidates = orig_cands
        engine._clip_text_hits = orig_clip
        engine._merge_semantic_intel = orig_merge
        spi.cap_confidence_padding = orig_cap

    stages["after_v2_uvi_final"] = _ids(rows)
    ranked = list(rows)
    stages["displayed_24"] = _ids(ranked[:DISPLAY_N])
    meta = dict(getattr(engine, "_last_text_intel_meta", {}) or {})
    uvi = dict(meta.get("universal_visual_intel") or {})
    v2 = dict(meta.get("pattern_intelligence_v2") or {})

    clip_meta = stages.pop("_clip_meta", {}) if "_clip_meta" in stages else meta
    if isinstance(clip_meta, dict) and "clip_k" not in clip_meta:
        clip_meta = {**clip_meta, **{k: meta.get(k) for k in ("clip_k", "clip_hits", "adaptive_clip")}}

    # Reconstruct drop reasons for GT missing from final
    final = stages.get("after_v2_uvi_final") or set()
    missing = []
    union = gt["union_gt"]
    clip_map = set()
    try:
        clip_map = {int(x) for x in (engine.faiss.clip_id_map or []) if int(x) > 0}
    except Exception:
        clip_map = set()

    for fid in sorted(union):
        if fid in final:
            continue
        reasons = []
        info = gt["meta"].get(fid) or {}
        if fid not in clip_map:
            reasons.append("no_clip_faiss_map")
        if not info.get("has_clip_db"):
            reasons.append("no_clip_db_embedding")
        if fid not in (stages.get("text_candidates") or set()):
            reasons.append("not_in_text_candidates")
        if fid not in (stages.get("clip_hits") or set()):
            reasons.append("not_in_clip_hits")
        if fid in (stages.get("merge_in_text") or set()) | (stages.get("merge_in_clip") or set()):
            if fid not in (stages.get("after_merge") or set()):
                reasons.append("dropped_semantic_gate:" + (gate_drop.get(fid) or meta.get("rejected_competing_subtype") and "rival" or "low_confidence"))
        if fid in (stages.get("before_cap") or set()) and fid not in (stages.get("after_cap") or set()):
            reasons.append("dropped_semantic_padding_cap")
        if fid in (stages.get("after_cap") or set()) and fid not in final:
            reasons.append("dropped_v2_or_uvi_or_limit")
        if fid in final and fid not in (stages.get("displayed_24") or set()):
            reasons.append("ranked_below_display_24")
        if not reasons:
            reasons.append("unknown")
        for rsn in reasons:
            drop_reasons[rsn] += 1
        missing.append(
            {
                "file_id": fid,
                "filename": info.get("filename") or "",
                "gt": {
                    "filename": fid in gt["filename"],
                    "ocr": fid in gt["ocr"],
                    "ai": fid in gt["ai"],
                },
                "animal": info.get("animal") or "",
                "reasons": reasons,
            }
        )

    def _gt_overlap(key: str) -> dict[str, int]:
        s = stages.get(key) or set()
        return {
            "n": len(s),
            "gt_union": len(s & union),
            "gt_filename": len(s & gt["filename"]),
            "gt_ai": len(s & gt["ai"]),
        }

    return {
        "limit": limit,
        "returned": len(ranked),
        "clip_meta": {
            "clip_k": (clip_meta or {}).get("clip_k") or meta.get("clip_k"),
            "clip_hits": (clip_meta or {}).get("clip_hits") or meta.get("clip_hits"),
            "adaptive_clip": (clip_meta or {}).get("adaptive_clip") or meta.get("adaptive_clip"),
            "first_clip_ms": meta.get("first_clip_ms"),
        },
        "intel": {
            "dropped_low_confidence": meta.get("dropped_low_confidence"),
            "rejected_competing_subtype": meta.get("rejected_competing_subtype"),
            "clip_contributed": meta.get("clip_contributed"),
            "uvi_dropped_unspecific": uvi.get("dropped_unspecific"),
            "uvi_dropped_mismatch": uvi.get("dropped_mismatch"),
            "v2": {k: v2.get(k) for k in list(v2)[:12]},
        },
        "stages": {k: _gt_overlap(k) for k in (
            "text_candidates",
            "clip_hits",
            "merge_in_text",
            "merge_in_clip",
            "after_merge",
            "before_cap",
            "after_cap",
            "after_v2_uvi_final",
            "displayed_24",
        ) if k in stages or k == "displayed_24"},
        "drop_reason_counts": dict(drop_reasons),
        "missing_gt_n": len(missing),
        "missing_gt_sample": missing[:40],
        "top24": [
            {
                "file_id": int(r.file_id),
                "filename": r.filename,
                "score": round(float(r.score or 0), 4),
                "in_filename_gt": int(r.file_id) in gt["filename"],
                "in_ai_gt": int(r.file_id) in gt["ai"],
                "clip_only": bool((r.debug or {}).get("clip_only")),
            }
            for r in ranked[:DISPLAY_N]
        ],
    }


def main() -> None:
    from core.search_engine import SearchEngine
    from core.settings import AppSettings

    settings = AppSettings.load()
    gt = load_index_gt(settings.db_path)
    print("indexed_searchable", gt["indexed_searchable"])
    print("filename leopard/leopar", len(gt["filename"]))
    print("ocr leopard/leopar", len(gt["ocr"]))
    print("ai leopard", len(gt["ai"]))
    print("union GT", len(gt["union_gt"]))
    print("filename AND ai", len(gt["filename_and_ai"]))
    print("filename NOT ai", len(gt["filename_not_ai"]))
    print("ai NOT filename", len(gt["ai_not_filename"]))

    engine = SearchEngine(settings, load_ai=True)
    clip_map_n = 0
    try:
        clip_map_n = sum(1 for x in engine.faiss.clip_id_map if int(x) > 0)
    except Exception:
        clip_map_n = 0
    gt_in_clip = len(gt["union_gt"] & {int(x) for x in (engine.faiss.clip_id_map or []) if int(x) > 0}) if clip_map_n else 0
    print("clip_map_positive", clip_map_n, "union_gt_in_clip_map", gt_in_clip)

    audits = {}
    for limit in (40, 400):
        print(f"\n=== search limit={limit} ===")
        audits[str(limit)] = run_search_audit(engine, gt, limit=limit)
        a = audits[str(limit)]
        print("returned", a["returned"], "clip_k", a["clip_meta"].get("clip_k"), "clip_hits", a["clip_meta"].get("clip_hits"))
        for k, v in a["stages"].items():
            print(f"  {k:24} n={v['n']:5} gt_union={v['gt_union']:4} gt_fn={v['gt_filename']:4} gt_ai={v['gt_ai']:4}")
        print("drop_reasons", a["drop_reason_counts"])
        print("missing_gt", a["missing_gt_n"])

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": QUERY,
        "read_only": True,
        "index": {
            "indexed_searchable": gt["indexed_searchable"],
            "filename_leopard_leopar": len(gt["filename"]),
            "ocr_leopard_leopar": len(gt["ocr"]),
            "ai_leopard": len(gt["ai"]),
            "union_gt": len(gt["union_gt"]),
            "filename_and_ai": len(gt["filename_and_ai"]),
            "filename_not_ai": len(gt["filename_not_ai"]),
            "ai_not_filename": len(gt["ai_not_filename"]),
            "clip_map_positive": clip_map_n,
            "union_gt_in_clip_map": gt_in_clip,
        },
        "audits": audits,
    }
    dest = ROOT / "data" / "reports" / "leopard_chain_audit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
