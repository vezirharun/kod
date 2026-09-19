"""Vezir Visual Intelligence — comprehensive capability benchmark.

No index rebuild, no ranking rewrite. Measures current system against
human-like visual search targets.

python scripts/visual_intel_capability_benchmark.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.natural_language_query import parse_natural_query
from core.pattern_intelligence_v2 import parse_pattern_query_v2
from core.query_evidence import build_search_plan
from core.semantic_pattern_intel import parse_pattern_intent
from core.textile_terms import normalize_turkish

OBJECT_QUERIES = [
    ("yılan", ["snake", "yilan", "yılan", "python", "serpent"]),
    ("leopar", ["leopard", "leopar", "leo"]),
    ("kaplan", ["tiger", "kaplan"]),
    ("kuş", ["bird", "kus", "kuş"]),
    ("gül", ["rose", "gul", "gül"]),
    ("çiçek", ["floral", "flower", "cicek", "çiçek"]),
    ("araba", ["car", "araba", "auto"]),
    ("Togg", ["togg"]),
    ("SUV", ["suv"]),
    ("kot", ["denim", "kot", "jean"]),
]

ATTR_QUERIES = ["kırmızı gül", "mavi araba", "siyah leopar", "mavi kot"]
COMPOSITE_QUERIES = ["leopar çiçek", "leopar + çiçek", "yılan + çiçek"]
OCR_QUERIES = ["gucci"]
COUNT_ORIENT = [
    "iki leopar",
    "üç çiçek",
    "sağa bakan leopar",
    "sola bakan leopar",
    "leoparın yanında çiçek",
    "çiçeğin üzerinde leopar",
]

COUNT_TOKS = {"iki", "uc", "üç", "2", "3"}
ORIENT_TOKS = {"saga", "sağa", "sola", "sag", "sol", "bakan", "right", "left"}
REL_TOKS = {"yaninda", "yanında", "uzerine", "üzerinde", "yaninda", "beside", "over"}


def _connect(db_path: str):
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _searchable(con) -> str:
    return (
        "(status='indexed' OR COALESCE(physical_preview_ready,0)=1) "
        "AND status NOT IN ('missing','excluded_internal')"
    )


def filename_gt(con, needles: list[str]) -> list[dict]:
    where = _searchable(con)
    clauses = " OR ".join(["lower(filename) LIKE ?" for _ in needles])
    params = [f"%{n.lower()}%" for n in needles]
    rows = con.execute(
        f"SELECT id, filename, pattern_family, pattern_type FROM files "
        f"WHERE {where} AND ({clauses})",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def ocr_gt(con, needles: list[str]) -> list[dict]:
    where = _searchable(con)
    clauses = " OR ".join(["lower(COALESCE(ocr_text,'')) LIKE ?" for _ in needles])
    params = [f"%{n.lower()}%" for n in needles]
    rows = con.execute(
        f"SELECT id, filename FROM files WHERE {where} AND ({clauses})",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def pack_row(r, needles: list[str]) -> dict:
    dbg = dict(r.debug or {})
    bd = dict(r.breakdown or {})
    qev = dbg.get("query_evidence_report") or bd.get("query_evidence_report") or {}
    fname = str(r.filename or "").lower()
    fn_hit = any(n.lower() in fname for n in needles if len(n) >= 3)
    channels = []
    for key, label in (
        ("filename_score", "filename"),
        ("ocr_score", "ocr"),
        ("semantic_score", "semantic"),
        ("family_score", "family"),
        ("texture_score", "texture"),
        ("nl_score", "dna_nl"),
    ):
        if float(bd.get(key) or 0) >= 0.5:
            channels.append(label)
    if float(dbg.get("clip_score") or 0) >= 0.20:
        channels.append("clip")
    concepts = qev.get("concepts") or {}
    lex_ch = set()
    for ev in concepts.values():
        lex_ch.update((ev.get("channels") or {}).keys())
    ocr_hit = "ocr" in lex_ch
    visual_object = float(qev.get("visual_object") or 0)
    visual_only = (
        visual_object >= 0.20
        and not fn_hit
        and not ocr_hit
        and "filename" not in lex_ch
    )
    return {
        "file_id": r.file_id,
        "filename": r.filename,
        "score": round(float(r.score or 0), 4),
        "tier": dbg.get("semantic_tier") or r.category,
        "filename_lexical": fn_hit,
        "ocr_lexical": ocr_hit,
        "visual_only": visual_only,
        "clip_only": bool(dbg.get("clip_only")),
        "channels": channels,
        "clip": round(float(dbg.get("clip_score") or 0), 4),
        "visual_object": round(visual_object, 4),
        "visual_attribute": round(float(qev.get("visual_attribute") or 0), 4),
        "visual_similarity": round(float(qev.get("visual_similarity") or 0), 4),
        "visual_family": round(float(qev.get("visual_family") or 0), 4),
        "visual_conflict": bool(qev.get("visual_conflict")),
        "v2_reason": dbg.get("v2_reason") or "",
        "v2_bucket": dbg.get("v2_bucket"),
        "composite": qev.get("composite") or "",
        "conflict": bool(qev.get("conflict")),
        "concepts": {
            k: {
                "state": v.get("state"),
                "channels": v.get("channels") or {},
                "conflict_with": v.get("conflict_with") or "",
            }
            for k, v in concepts.items()
        },
        "match": (r.text_match_reason or "")[:160],
    }


def parse_slots(q: str) -> dict:
    parsed = parse_natural_query(q)
    v2 = parse_pattern_query_v2(q)
    intent = parse_pattern_intent(q)
    plan = build_search_plan(q, parsed=parsed, v2q=v2)
    norm = normalize_turkish(q)
    toks = set(re.findall(r"[a-z0-9]+", norm)) | set(norm.split())
    return {
        "nl_motif": parsed.motif,
        "nl_color": parsed.color,
        "nl_residual": parsed.residual,
        "v2_required": list(v2.required),
        "v2_composition": v2.composition,
        "v2_relationship": v2.relationship,
        "intent_motif": intent.motif,
        "intent_family": intent.family,
        "plan_objects": [c.concept_id for c in plan.objects],
        "plan_attrs": [f"{c.kind}:{c.concept_id}" for c in plan.attributes],
        "plan_and": plan.and_groups,
        "parsed_count": bool(toks & COUNT_TOKS or "iki" in norm or "uc" in norm),
        "parsed_orient": bool(toks & ORIENT_TOKS or "bakan" in norm),
        "parsed_relation": bool(
            any(t in norm for t in ("yaninda", "yanında", "uzerine", "üzerinde", "beside"))
        ),
    }


def ranking_uses_slot(slots: dict, results: list[dict], kind: str) -> str:
    if kind == "count":
        if not slots["parsed_count"]:
            return "UNSUPPORTED"
        # ranking would change n or scores vs base object query — checked later
        return "PARSED-ONLY"
    if kind == "orientation":
        if not slots["parsed_orient"]:
            return "UNSUPPORTED"
        return "PARSED-ONLY"
    if kind == "relation":
        if slots.get("v2_relationship") in ("overlay", "mixed"):
            # overlay parsed; ranking still not a dedicated slot
            return "PARSED-ONLY"
        if slots["parsed_relation"]:
            return "PARSED-ONLY"
        return "UNSUPPORTED"
    return "UNSUPPORTED"


def run_query(engine, q: str, needles: list[str], limit: int = 20) -> dict:
    t0 = time.perf_counter()
    rows = engine.search_by_text(q, limit=limit, threshold=0.55)
    dt = time.perf_counter() - t0
    meta = dict(getattr(engine, "_last_text_intel_meta", {}) or {})
    retrieve = dict(getattr(engine, "_last_retrieve_meta", {}) or {})
    packed = [pack_row(r, needles) for r in rows]
    ch_used = sorted({c for p in packed for c in p["channels"]})
    clip_unavail = bool(getattr(getattr(engine, "faiss", None), "clip_map_unavailable", True))
    return {
        "query": q,
        "n": len(rows),
        "ttfr_s": round(dt, 3),
        "total_s": round(dt, 3),
        "adaptive": retrieve,
        "rejected_competing_subtype": meta.get("rejected_competing_subtype"),
        "clip_map_unavailable": clip_unavail,
        "clip_attempted": bool(meta.get("clip_attempted")),
        "clip_hits": meta.get("clip_hits"),
        "evidence_channels": ch_used,
        "composite_pass": sum(1 for p in packed if p.get("composite") == "PASS"),
        "conflict_kept": sum(1 for p in packed if p.get("conflict")),
        "filename_lexical_in_top": sum(1 for p in packed if p.get("filename_lexical")),
        "clip_in_top": sum(1 for p in packed if p.get("clip", 0) >= 0.20),
        "visual_in_top": sum(1 for p in packed if p.get("visual_object", 0) >= 0.20),
        "visual_only_n": sum(1 for p in packed if p.get("visual_only")),
        "visual_only": [
            {
                "file_id": p["file_id"],
                "filename": p["filename"],
                "score": p["score"],
                "visual_object": p.get("visual_object"),
                "visual_similarity": p.get("visual_similarity"),
                "visual_conflict": p.get("visual_conflict"),
                "conflict": p.get("conflict"),
                "clip_only": p.get("clip_only"),
            }
            for p in packed
            if p.get("visual_only")
        ][:12],
        "top10": packed[:10],
        "top20_ids": [p["file_id"] for p in packed[:20]],
        "slots": parse_slots(q),
    }


def dna_compare(con, id_a: int, id_b: int) -> dict:
    q = """
    SELECT f.id, f.filename, f.pattern_family, f.pattern_type,
           fe.texture_map
    FROM files f LEFT JOIN features fe ON fe.file_id=f.id
    WHERE f.id IN (?, ?)
    """
    rows = {int(r["id"]): dict(r) for r in con.execute(q, (id_a, id_b))}
    a, b = rows.get(id_a) or {}, rows.get(id_b) or {}

    def tm(rec):
        raw = rec.get("texture_map") or "{}"
        try:
            return json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            return {}

    ta, tb = tm(a), tm(b)
    fa, fb = str(a.get("pattern_family") or ta.get("pattern_family") or ""), str(
        b.get("pattern_family") or tb.get("pattern_family") or ""
    )
    aa, ab = str(ta.get("animal_print_type") or a.get("pattern_type") or ""), str(
        tb.get("animal_print_type") or b.get("pattern_type") or ""
    )
    ca, cb = str(ta.get("color_family") or ""), str(tb.get("color_family") or "")
    if fa and fa == fb and aa and aa == ab:
        label = "VERY SIMILAR" if ca == cb and ca else "SAME FAMILY"
    elif fa and fa == fb:
        label = "COLOR VARIANT" if ca and cb and ca != cb else "SAME FAMILY"
    elif fa and fb:
        label = "DIFFERENT"
    else:
        label = "UNKNOWN"
    return {
        "a": {"id": id_a, "filename": a.get("filename"), "family": fa, "animal": aa, "color": ca},
        "b": {"id": id_b, "filename": b.get("filename"), "family": fb, "animal": ab, "color": cb},
        "label": label,
        "method": "indexed_metadata_not_clip",
        "clip_used": False,
    }


def main() -> None:
    from core.search_engine import SearchEngine
    from core.settings import AppSettings

    settings = AppSettings.load()
    con = _connect(settings.db_path)
    engine = SearchEngine(settings, load_ai=True)
    engine.ensure_ai_loaded()

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "constraints": {
            "production_db": "read_only",
            "faiss_rebuild": False,
            "clip_map": (
                "unavailable"
                if getattr(engine.faiss, "clip_map_unavailable", True)
                else "recovered_partial"
            ),
            "clip_count": int(engine.faiss.clip_count),
            "dino_count": int(engine.faiss.dino_count),
            "load_ai": True,
            "clip_model": bool(getattr(engine.extractor, "_clip_model", None)),
            "ground_truth": "lexical filename/OCR vs CLIP visual evidence — not auto-P",
        },
        "object": {},
        "attribute": {},
        "composite": {},
        "ocr_brand": {},
        "fine_grained": {},
        "count_orient": {},
        "visual_similarity": {},
        "visual_vs_filename": {},
        "adaptive": {},
        "capability_matrix": {},
    }

    # --- 1 object recognition ---
    for q, needles in OBJECT_QUERIES:
        print("object", q)
        gt_fn = filename_gt(con, needles)
        gt_ocr = ocr_gt(con, needles)
        run = run_query(engine, q, needles)
        hit_ids = set(run["top20_ids"])
        fn_ids = {int(x["id"]) for x in gt_fn}
        recalled = sorted(fn_ids & hit_ids)
        missed = [x for x in gt_fn if int(x["id"]) not in hit_ids]
        run["gt_filename_n"] = len(gt_fn)
        run["gt_ocr_n"] = len(gt_ocr)
        run["filename_gt_recalled_in_top20"] = len(recalled)
        run["filename_gt_missed_sample"] = [
            {"id": x["id"], "filename": x["filename"], "family": x["pattern_family"], "type": x["pattern_type"]}
            for x in missed[:12]
        ]
        report["object"][q] = run

    # --- snake audit (critical) ---
    snake_fn = filename_gt(con, ["snake", "python", "serpent", "snakeskin"])
    yilan = report["object"]["yılan"]
    yilan_ids = set(yilan["top20_ids"])
    report["visual_vs_filename"]["snake_filename_audit"] = {
        "filename_snake_n": len(snake_fn),
        "retrieved_n": yilan["n"],
        "recalled": [
            {"id": x["id"], "filename": x["filename"], "type": x["pattern_type"]}
            for x in snake_fn
            if int(x["id"]) in yilan_ids
        ],
        "missed": [
            {"id": x["id"], "filename": x["filename"], "family": x["pattern_family"], "type": x["pattern_type"]}
            for x in snake_fn
            if int(x["id"]) not in yilan_ids
        ],
        "success_if_only_3_labels": False,
        "verdict": "PASS" if yilan["n"] > 3 and yilan["filename_lexical_in_top"] >= 3 else "FAIL",
    }

    # --- 2 attributes ---
    for q in ATTR_QUERIES:
        print("attr", q)
        needles = re.findall(r"[A-Za-zçğıöşüÇĞİÖŞÜ]+", q)
        run = run_query(engine, q, needles)
        top = run["top10"]
        color_states = []
        obj_states = []
        for p in top[:8]:
            for cid, ev in (p.get("concepts") or {}).items():
                if ev.get("state"):
                    if cid in ("red_pink", "blue", "black_white", "black"):
                        color_states.append((p["filename"][:40], ev["state"], p["score"]))
                    else:
                        obj_states.append((cid, ev["state"], p["score"]))
        pass_scores = [p["score"] for p in top if p.get("composite") == "PASS"]
        contra_scores = [
            p["score"]
            for p in top
            if any(v.get("state") == "CONTRADICTED" for v in (p.get("concepts") or {}).values())
        ]
        run["color_object_and"] = {
            "pass_max": max(pass_scores) if pass_scores else None,
            "contradicted_max": max(contra_scores) if contra_scores else None,
            "and_ranks_above_contradicted": (
                bool(pass_scores and contra_scores and max(pass_scores) > max(contra_scores))
                if pass_scores and contra_scores
                else None
            ),
        }
        report["attribute"][q] = run

    # --- 3 composite ---
    for q in COMPOSITE_QUERIES:
        print("composite", q)
        run = run_query(engine, q, ["leopard", "leopar", "floral", "cicek", "snake", "yilan"])
        run["empty_and_fail"] = run["n"] == 0
        run["verdict"] = (
            "FAIL"
            if run["n"] == 0
            else ("PASS" if run["composite_pass"] >= 1 and (run["top10"][0].get("composite") == "PASS") else "PARTIAL")
        )
        report["composite"][q] = run

    # --- 4 OCR / brand ---
    print("ocr gucci")
    gucci = run_query(engine, "gucci", ["gucci", "gg"])
    ocr_only = ocr_gt(con, ["gucci"])
    fn_g = filename_gt(con, ["gucci"])
    fn_ids = {int(x["id"]) for x in fn_g}
    ocr_only_ids = [int(x["id"]) for x in ocr_only if int(x["id"]) not in fn_ids]
    hit = set(gucci["top20_ids"])
    gucci["filename_gucci_n"] = len(fn_g)
    gucci["ocr_any_n"] = len(ocr_only)
    gucci["ocr_only_ids"] = ocr_only_ids[:15]
    gucci["ocr_only_in_results"] = [i for i in ocr_only_ids if i in hit]
    gucci["ocr_on_image_without_filename"] = (
        "PASS" if gucci["ocr_only_in_results"] else "FAIL"
    )
    report["ocr_brand"]["gucci"] = gucci

    # --- 5 fine-grained ---
    report["fine_grained"] = {
        "leopar_vs_kaplan_overlap_top10": len(
            set(report["object"]["leopar"]["top20_ids"][:10])
            & set(report["object"]["kaplan"]["top20_ids"][:10])
        ),
        "yilan_vs_leopar_overlap_top10": len(
            set(report["object"]["yılan"]["top20_ids"][:10])
            & set(report["object"]["leopar"]["top20_ids"][:10])
        ),
        "unknown_not_forced": "PARTIAL — empty animal_type kept via filename; CLIP rivals still hard-drop",
        "wrong_label_kills_filename": "FIXED_FOR_SNAKE_FILENAME — zebra-without-filename still dropped",
        "togg_n": report["object"]["Togg"]["n"],
        "suv_n": report["object"]["SUV"]["n"],
        "togg_vs_car": "UNSUPPORTED_VISUAL — CLIP down; Togg not visually gated",
        "suv_vs_sedan": "UNSUPPORTED — SUV not in ontology as type slot",
        "gul_vs_floral": "PARTIAL — rose filename dominates; generic floral still enters via family",
    }

    # --- 6 count / orientation / relation ---
    for q in COUNT_ORIENT:
        print("slot", q)
        run = run_query(engine, q, ["leopard", "leopar", "floral", "cicek"])
        slots = run["slots"]
        run["count_status"] = ranking_uses_slot(slots, run["top10"], "count")
        run["orient_status"] = ranking_uses_slot(slots, run["top10"], "orientation")
        run["relation_status"] = ranking_uses_slot(slots, run["top10"], "relation")
        report["count_orient"][q] = {
            "n": run["n"],
            "slots": slots,
            "count_status": run["count_status"],
            "orient_status": run["orient_status"],
            "relation_status": run["relation_status"],
            "top3": [p["filename"] for p in run["top10"][:3]],
        }

    # --- 7 visual similarity ---
    snake_hits = report["object"]["yılan"]["top20_ids"]
    leo_hits = report["object"]["leopar"]["top20_ids"]
    pair_same = dna_compare(con, snake_hits[0], snake_hits[1]) if len(snake_hits) >= 2 else {}
    pair_diff = dna_compare(con, snake_hits[0], leo_hits[0]) if snake_hits and leo_hits else {}
    clip_on = not bool(getattr(engine.faiss, "clip_map_unavailable", True)) and bool(
        getattr(engine.extractor, "_clip_model", None)
    )
    report["visual_similarity"] = {
        "clip": "recovered_partial" if clip_on else "unavailable",
        "clip_in_yilan_top": report["object"]["yılan"].get("clip_in_top"),
        "visual_in_yilan_top": report["object"]["yılan"].get("visual_in_top"),
        "dino_used": False,
        "same_query_top2": pair_same,
        "snake_vs_leopard_top1": pair_diff,
        "verdict": (
            "PARTIAL — CLIP query-time evidence attached; pair-similarity still metadata"
            if clip_on
            else "EXPERIMENTAL — CLIP unavailable"
        ),
    }

    print("golden Togg SUV")
    report["golden_extra"] = {
        "Togg SUV": run_query(engine, "Togg SUV", ["togg", "suv"]),
    }

    golden_map = {
        "yılan": report["object"]["yılan"],
        "leopar": report["object"]["leopar"],
        "kaplan": report["object"]["kaplan"],
        "gül": report["object"]["gül"],
        "kırmızı gül": report["attribute"]["kırmızı gül"],
        "kuş": report["object"]["kuş"],
        "araba": report["object"]["araba"],
        "mavi araba": report["attribute"]["mavi araba"],
        "Togg SUV": report["golden_extra"]["Togg SUV"],
        "leopar çiçek": report["composite"]["leopar çiçek"],
        "Gucci": report["ocr_brand"]["gucci"],
    }
    visual_only_catalog = {}
    for q, run in golden_map.items():
        visual_only_catalog[q] = {
            "n": run.get("n"),
            "clip_attempted": run.get("clip_attempted"),
            "clip_in_top": run.get("clip_in_top"),
            "visual_in_top": run.get("visual_in_top"),
            "visual_only_n": run.get("visual_only_n"),
            "visual_only": run.get("visual_only") or [],
            "filename_lexical_in_top": run.get("filename_lexical_in_top"),
            "conflict_kept": run.get("conflict_kept"),
        }
    report["visual_only_catalog"] = visual_only_catalog

    # --- 8/9 conflict examples from yılan top ---
    conflicts = []
    for p in report["object"]["yılan"]["top10"]:
        for cid, ev in (p.get("concepts") or {}).items():
            if ev.get("conflict_with") or p.get("conflict"):
                conflicts.append(
                    {
                        "filename": p["filename"],
                        "state": ev.get("state"),
                        "conflict_with": ev.get("conflict_with"),
                        "channels": ev.get("channels"),
                        "killed": False,
                    }
                )
    report["visual_vs_filename"]["conflicts_in_yilan_top10"] = conflicts
    report["visual_vs_filename"]["ai_wrong_label_auto_kill"] = (
        "NO" if report["visual_vs_filename"]["snake_filename_audit"]["verdict"] == "PASS" else "YES"
    )

    # --- 10 adaptive ---
    report["adaptive"] = {
        "text_search_expand_800_1600_3200": False,
        "observed_steps": [
            {"q": q, "steps": report["object"][q].get("adaptive"), "t": report["object"][q]["total_s"]}
            for q, _ in OBJECT_QUERIES
        ],
        "visual_path_has_adaptive": True,
        "visual_path_tested": False,
        "reason": "Text search is one-shot. Adaptive 800→6400 is image/DINO FAISS.",
        "live_rerank_text": False,
        "live_rerank_visual": "implemented_via_result_callback_not_exercised",
    }

    # --- 12 matrix ---
    def st(name, status, note):
        report["capability_matrix"][name] = {"status": status, "note": note}

    y = report["object"]["yılan"]
    kg = report["attribute"]["kırmızı gül"]
    ma = report["attribute"]["mavi araba"]
    lc = report["composite"]["leopar çiçek"]
    gu = report["ocr_brand"]["gucci"]
    vo_snake = (y.get("visual_only_n") or 0) + sum(
        1 for p in (y.get("visual_only") or [])
    )
    clip_live = bool(y.get("clip_attempted")) and not y.get("clip_map_unavailable")
    st(
        "Object recognition",
        "PARTIAL" if (y["n"] and clip_live) else "PARTIAL",
        f"yılan n={y['n']} visual_in_top={y.get('visual_in_top')} visual_only={y.get('visual_only_n')} "
        f"clip_attempted={y.get('clip_attempted')}",
    )
    st(
        "Fine-grained recognition",
        "PARTIAL",
        f"leopar∩kaplan top10={report['fine_grained']['leopar_vs_kaplan_overlap_top10']}; "
        "SUV type slot still not ranked from visual.",
    )
    and_ok = (kg.get("color_object_and") or {}).get("and_ranks_above_contradicted")
    st(
        "Color understanding",
        "PARTIAL" if and_ok else "EXPERIMENTAL",
        f"kırmızı gül AND={and_ok}; mavi araba n={ma.get('n')} visual_only={ma.get('visual_only_n')}",
    )
    st(
        "OCR retrieval",
        "SUPPORTED" if gu.get("ocr_on_image_without_filename") == "PASS" else "FAIL",
        f"OCR-only Gucci in results: {gu.get('ocr_only_in_results')}",
    )
    st("Brand recognition", "PARTIAL", "Gucci filename+OCR. Exact vs semantic still conservative.")
    st("Pattern family", "SUPPORTED", "animal_print / floral family channels used in ranking.")
    st("Texture understanding", "PARTIAL", "Texture labels used; not independent of family/DNA.")
    st(
        "Composite AND",
        "PARTIAL" if lc.get("verdict") in ("PASS", "PARTIAL") else "FAIL",
        f"leopar çiçek n={lc['n']} composite_pass={lc['composite_pass']} empty_and={lc['empty_and_fail']}",
    )
    st("Count", "UNSUPPORTED", "iki/üç parsed as residual; no ranking slot.")
    st("Orientation", "UNSUPPORTED", "sağa/sola bakan parsed residual; no ranking slot.")
    st(
        "Spatial relation",
        "EXPERIMENTAL",
        "v2 overlay parse for 'üzerine'; ranking not a dedicated spatial evidence channel.",
    )
    st(
        "Visual similarity",
        "PARTIAL" if clip_live else "EXPERIMENTAL",
        report["visual_similarity"]["verdict"],
    )
    st(
        "Conflict resolution",
        "PARTIAL",
        "filename/visual vs metadata = CONFLICT keep. Auto-delete disabled.",
    )
    st(
        "Visual vs filename",
        "PARTIAL" if (y.get("visual_only_n") or 0) == 0 else "SUPPORTED",
        f"visual-only yılan={y.get('visual_only_n')} mavi araba={ma.get('visual_only_n')} "
        f"kuş={report['object']['kuş'].get('visual_only_n')}",
    )
    st("Adaptive retrieval", "PARTIAL", "Implemented on visual DINO path; unused on text search.")
    st("Live reranking", "PARTIAL", "result_callback on visual search; text search one-shot.")
    st("Explainable evidence", "SUPPORTED", "Inspector Query Evidence + visual_* channels.")

    scores = {
        "SUPPORTED": 1.0,
        "PARTIAL": 0.55,
        "EXPERIMENTAL": 0.25,
        "UNSUPPORTED": 0.0,
        "FAIL": 0.0,
    }
    vals = [scores[v["status"]] for v in report["capability_matrix"].values()]
    report["target_coverage"] = round(100.0 * sum(vals) / max(len(vals), 1), 1)

    dest = ROOT / "data" / "reports" / "visual_intel_capability_benchmark.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)
    print("coverage", report["target_coverage"])
    for k, v in report["capability_matrix"].items():
        print(f"  {k:24} {v['status']}")
    con.close()


if __name__ == "__main__":
    main()
