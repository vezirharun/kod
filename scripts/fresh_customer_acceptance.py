"""Fresh-customer acceptance — no teaching, no production writes.

Simulates a first-time Vezir user: folder already scanned (existing CLIP/OCR/DNA),
but user labels / family overrides / Visual Memory claimed_class are ignored.

python scripts/fresh_customer_acceptance.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = (
    "yılan",
    "leopar",
    "kaplan",
    "gül",
    "kuş",
    "kırmızı gül",
    "mavi araba",
    "Togg SUV",
    "leopar çiçek",
    "yılan derisi",
    "leopar desenli kumaş",
    "gucci",
)

# Query → expected visual/object tokens vs rival tokens.
EXPECT = {
    "yılan": {"want": ("snake", "yilan", "yılan", "python"), "rival": ("leopard", "leopar", "tiger", "kaplan", "zebra")},
    "leopar": {"want": ("leopard", "leopar"), "rival": ("snake", "yilan", "tiger", "kaplan", "zebra")},
    "kaplan": {"want": ("tiger", "kaplan"), "rival": ("leopard", "leopar", "snake", "zebra")},
    "gül": {"want": ("rose", "gul", "gül"), "rival": ("leopard", "snake", "denim", "kot")},
    "kuş": {"want": ("bird", "kus", "kuş", "karga", "leylek"), "rival": ("denim", "kot", "jeans", "car")},
    "kırmızı gül": {"want": ("rose", "gul", "gül"), "rival": ("leopard", "snake"), "composite": ("red", "kirmizi", "kırmızı", "red_pink")},
    "mavi araba": {"want": ("car", "araba", "togg", "otomobil"), "rival": ("denim", "kot", "floral", "dantel")},
    "Togg SUV": {"want": ("togg",), "rival": ("gucci", "floral"), "composite": ("suv",)},
    "leopar çiçek": {"want": ("leopard", "leopar"), "rival": ("snake",), "composite": ("floral", "flower", "cicek", "çiçek", "rose")},
    "yılan derisi": {"want": ("snake", "yilan", "yılan", "skin", "deri"), "rival": ("leopard", "leopar")},
    "leopar desenli kumaş": {"want": ("leopard", "leopar"), "rival": ("snake", "tiger")},
    "gucci": {"want": ("gucci", "gg"), "rival": (), "ocr_ok": True},
}

UNSUPPORTED_IF = {
    "iki leopar": "count",
    "sağa bakan leopar": "orientation",
}


def _tok(s: str) -> str:
    from core.textile_terms import normalize_turkish

    return normalize_turkish(s or "")


def _has_any(blob: str, needles: tuple[str, ...]) -> bool:
    b = _tok(blob)
    return any(_tok(n) and _tok(n) in b for n in needles)


def isolate_engine(engine) -> dict:
    """Drop customer teaching paths without writing DB/FAISS."""
    notes = {"family_overrides": "stubbed", "feedback": "stubbed", "visual_memory_claimed_class": "not_read"}
    engine._records_with_family_overrides = lambda recs, customer="": recs
    fb = getattr(engine, "_feedback", None)
    if fb is not None:
        fb.score_adjustments = lambda text: {}
        fb.wrong_result_ids = lambda text: set()
    return notes


def primary_channel(p: dict) -> str:
    vis = float(p.get("visual_object") or 0) >= 0.20 and str(p.get("visual_grade") or "") in (
        "visual_exact",
        "visual_strong",
    )
    ocr = float(p.get("ocr_score") or 0) >= 0.5 or p.get("ocr_hit")
    fname = float(p.get("filename_score") or 0) >= 0.5 or p.get("filename_hit")
    meta = p.get("metadata_hit")
    if vis and not fname and not ocr:
        return "visual"
    if vis and (fname or ocr):
        return "visual+lexical"
    if ocr and not vis:
        return "ocr"
    if fname and not vis:
        return "filename"
    if meta and not vis:
        return "metadata"
    return "coincidence"


def pack_row(r, q: str, spec: dict) -> dict:
    dbg = dict(r.debug or {})
    bd = dict(r.breakdown or {})
    qev = dbg.get("query_evidence_report") or bd.get("query_evidence_report") or {}
    concepts = qev.get("concepts") or {}
    fname = str(r.filename or "")
    ch_visual = False
    ch_ocr = False
    ch_fn = False
    ch_meta = False
    for ev in concepts.values():
        ch = ev.get("channels") or {}
        if float(ch.get("visual") or 0) >= 0.20:
            ch_visual = True
        if float(ch.get("ocr") or 0) > 0:
            ch_ocr = True
        if float(ch.get("filename") or 0) > 0:
            ch_fn = True
        if any(float(ch.get(k) or 0) > 0 for k in ("family", "texture", "dna", "semantic")):
            ch_meta = True
    want = spec.get("want") or ()
    rival = spec.get("rival") or ()
    lexical_want = _has_any(fname, want)
    lexical_rival = _has_any(fname, rival)
    row = {
        "file_id": getattr(r, "file_id", None),
        "filename": fname,
        "score": round(float(r.score or 0), 4),
        "clip_only": bool(dbg.get("clip_only")),
        "visual_grade": dbg.get("visual_grade") or qev.get("visual_grade") or "",
        "visual_verdict": dbg.get("visual_verdict") or qev.get("visual_verdict") or "",
        "visual_object": round(float(qev.get("visual_object") or 0), 4),
        "visual_conflict": bool(qev.get("visual_conflict")),
        "conflict": bool(qev.get("conflict")),
        "composite": qev.get("composite") or "",
        "ocr_score": round(float(bd.get("ocr_score") or 0), 4),
        "filename_score": round(float(bd.get("filename_score") or 0), 4),
        "ocr_hit": ch_ocr,
        "filename_hit": ch_fn or lexical_want,
        "metadata_hit": ch_meta,
        "lexical_want": lexical_want,
        "lexical_rival": lexical_rival,
        "user_feedback_delta": round(float(dbg.get("user_feedback_delta") or 0), 4),
        "family_override": bool(dbg.get("feedback_family_override") or dbg.get("family_override")),
    }
    row["channel"] = primary_channel(row)
    if row["channel"] == "coincidence" and ch_visual:
        row["channel"] = "visual"
    return row


def judge_query(q: str, packed: list[dict], spec: dict) -> str:
    if q in UNSUPPORTED_IF:
        return "UNSUPPORTED"
    if not packed:
        return "FAIL"
    top = packed[:5]
    visual_n = sum(1 for p in top if str(p.get("channel") or "").startswith("visual"))
    rival_n = sum(1 for p in top if p.get("lexical_rival") and not p.get("lexical_want"))
    ocr_ok = bool(spec.get("ocr_ok"))
    ocr_n = sum(1 for p in top if p.get("channel") == "ocr" or (ocr_ok and p.get("ocr_hit")))
    fname_only = sum(1 for p in top if p.get("channel") == "filename")
    coincidence = sum(1 for p in top if p.get("channel") == "coincidence")
    composite_need = spec.get("composite")
    composite_pass = sum(1 for p in packed if p.get("composite") == "PASS")

    t1 = top[0]
    if t1.get("lexical_rival") and not t1.get("lexical_want") and not str(t1.get("channel") or "").startswith("visual"):
        return "FAIL"
    if ocr_ok and (ocr_n >= 1 or t1.get("ocr_hit") or t1.get("lexical_want")):
        if visual_n or ocr_n:
            return "PASS" if (ocr_n >= 1 or t1.get("channel") in ("ocr", "visual", "visual+lexical", "filename")) else "PARTIAL"
    if composite_need:
        if composite_pass >= 1 and visual_n >= 1:
            return "PARTIAL" if composite_pass < 3 else "PASS"
        if visual_n >= 2:
            return "PARTIAL"
        return "FAIL"
    if visual_n >= 3 and rival_n == 0 and coincidence <= 1:
        return "PASS"
    if visual_n >= 1 and fname_only <= 3:
        return "PARTIAL"
    if fname_only >= 3 and visual_n == 0:
        return "FAIL"
    if coincidence >= 3 and visual_n == 0:
        return "FAIL"
    return "PARTIAL" if packed else "FAIL"


def channel_counts(packed: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in packed[:10]:
        ch = str(p.get("channel") or "coincidence")
        out[ch] = out.get(ch, 0) + 1
    return out


def main() -> None:
    from core.search_engine import SearchEngine
    from core.settings import AppSettings
    from core.visual_memory.knowledge import knowledge_pack_info

    subprocess.check_call(
        [sys.executable, str(ROOT / "scripts" / "build_global_knowledge_pack.py")]
    )
    pack_info = knowledge_pack_info()
    settings = AppSettings.load()
    engine = SearchEngine(settings, load_ai=True)
    isolation = isolate_engine(engine)
    faiss = getattr(engine, "faiss", None)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "fresh_customer",
        "teaching": False,
        "production_db_writes": False,
        "faiss_rebuild": False,
        "isolation": isolation,
        "knowledge_pack": pack_info,
        "clip_map_unavailable": bool(getattr(faiss, "clip_map_unavailable", True)),
        "clip_count": int(getattr(faiss, "clip_count", 0) or 0),
        "queries": {},
        "summary": {},
    }
    print(
        f"Knowledge Pack v{pack_info.get('version')} "
        f"customer_memory={pack_info.get('customer_memory')} "
        f"CLIP={report['clip_count']}"
    )
    verdicts: dict[str, int] = {}
    for q in QUERIES:
        spec = EXPECT.get(q) or {"want": (), "rival": ()}
        t0 = time.perf_counter()
        rows = engine.search_by_text(q, limit=20, threshold=0.55, customer="")
        dt = time.perf_counter() - t0
        packed = [pack_row(r, q, spec) for r in rows]
        # Fresh customer: teaching must stay unused.
        taught = sum(
            1
            for p in packed
            if abs(float(p.get("user_feedback_delta") or 0)) > 1e-9 or p.get("family_override")
        )
        verdict = judge_query(q, packed, spec)
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        ch = channel_counts(packed)
        report["queries"][q] = {
            "verdict": verdict,
            "n": len(rows),
            "total_s": round(dt, 3),
            "taught_hits": taught,
            "channels_top10": ch,
            "visual_n": sum(1 for p in packed if str(p.get("channel")).startswith("visual")),
            "ocr_n": sum(1 for p in packed if p.get("channel") == "ocr" or p.get("ocr_hit")),
            "filename_n": sum(1 for p in packed if p.get("channel") == "filename"),
            "metadata_n": sum(1 for p in packed if p.get("channel") == "metadata"),
            "coincidence_n": sum(1 for p in packed if p.get("channel") == "coincidence"),
            "composite_pass": sum(1 for p in packed if p.get("composite") == "PASS"),
            "top10": packed[:10],
        }
        print(
            f"{verdict:12} {q!r:28} n={len(rows):2} t={dt:.1f}s "
            f"vis={report['queries'][q]['visual_n']} "
            f"fn={report['queries'][q]['filename_n']} "
            f"ocr={report['queries'][q]['ocr_n']} "
            f"coin={report['queries'][q]['coincidence_n']}"
        )
        for i, p in enumerate(packed[:5], 1):
            print(f"    {i}. {p['channel']:16} {p['score']:.3f} {p['filename'][:50]}")

    report["summary"] = {
        "verdicts": verdicts,
        "queries": len(QUERIES),
        "gaps": [
            q
            for q, row in report["queries"].items()
            if row["verdict"] in ("FAIL", "PARTIAL", "UNSUPPORTED")
        ],
    }
    dest = ROOT / "data" / "reports" / "fresh_customer_acceptance.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)
    print("summary", json.dumps(verdicts, ensure_ascii=False))


if __name__ == "__main__":
    main()
