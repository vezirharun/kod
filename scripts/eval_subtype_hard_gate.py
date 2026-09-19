"""v1.1 subtype hard-gate precision. Does not reindex or rebuild FAISS."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = [
    "leopar",
    "yılan",
    "yılan derisi",
    "zebra",
    "kaplan",
    "çiçek",
    "gül",
    "papatya",
    "lale",
    "geometrik",
    "çizgi",
    "puantiye",
    "animal print",
    "Louis Vuitton",
]

MOTIF_TOKENS = {
    "leopard": ("leopar", "leopard", "leo"),
    "snake": ("yilan", "snake", "python", "snakeskin", "serpent"),
    "zebra": ("zebra",),
    "tiger": ("kaplan", "tiger"),
    "rose": ("gul", "gül", "rose"),
    "daisy": ("papatya", "daisy"),
    "tulip": ("lale", "tulip"),
    "polka_dot": ("puantiye", "polka", "noktali"),
    "stripe": ("cizgi", "stripe", "striped"),
    "geometric": ("geometrik", "geometric"),
    "floral": ("cicek", "floral", "flower"),
    "louis_vuitton": ("louis", "vuitton", "lv"),
}

ANIMAL = ("leopard", "snake", "zebra", "tiger")
FLORAL = ("rose", "daisy", "tulip", "orchid", "peony")


def _blob(r) -> str:
    from core.textile_terms import normalize_turkish

    tm = r.debug.get("semantic_evidence") or {}
    return normalize_turkish(
        f"{r.filename} {r.path} {tm.get('animal_type') or ''} {r.pattern_family or ''}"
    )


def _gt(query_motif: str, family: str, r) -> str:
    from core.textile_terms import normalize_turkish

    ev = r.debug.get("semantic_evidence") or {}
    animal = str(ev.get("animal_type") or "")
    pt = str(ev.get("motif") or "")
    if family == "animal_print" and animal in ANIMAL:
        return animal
    if family == "floral" and pt in FLORAL:
        return pt
    blob = normalize_turkish(f"{r.filename}")
    hits = []
    pool = ANIMAL if family == "animal_print" else FLORAL if family == "floral" else (query_motif,)
    for m in pool:
        toks = MOTIF_TOKENS.get(m, (m,))
        if any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", blob) for t in toks):
            hits.append(m)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "mixed"
    return "unknown"


def _prec(rows, motif, family, k: int) -> dict:
    top = rows[:k]
    if not top:
        return {"n": 0, "precision": 0.0, "relevant": 0, "wrong_subtype": 0, "unknown": 0}
    rel = 0
    wrong = 0
    unk = 0
    rivals = {
        "animal_print": [m for m in ANIMAL if m != motif],
        "floral": [m for m in FLORAL if m != motif],
    }.get(family, [])
    parent = motif in ("floral", "animal_print", "geometric")
    for r in top:
        lab = _gt(motif, family, r)
        if parent:
            fam = (r.debug.get("semantic_evidence") or {}).get("family") or r.pattern_family
            ok = fam == family or lab in (
                ANIMAL if family == "animal_print" else FLORAL if family == "floral" else [motif]
            )
            rel += int(ok)
            continue
        if lab == motif:
            rel += 1
        elif lab in rivals or lab == "mixed":
            wrong += 1
        else:
            unk += 1
    return {
        "n": len(top),
        "precision": round(rel / len(top), 3),
        "relevant": rel,
        "wrong_subtype": wrong,
        "unknown": unk,
    }


def main() -> None:
    from core.search_engine import SearchEngine
    from core.search_models import SearchQuery
    from core.semantic_pattern_intel import parse_pattern_intent
    from core.settings import AppSettings

    settings = AppSettings.load()
    settings.semantic_pattern_intel_enabled = True
    engine = SearchEngine(settings, load_ai=True)
    engine.ensure_ai_loaded()
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "clip_loaded": bool(getattr(engine.extractor, "_clip_model", None)),
        "faiss_clip": engine.faiss.clip_count,
        "queries": {},
    }
    lines = []
    for q in QUERIES:
        intent = parse_pattern_intent(q)
        resp = engine.execute_search(SearchQuery(mode="text", text=q, threshold=0.45))
        rows = list(resp.results or [])[:20]
        meta = resp.meta or {}
        p10 = _prec(rows, intent.motif, intent.family, 10)
        p20 = _prec(rows, intent.motif, intent.family, 20)
        top = [
            {
                "filename": r.filename,
                "score": round(float(r.score), 3),
                "tier": r.debug.get("semantic_tier"),
                "clip": round(float(r.debug.get("clip_score") or 0), 3),
                "margin": round(float(r.debug.get("clip_margin") or 0), 3),
                "neg": r.debug.get("negative_motifs") or [],
                "gt": _gt(intent.motif, intent.family, r),
                "visual_win": bool(r.debug.get("visual_win")),
            }
            for r in rows[:10]
        ]
        rec = {
            "intent": intent.as_key(),
            "n": len(rows),
            "precision@10": p10,
            "precision@20": p20,
            "clip_used": bool(meta.get("ai_faiss_used") or meta.get("clip_attempted")),
            "clip_hits": meta.get("clip_hits"),
            "rejected_competing_subtype": meta.get("rejected_competing_subtype"),
            "negative_evidence": meta.get("negative_evidence"),
            "dropped": meta.get("dropped_low_confidence"),
            "tiers": meta.get("tiers"),
            "top10": top,
        }
        out["queries"][q] = rec
        line = (
            f"{q:16} n={len(rows):3} p@10={p10['precision']:.3f} "
            f"(wrong={p10['wrong_subtype']}) p@20={p20['precision']:.3f} "
            f"(wrong={p20['wrong_subtype']}) "
            f"rejected={meta.get('rejected_competing_subtype')} clip={meta.get('ai_faiss_used')}"
        )
        print(line)
        lines.append(line)
    dest = ROOT / "data" / "reports" / "subtype_hard_gate_precision.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
