"""Read-only semantic/text search capacity eval. Does not modify engine or DB."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.search_engine import SearchEngine
from core.search_models import SearchQuery
from core.settings import AppSettings
from core.textile_terms import normalize_turkish, query_family_hints

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "reports" / "semantic_search_capacity.json"

QUERIES_BASIC = [
    "puantiye", "noktalı", "polka dot", "çiçek", "floral", "papatya", "gül",
    "lale", "yaprak", "geometrik", "çizgi", "çizgili", "ekose", "kareli",
    "leopar", "zebra", "yılan derisi", "mermer", "kamuflaj", "paisley", "monogram",
]
QUERIES_FLOWER = ["çiçek", "papatya", "gül", "lale", "orkide", "şakayık", "ayçiçeği", "yaprak"]
QUERIES_DOT = ["puantiye", "polka dot", "dot", "noktalı"]
QUERIES_BRAND = [
    "Louis Vuitton", "LV", "Gucci", "GG", "Chanel", "Dior", "YSL", "Fendi", "FF",
]
QUERIES_NL = [
    "küçük çiçekli kumaş",
    "beyaz zemin üzerine siyah puantiye",
    "kahverengi leopar desen",
    "çizgili geometrik desen",
    "küçük papatyalı floral desen",
    "LV tarzı monogram desen",
]

NAME_TOKENS = {
    "puantiye": ("puantiye", "puan", "polka", "dot", "nokta"),
    "noktalı": ("nokta", "puantiye", "polka", "dot"),
    "polka dot": ("polka", "dot", "puantiye"),
    "dot": ("dot", "polka", "puantiye"),
    "çiçek": ("cicek", "çiçek", "flower", "floral", "fiori"),
    "floral": ("floral", "flower", "cicek", "çiçek", "fiori"),
    "papatya": ("papatya", "daisy"),
    "gül": ("gul", "gül", "rose"),
    "lale": ("lale", "tulip"),
    "orkide": ("orkide", "orchid"),
    "şakayık": ("sakayik", "şakayık", "peony"),
    "ayçiçeği": ("aycicegi", "ayçiçeği", "sunflower"),
    "yaprak": ("yaprak", "leaf"),
    "geometrik": ("geo", "geometric", "geometrik"),
    "çizgi": ("cizgi", "çizgi", "stripe", "line"),
    "çizgili": ("cizgili", "çizgili", "stripe"),
    "ekose": ("ekose", "plaid", "check", "tartan"),
    "kareli": ("kareli", "check", "plaid", "ekose"),
    "leopar": ("leopar", "leopard"),
    "zebra": ("zebra",),
    "yılan derisi": ("yilan", "yılan", "snake", "serpent"),
    "mermer": ("mermer", "marble"),
    "kamuflaj": ("kamuflaj", "camo", "camouflage"),
    "paisley": ("paisley", "sal", "şal"),
    "monogram": ("monogram", "logo", "lv", "gucci"),
    "Louis Vuitton": ("louis", "vuitton", "lv"),
    "LV": ("lv", "louis", "vuitton"),
    "Gucci": ("gucci", "gg"),
    "GG": ("gg", "gucci"),
    "Chanel": ("chanel",),
    "Dior": ("dior",),
    "YSL": ("ysl", "laurent"),
    "Fendi": ("fendi",),
    "FF": ("ff", "fendi"),
}


def _norm(s: str) -> str:
    return normalize_turkish(s or "").lower()


def name_hit(q: str, filename: str, path: str) -> bool:
    blob = _norm(f"{filename} {path}")
    toks = NAME_TOKENS.get(q)
    if not toks:
        toks = tuple(_norm(q).replace("-", " ").split())
    return any(_norm(t) in blob for t in toks if t)


def parse_tm(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def run_one(engine: SearchEngine, q: str, threshold: float) -> dict:
    t0 = time.perf_counter()
    resp = engine.execute_search(SearchQuery(mode="text", text=q, threshold=threshold))
    ms = (time.perf_counter() - t0) * 1000
    above = [r for r in resp.results if r.score >= threshold]
    hints = resp.meta.get("query_family_hints") or {}
    want_fam = str(hints.get("pattern_family") or "")
    scores = [r.score for r in above[:50]]
    src_hits = Counter()
    fams = Counter()
    exts = Counter()
    visual_only = 0
    text_only = 0
    mixed = 0
    true_n = 0
    false_n = 0
    unlabeled = 0
    top = []
    for i, r in enumerate(above[:30]):
        bd = dict(r.breakdown or r.text_score_breakdown or {})
        fname = str(r.filename or "")
        path = str(r.path or "")
        nh = name_hit(q, fname, path)
        ocr_s = float(bd.get("ocr_score") or 0)
        fam_s = float(bd.get("family_score") or 0)
        sem_s = float(bd.get("semantic_score") or 0)
        nl_s = float(bd.get("nl_score") or 0)
        clip_s = float(bd.get("clip") or 0)
        dino_s = float(bd.get("dino") or 0)
        fam = str(r.pattern_family or "")
        ext = Path(path).suffix.lower() or "?"
        exts[ext] += 1
        fams[fam] += 1
        if nh:
            src_hits["filename"] += 1
        if ocr_s >= 0.3:
            src_hits["ocr"] += 1
        if fam_s >= 0.4 or sem_s >= 0.3:
            src_hits["texture_semantic"] += 1
        if float(bd.get("brand_alias_score") or 0) >= 0.5:
            src_hits["brand"] += 1
        if clip_s > 0:
            src_hits["clip"] += 1
        if dino_s > 0:
            src_hits["dino"] += 1
        vis = (not nh) and ocr_s < 0.3
        txt = nh or ocr_s >= 0.5
        if vis and (fam_s >= 0.4 or sem_s >= 0.3 or nl_s >= 0.5):
            visual_only += 1
        elif txt and fam_s < 0.3 and sem_s < 0.2:
            text_only += 1
        else:
            mixed += 1
        if want_fam:
            if fam == want_fam or (want_fam == "floral" and fam == "floral") or nh:
                true_n += 1
            elif fam and fam not in ("unknown", "", want_fam) and not nh:
                false_n += 1
            else:
                unlabeled += 1
        elif nh:
            true_n += 1
        else:
            unlabeled += 1
        if i < 12:
            top.append(
                {
                    "filename": fname,
                    "score": round(r.score, 3),
                    "family": fam,
                    "name_hit": nh,
                    "reason": (r.text_match_reason or "")[:180],
                    "fn": round(float(bd.get("filename_score") or 0), 3),
                    "ocr": round(ocr_s, 3),
                    "family_sc": round(fam_s, 3),
                    "sem": round(sem_s, 3),
                    "nl": round(nl_s, 3),
                    "brand": round(float(bd.get("brand_alias_score") or 0), 3),
                    "clip": round(clip_s, 3),
                    "ext": ext,
                }
            )
    n = max(1, min(30, len(above)))
    return {
        "query": q,
        "want_family": want_fam,
        "spell": resp.meta.get("spell_corrected") or "",
        "method": resp.meta.get("search_method"),
        "candidates_scored": int(getattr(resp.stats, "candidates_evaluated", 0) or 0),
        "above": len(above),
        "displayed": int(getattr(resp.stats, "displayed", 0) or len(above)),
        "top_k": min(12, len(above)),
        "true_top30": true_n,
        "false_top30": false_n,
        "unlabeled_top30": unlabeled,
        "visual_label_only": visual_only,
        "filename_ocr_only": text_only,
        "mixed": mixed,
        "sources": dict(src_hits),
        "families_top30": dict(fams),
        "ext_top30": dict(exts),
        "score_avg_top50": round(sum(scores) / len(scores), 3) if scores else 0,
        "score_p50": round(sorted(scores)[len(scores)//2], 3) if scores else 0,
        "ai_faiss_used": bool(resp.meta.get("ai_embedding_enabled") and (
            resp.meta.get("faiss_dino_count") or resp.meta.get("faiss_clip_count")
        )),
        "search_ms": round(ms, 1),
        "top": top,
        "semantic_terms": (resp.meta.get("semantic_query_terms") or [])[:12],
    }


def inventory(db_path: Path) -> dict:
    con = sqlite3.connect(str(db_path))
    n_files = con.execute(
        "SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND status NOT IN ('missing','excluded_internal')"
    ).fetchone()[0]
    n_idx = con.execute("SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND status='indexed'").fetchone()[0]
    n_ocr = con.execute(
        "SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND TRIM(COALESCE(ocr_text,''))!=''"
    ).fetchone()[0]
    fam = con.execute(
        """
        SELECT json_extract(fe.texture_map,'$.pattern_family') f, COUNT(*) n
        FROM files f JOIN features fe ON fe.file_id=f.id
        WHERE f.source_id IN (8,9)
        GROUP BY 1 ORDER BY n DESC LIMIT 15
        """
    ).fetchall()
    name_gül = con.execute(
        "SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND (filename LIKE '%gül%' OR filename LIKE '%gul%' OR filename LIKE '%rose%')"
    ).fetchone()[0]
    name_pap = con.execute(
        "SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND (filename LIKE '%papatya%' OR filename LIKE '%daisy%')"
    ).fetchone()[0]
    name_lv = con.execute(
        "SELECT COUNT(*) FROM files WHERE source_id IN (8,9) AND (lower(filename) LIKE '%lv%' OR lower(filename) LIKE '%vuitton%' OR lower(filename) LIKE '%louis%')"
    ).fetchone()[0]
    floral = con.execute(
        """
        SELECT COUNT(*) FROM files f JOIN features fe ON fe.file_id=f.id
        WHERE f.source_id IN (8,9) AND json_extract(fe.texture_map,'$.pattern_family')='floral'
        """
    ).fetchone()[0]
    mono = con.execute(
        """
        SELECT COUNT(*) FROM files f JOIN features fe ON fe.file_id=f.id
        WHERE f.source_id IN (8,9) AND json_extract(fe.texture_map,'$.pattern_family')='monogram_logo'
        """
    ).fetchone()[0]
    con.close()
    return {
        "files_8_9": n_files,
        "status_indexed": n_idx,
        "ocr_nonempty": n_ocr,
        "floral_labeled": floral,
        "monogram_labeled": mono,
        "filename_contains_gul_rose": name_gül,
        "filename_contains_papatya_daisy": name_pap,
        "filename_contains_lv": name_lv,
        "families": [{"family": r[0], "n": r[1]} for r in fam],
    }


def main() -> None:
    settings = AppSettings.load()
    inv = inventory(Path(settings.db_path))
    print("inventory", json.dumps(inv, ensure_ascii=False)[:800], flush=True)
    # Match production UI text search: SearchEngine(..., load_ai=False)
    engine = SearchEngine(settings, load_ai=False)
    thr = float(settings.similarity_threshold or 0.6)
    all_q = list(dict.fromkeys(QUERIES_BASIC + QUERIES_FLOWER + QUERIES_DOT + QUERIES_BRAND + QUERIES_NL))
    rows = []
    for q in all_q:
        print("query", q, flush=True)
        rows.append(run_one(engine, q, thr))
        print("  above", rows[-1]["above"], "true", rows[-1]["true_top30"], "false", rows[-1]["false_top30"], flush=True)

    clip_probe = None
    try:
        eng2 = SearchEngine(settings, load_ai=True)
        clip_probe = {
            "clip_model": bool(getattr(eng2.extractor, "_clip_model", None)),
            "faiss_available": bool(getattr(eng2.faiss, "available", False)),
            "search_text_visual": bool(settings.search_text_visual),
        }
        p = run_one(eng2, "gül", thr)
        clip_probe["gul_clip_hits"] = p["sources"].get("clip", 0)
        clip_probe["gul_above"] = p["above"]
        p2 = run_one(eng2, "Louis Vuitton", thr)
        clip_probe["lv_clip_hits"] = p2["sources"].get("clip", 0)
        clip_probe["lv_top"] = p2["top"][:6]
    except Exception as exc:
        clip_probe = {"error": str(exc)}

    report = {
        "inventory": inv,
        "settings": {
            "search_text_visual": settings.search_text_visual,
            "semantic_text_search_enabled": settings.semantic_text_search_enabled,
            "ai_embedding_enabled": settings.ai_embedding_enabled,
            "ai_search_min_vectors": settings.ai_search_min_vectors,
            "threshold": thr,
            "ui_load_ai_for_text": False,
        },
        "queries": rows,
        "clip_probe": clip_probe,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
