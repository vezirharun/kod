"""Shadow eval: UVI Search Enforcement v1. No reindex / FAISS / DB writes.

Auto-P is NOT human ground truth — it uses UVI object-evidence tiers plus
path heuristics for screen-test regressions (araba/kuş/balık/kot).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.textile_terms import normalize_turkish

QUERIES = [
    "araba",
    "kuş",
    "balık",
    "kot",
    "karga",
    "leylek",
    "gül",
    "leopar",
    "yılan",
    "leopar çiçek",
    "küçük kargalı desen",
]

MATCH_EV = {"object_exact", "object_subtype", "object_family"}
_AUTO_P_NOTE = "auto-P uses UVI object_evidence / path heuristics, not human labels"


def _tok(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", normalize_turkish(text or "")) if len(t) >= 3}


def _heuristic_fp(query: str, path: str, filename: str, evidence: str) -> bool:
    blob = _tok(f"{path} {filename}")
    q = normalize_turkish(query)
    if q == "araba":
        return evidence not in MATCH_EV and bool(blob & {"dantel", "floral", "cicek", "dokusu"})
    if q in ("kus", "kuş") or query == "kuş":
        return evidence not in MATCH_EV and bool(blob & {"denim", "kot", "jeans"})
    if q in ("balik", "balık") or query == "balık":
        return evidence not in MATCH_EV and bool(blob & {"floral", "cicek", "dantel"})
    return False


def _heuristic_tp(query: str, path: str, filename: str, evidence: str) -> bool:
    blob = _tok(f"{path} {filename}")
    q = normalize_turkish(query)
    if q == "kot":
        return bool(blob & {"kot", "denim", "jeans", "jean"})
    if evidence in MATCH_EV:
        return True
    return False


def main() -> None:
    from core.universal_visual_intel import CAPABILITIES, ONTOLOGY, parse_universal_query

    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "production_index_touched": False,
        "production_db_touched": False,
        "ground_truth": "auto-P / path heuristic — NOT human labels",
        "capabilities": CAPABILITIES,
        "ontology_size": len(ONTOLOGY),
        "queries": {},
        "error": None,
        "clip_active": False,
    }
    try:
        from core.search_engine import SearchEngine
        from core.search_models import SearchQuery
        from core.settings import AppSettings

        settings = AppSettings.load()
        settings.semantic_pattern_intel_enabled = True
        settings.pattern_intelligence_v2_enabled = True
        settings.universal_visual_intel_enabled = True
        engine = SearchEngine(settings, load_ai=True)
        engine.ensure_ai_loaded()
        clip_ok = bool(
            engine.extractor.ai_available
            and getattr(engine.extractor, "_clip_model", None) is not None
            and engine.faiss.available
            and engine.faiss.clip_count > 0
        )
        out["clip_active"] = clip_ok
        for q in QUERIES:
            uq = parse_universal_query(q)
            resp = engine.execute_search(SearchQuery(mode="text", text=q, threshold=0.40))
            rows = list(resp.results or [])[:20]
            uvi = (resp.meta or {}).get("universal_visual_intel") or {}
            labels = []
            fp = fn_proxy = 0
            for r in rows:
                ev = str((r.debug or {}).get("object_evidence") or "")
                labels.append(
                    {
                        "file": (r.filename or "")[:80],
                        "path": (getattr(r, "path", "") or "")[:120],
                        "tier": (r.debug or {}).get("uvi_tier"),
                        "evidence": ev,
                        "reason": (r.debug or {}).get("uvi_reason"),
                        "explain": (r.debug or {}).get("uvi_explain"),
                        "score": round(float(r.score or 0), 4),
                    }
                )
                if _heuristic_fp(q, getattr(r, "path", "") or "", r.filename or "", ev):
                    fp += 1
                if _heuristic_tp(q, getattr(r, "path", "") or "", r.filename or "", ev):
                    fn_proxy += 1

            def prec(k: int, pred) -> dict:
                chunk = labels[:k]
                if not chunk:
                    return {"p": 0.0, "n": 0, "note": _AUTO_P_NOTE}
                hits = sum(1 for x in chunk if pred(x))
                return {"p": round(hits / len(chunk), 3), "n": len(chunk), "hits": hits, "note": _AUTO_P_NOTE}

            if q == "kot":
                match_pred = lambda x: _heuristic_tp(
                    "kot", x.get("path") or "", x.get("file") or "", x.get("evidence") or ""
                )
            else:
                match_pred = lambda x: x.get("evidence") in MATCH_EV
            mismatch_pred = lambda x: x.get("evidence") == "object_mismatch"
            semantic_pred = lambda x: (
                False
                if q == "kot"
                else (
                    x.get("evidence") in ("object_semantic", "object_unknown")
                    or x.get("tier") == "SEMANTIC_SIMILAR"
                )
            )
            rec = {
                "query": q,
                "node": uq.node_id,
                "status": uq.status,
                "result_count": len(rows),
                "p_at_10": prec(10, match_pred),
                "p_at_20": prec(20, match_pred),
                "object_match_rate": prec(10, match_pred)["p"] if labels else 0.0,
                "object_mismatch_rate": prec(10, mismatch_pred)["p"] if labels else 0.0,
                "semantic_only_rate": prec(10, semantic_pred)["p"] if labels else 0.0,
                "false_positive_count": fp,
                "false_negative_count": max(0, min(10, len(rows)) - fn_proxy) if q == "kot" else None,
                "dropped_unspecific": uvi.get("dropped_unspecific"),
                "dropped_mismatch": uvi.get("dropped_mismatch"),
                "tiers": uvi.get("tiers"),
                "evidence": uvi.get("evidence"),
                "top": labels[:8],
            }
            out["queries"][q] = rec
            print(
                f"{q:24} n={rec['result_count']:3} p@10={rec['p_at_10']['p']:.3f} "
                f"fp={fp} dropped_mis={uvi.get('dropped_mismatch')}"
            )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        print("shadow eval unavailable:", out["error"])

    dest = ROOT / "data" / "reports" / "uvi_search_enforcement_v1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", dest)
    print("production_index_touched=False production_db_touched=False")


if __name__ == "__main__":
    main()
