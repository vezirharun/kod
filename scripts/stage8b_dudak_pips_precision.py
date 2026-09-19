"""Stage 8B — Dudak / Pips / Kaplan precision diagnosis (READ-ONLY).

No code/behavior/learning changes. Real DB only.
"""
from __future__ import annotations

import json
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ROOT / "data" / "patterns.db"
MEMORY = ROOT / "data" / "search_memory.db"
OUT = ROOT / "data" / "reports"


def _conn(path: Path):
    c = sqlite3.connect(str(path))
    c.row_factory = sqlite3.Row
    return c


def find_concepts(memory: Path, needles: list[str]) -> list[dict[str, Any]]:
    c = _conn(memory)
    rows = c.execute(
        "SELECT id, canonical, aliases FROM concept_registry "
        "WHERE IFNULL(status,'active') NOT IN ('inactive','retired')"
    ).fetchall()
    c.close()
    out = []
    for r in rows:
        can = str(r["canonical"] or "")
        blob = can.lower()
        try:
            aliases = json.loads(r["aliases"] or "[]")
        except Exception:
            aliases = []
        blob += " " + " ".join(str(a).lower() for a in aliases)
        if any(n.lower() in blob for n in needles):
            out.append(
                {
                    "id": int(r["id"]),
                    "canonical": can,
                    "aliases": aliases,
                }
            )
    return out


def positive_file_ids(memory: Path, canonical: str) -> dict[str, Any]:
    c = _conn(memory)
    rows = c.execute(
        """
        SELECT ce.file_id, ce.source, ce.file_path
        FROM concept_examples ce
        JOIN concept_registry cr ON cr.id=ce.concept_id
        WHERE cr.canonical=? AND ce.role='positive' AND ce.file_id>0
        """,
        (canonical,),
    ).fetchall()
    c.close()
    ids = {int(r["file_id"]) for r in rows}
    by_src: dict[str, int] = {}
    for r in rows:
        s = str(r["source"] or "")
        by_src[s] = by_src.get(s, 0) + 1
    return {
        "canonical": canonical,
        "count": len(ids),
        "ids": sorted(ids),
        "by_source": by_src,
    }


def precision_at(ranked_ids: list[int], gt: set[int], k: int) -> dict[str, Any]:
    top = ranked_ids[:k]
    if not top:
        return {"k": k, "precision": None, "hits": 0, "n": 0, "status": "olculemedi"}
    hits = sum(1 for i in top if i in gt)
    return {
        "k": k,
        "precision": round(hits / len(top), 4),
        "hits": hits,
        "n": len(top),
        "status": "ok",
    }


def live_search(
    query: str,
    *,
    limit: int = 50,
    engine: Any | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        if engine is None:
            from core.search_engine import SearchEngine
            from core.settings import AppSettings

            settings = AppSettings()
            settings.db_path = str(PATTERNS)
            try:
                engine = SearchEngine(settings, load_ai=False)
            except TypeError:
                engine = SearchEngine(settings)
        results = engine.search_by_text(query, limit=limit)
        rows = []
        for r in results[:limit]:
            dbg = getattr(r, "debug", None) or {}
            if not isinstance(dbg, dict):
                dbg = {}
            fid = getattr(r, "file_id", None)
            try:
                fid = int(fid) if fid is not None else None
            except Exception:
                fid = None
            rows.append(
                {
                    "file_id": fid,
                    "filename": getattr(r, "filename", None),
                    "score": round(float(getattr(r, "score", 0) or 0), 4),
                    "pattern_family": getattr(r, "pattern_family", None),
                    "animal_print_type": getattr(r, "animal_print_type", None),
                    "learned_exact": bool(dbg.get("learned_concept_exact")),
                    "learned_canonical": dbg.get("learned_canonical"),
                    "learned": bool(dbg.get("learned_concept")),
                    "user_taught": bool(dbg.get("user_taught_positive")),
                    "search_reason": dbg.get("search_reason"),
                    "result_layer": dbg.get("result_layer"),
                    "clip_only": bool(dbg.get("clip_only")),
                    "text_mode": bool(dbg.get("text_mode")),
                    "query_attribute_intel": (
                        dbg.get("query_attribute_intel")
                        if isinstance(dbg.get("query_attribute_intel"), dict)
                        else None
                    ),
                    "object_pattern_gate": (
                        dbg.get("object_pattern_gate")
                        if isinstance(dbg.get("object_pattern_gate"), dict)
                        else None
                    ),
                    "color_evidence_score": (
                        dbg.get("color_evidence_score")
                        if isinstance(dbg.get("color_evidence_score"), dict)
                        else None
                    ),
                }
            )
        return {
            "status": "ok",
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "rows": rows,
            "ids": [r["file_id"] for r in rows if r.get("file_id")],
        }
    except Exception as exc:
        return {
            "status": "olculemedi",
            "reason": str(exc)[:300],
            "trace_tail": traceback.format_exc()[-500:],
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "rows": [],
            "ids": [],
        }


def other_taught_labels(memory: Path, file_id: int) -> list[str]:
    c = _conn(memory)
    rows = c.execute(
        """
        SELECT cr.canonical, ce.role, ce.source
        FROM concept_examples ce
        JOIN concept_registry cr ON cr.id=ce.concept_id
        WHERE ce.file_id=? AND ce.role='positive'
        """,
        (file_id,),
    ).fetchall()
    c.close()
    return [f"{r['canonical']}:{r['source']}" for r in rows]


def diagnose_false(
    row: dict[str, Any],
    gt: set[int],
    memory: Path,
    target: str,
) -> dict[str, Any]:
    fid = row.get("file_id")
    if fid is None or fid in gt:
        return {}
    labels = other_taught_labels(memory, int(fid))
    cause = "unknown_visual_or_text"
    if row.get("learned_exact") or row.get("learned"):
        cause = f"learned_concept:{row.get('learned_canonical') or '?'}"
    elif row.get("clip_only"):
        cause = "clip_semantic"
    elif row.get("text_mode") and not row.get("learned"):
        cause = "text_index"
    elif (row.get("object_pattern_gate") or {}).get("reason"):
        cause = f"object_gate:{(row.get('object_pattern_gate') or {}).get('reason')}"
    elif row.get("animal_print_type") and target.lower() in {"dudak", "lips", "pips"}:
        cause = f"animal_print_type:{row.get('animal_print_type')}"
    return {
        "file_id": fid,
        "filename": row.get("filename"),
        "score": row.get("score"),
        "learned_canonical": row.get("learned_canonical"),
        "search_reason": row.get("search_reason"),
        "pattern_family": row.get("pattern_family"),
        "other_taught": labels,
        "likely_cause": cause,
    }


def eval_query(
    query: str,
    gt: set[int],
    memory: Path,
    *,
    limit: int = 50,
    target: str = "",
    engine: Any | None = None,
) -> dict[str, Any]:
    live = live_search(query, limit=limit, engine=engine)
    if live.get("status") != "ok":
        return {
            "query": query,
            "gt_size": len(gt),
            "live": live,
            "p_at": {},
            "false_positives": [],
            "status": "olculemedi",
        }
    ids = [int(i) for i in live["ids"] if i]
    p_at = {
        f"p@{k}": precision_at(ids, gt, k) for k in (10, 20, 50)
    }
    fps = []
    for row in live["rows"][:50]:
        if row.get("file_id") not in gt:
            d = diagnose_false(row, gt, memory, target or query)
            if d:
                fps.append(d)
    # cause histogram
    hist: dict[str, int] = {}
    for f in fps[:50]:
        c = str(f.get("likely_cause") or "unknown")
        hist[c] = hist.get(c, 0) + 1
    return {
        "query": query,
        "gt_size": len(gt),
        "status": "ok",
        "elapsed_sec": live.get("elapsed_sec"),
        "p_at": p_at,
        "top10_ids": ids[:10],
        "top10_learned": [
            {
                "file_id": r.get("file_id"),
                "score": r.get("score"),
                "learned_canonical": r.get("learned_canonical"),
                "learned_exact": r.get("learned_exact"),
                "in_gt": r.get("file_id") in gt,
            }
            for r in live["rows"][:10]
        ],
        "false_positives_top50": fps[:30],
        "fp_cause_hist": hist,
        "gt_recall_at_50": round(
            len([i for i in ids[:50] if i in gt]) / max(len(gt), 1), 4
        )
        if gt
        else None,
    }


def main() -> int:
    # Discover concepts
    dudak_like = find_concepts(MEMORY, ["dudak", "lip", "lips"])
    pips_like = find_concepts(MEMORY, ["pips", "pip", "puantiye", "polka", "dot"])
    tiger_like = find_concepts(MEMORY, ["tiger", "kaplan"])

    # Resolve canonical GT sets
    def pick_can(cands: list[dict], preferred: list[str]) -> str | None:
        low = {c["canonical"].lower(): c["canonical"] for c in cands}
        for p in preferred:
            if p.lower() in low:
                return low[p.lower()]
        return cands[0]["canonical"] if cands else None

    dudak_can = pick_can(dudak_like, ["dudak", "lips", "Dudak"])
    # User "pips" — registry'de Pips yok; en yakin ogretilmis: puantiye (polka/dot).
    pips_can = pick_can(pips_like, ["pips", "Pips", "puantiye"])
    tiger_can = pick_can(tiger_like, ["Tiger", "kaplan"])

    gt_dudak = positive_file_ids(MEMORY, dudak_can) if dudak_can else {"count": 0, "ids": []}
    gt_pips = positive_file_ids(MEMORY, pips_can) if pips_can else {"count": 0, "ids": []}
    gt_tiger = positive_file_ids(MEMORY, tiger_can) if tiger_can else {"count": 0, "ids": []}

    report: dict[str, Any] = {
        "stage": "8B_precision_diagnosis",
        "ts": datetime.now(timezone.utc).isoformat(),
        "behavior_changed": False,
        "gt_notes": {
            "dudak_gt_ceiling": (
                "dudak positives=2 -> Precision@10 teorik tavan 0.20 "
                "(GT disindaki dogru dudaklar olculemez)."
            ),
            "pips_mapping": (
                "Registry'de 'Pips' canonical yok. GT olarak puantiye (+polka/dot alias) kullanildi. "
                "Sorgu 'pips' yine calistirilir; eslesme yoksa olculemedi/yanlis kavram raporlanir."
            ),
        },
        "concepts_found": {
            "dudak_like": dudak_like,
            "pips_like": pips_like,
            "tiger_like": [{"canonical": c["canonical"], "id": c["id"]} for c in tiger_like[:8]],
        },
        "ground_truth": {
            "dudak": {k: v for k, v in gt_dudak.items() if k != "ids"},
            "pips": {k: v for k, v in gt_pips.items() if k != "ids"},
            "tiger": {k: v for k, v in gt_tiger.items() if k != "ids"},
            "dudak_ids": gt_dudak.get("ids"),
            "pips_ids": gt_pips.get("ids"),
            "tiger_ids": gt_tiger.get("ids"),
        },
        "queries": {},
        "kaplan_regression": {},
        "critical_fix": "",
        "summary": {},
    }

    dudak_gt = set(gt_dudak.get("ids") or [])
    pips_gt = set(gt_pips.get("ids") or [])
    tiger_gt = set(gt_tiger.get("ids") or [])

    # Overlap dudak ∩ pips
    report["dudak_pips_overlap"] = {
        "n": len(dudak_gt & pips_gt),
        "ids": sorted(dudak_gt & pips_gt),
    }

    queries_plan = []
    if dudak_gt:
        queries_plan += [("dudak", dudak_gt, "dudak"), ("lips", dudak_gt, "dudak")]
    else:
        report["queries"]["dudak"] = {"status": "olculemedi", "reason": "GT yok"}
    if pips_gt:
        queries_plan += [("pips", pips_gt, "pips"), ("puantiye", pips_gt, "pips")]
    else:
        report["queries"]["pips"] = {
            "status": "olculemedi",
            "reason": "Pips/puantiye positive bulunamadi",
            "candidates": pips_like,
        }
    if tiger_gt:
        # Canli sure uzun — kaplan + bir bilesik + tiger; kucuk ayri
        queries_plan += [
            ("kaplan", tiger_gt, "tiger"),
            ("tiger", tiger_gt, "tiger"),
            ("siyah krem kaplan", tiger_gt, "tiger"),
        ]

    # Shared engine — avoid multi cold-starts (~7min each).
    engine = None
    try:
        from core.search_engine import SearchEngine
        from core.settings import AppSettings

        settings = AppSettings()
        settings.db_path = str(PATTERNS)
        try:
            engine = SearchEngine(settings, load_ai=False)
        except TypeError:
            engine = SearchEngine(settings)
        print("[stage8b] SearchEngine ready", flush=True)
    except Exception as exc:
        print(f"[stage8b] engine init failed: {exc}", flush=True)
        engine = None

    for q, gt, target in queries_plan:
        print(f"[stage8b] searching: {q!r} gt={len(gt)}", flush=True)
        report["queries"][q] = eval_query(
            q, gt, MEMORY, limit=50, target=target, engine=engine
        )

    # Kaplan regression note from this run if present
    kap = report["queries"].get("kaplan") or {}
    if kap.get("status") == "ok":
        top = kap.get("top10_learned") or []
        all_tiger = all(
            (t.get("learned_canonical") or "").lower() == "tiger" and t.get("learned_exact")
            for t in top
        ) if top else False
        report["kaplan_regression"] = {
            "status": "ok",
            "top10_all_learned_tiger": all_tiger,
            "p_at": kap.get("p_at"),
            "top10": top,
            "note": "Regression kaydi: onceki oturumda da top10 learned Tiger idi.",
        }
    else:
        report["kaplan_regression"] = {
            "status": kap.get("status") or "olculemedi",
            "note": "Bu kosuda kaplan aramasi yok/basarisiz; onceki oturum: top10 Tiger learned (446s).",
        }

    # Critical fix suggestion from measured FP causes
    cause_all: dict[str, int] = {}
    for q, block in report["queries"].items():
        if not isinstance(block, dict):
            continue
        for c, n in (block.get("fp_cause_hist") or {}).items():
            cause_all[c] = cause_all.get(c, 0) + int(n)
    top_cause = sorted(cause_all.items(), key=lambda x: -x[1])
    report["fp_cause_total"] = top_cause
    if not dudak_gt:
        report["critical_fix"] = (
            "Once GT: dudak positives sayisi cok dusuk/yok — precision guvenilir olmaz; "
            "teach_me ile dudak GT genisletmeden ranking duzeltmesi onerme."
        )
    elif not pips_gt:
        report["critical_fix"] = (
            "Pips icin ogretilmis positive yok/bulunamadi — dudak-pips karisimi "
            "ogrenme belleginden olculemedi; once Pips GT'yi dogrula (yeni motor yok)."
        )
    elif top_cause:
        c0 = top_cause[0][0]
        if c0.startswith("learned_concept"):
            report["critical_fix"] = (
                f"En buyuk FP kaynagi {c0}: yanlis learned exact/related eslesmesi. "
                "Tek duzeltme: dudak/pips/tiger query match + negative boundary; "
                "ogrenme verisini silme, alias/relation daralt."
            )
        elif c0.startswith("clip"):
            report["critical_fix"] = (
                "En buyuk FP CLIP semantic. Tek duzeltme: learned exact varken "
                "CLIP sibling boost'unu kisitla (yeni motor yok)."
            )
        else:
            report["critical_fix"] = (
                f"En buyuk FP kaynagi: {c0}. Once bu katmani hedefle; yeni AI motoru yazma."
            )
    else:
        report["critical_fix"] = "Olculen FP yok veya arama olculemedi."

    # Compact summary
    summary = {}
    for q, block in report["queries"].items():
        if isinstance(block, dict) and block.get("p_at"):
            summary[q] = {
                "gt": block.get("gt_size"),
                "p@10": (block.get("p_at") or {}).get("p@10"),
                "p@20": (block.get("p_at") or {}).get("p@20"),
                "p@50": (block.get("p_at") or {}).get("p@50"),
                "fp_hist": block.get("fp_cause_hist"),
                "elapsed": block.get("elapsed_sec"),
            }
        else:
            summary[q] = {"status": (block or {}).get("status"), "reason": (block or {}).get("reason")}
    report["summary"] = summary

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"stage8b_dudak_pips_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = OUT / "stage8b_dudak_pips_latest.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "out": str(latest),
        "gt": report["ground_truth"],
        "overlap": report["dudak_pips_overlap"],
        "summary": summary,
        "kaplan_regression": report["kaplan_regression"],
        "critical_fix": report["critical_fix"],
        "fp_cause_total": top_cause[:8],
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
