"""Stage 8 — Search intelligence E2E quality audit (READ-ONLY, lightweight).

Real patterns.db + search_memory.db. No search behavior changes.
Unmeasurable → status 'olculemedi' (never invent results).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PATTERNS_DB = ROOT / "data" / "patterns.db"
MEMORY_DB = ROOT / "data" / "search_memory.db"
OUT_DIR = ROOT / "data" / "reports"

QUERY_GROUPS: dict[str, list[str]] = {
    "tr_en": [
        "kaplan",
        "tiger",
        "leopar",
        "leopard",
        "yılan",
        "snake",
        "çiçek",
        "flower",
        "dudak",
        "lips",
    ],
    "typo": ["kaplaan", "tigr", "leoprad", "yilan", "flowr"],
    "compound": [
        "küçük kaplan",
        "yoğun kaplan",
        "siyah krem kaplan",
        "küçük yoğun siyah krem kaplan",
        "kırmızı çiçek",
    ],
    "sibling": ["tiger", "leopard", "snake", "snake skin", "yılan"],
    "color": [
        "siyah",
        "beyaz",
        "krem",
        "kırmızı",
        "mavi",
        "siyah krem",
        "black cream tiger",
    ],
    "object_pattern": ["kaplan", "çiçek", "tiger", "flower"],
    "category": ["Tiger", "Floral", "Animal Print"],
}

LIVE_SEARCH_QUERIES = [
    "kaplan",
    "siyah krem kaplan",
    "leopard",
    "dudak",
    "kırmızı çiçek",
]

# Focused concept rows for sibling/TR checks (avoid full O(concepts×queries) hang).
FOCUS_CONCEPTS = [
    ("Tiger", ["kaplan", "tiger"]),
    ("Leopard", ["leopar", "leopard"]),
    ("Snake", ["yilan", "yılan", "snake"]),
    ("Snake Skin", ["snake skin", "yilan derisi", "yılan derisi"]),
    ("Floral", ["cicek", "çiçek", "flower", "floral"]),
    ("Rose", ["gul", "gül", "rose"]),
    ("dudak", ["dudak", "lips", "lip"]),
    ("kalp", ["kalp", "heart"]),
]


def _mem_path(patterns: Path) -> Path:
    p = patterns.with_name("search_memory.db")
    return p if p.is_file() else MEMORY_DB


def probe_query_meaning(query: str) -> dict[str, Any]:
    from core.search_intelligence_chain import analyze_query_intelligence

    t0 = time.perf_counter()
    analysis = analyze_query_intelligence(query)
    return {
        "elapsed_sec": round(time.perf_counter() - t0, 4),
        "normalized": analysis.get("normalized"),
        "concept_core": analysis.get("concept_core"),
        "concept_relation_hint": analysis.get("concept_relation_hint"),
        "intent_type": analysis.get("intent_type"),
        "colors": list(analysis.get("colors") or []),
        "attributes": analysis.get("attributes") or {},
    }


def probe_learned_light(memory_db: Path, query: str) -> dict[str, Any]:
    """Focused relation_to_concept only — no full registry scan."""
    if not memory_db.is_file():
        return {"status": "olculemedi", "reason": "search_memory.db yok"}
    try:
        from core.concept_query_normalize import relation_to_concept

        hits = []
        for can, aliases in FOCUS_CONCEPTS:
            rel = relation_to_concept(query, can, aliases=aliases)
            if rel:
                hits.append({"canonical": can, "relation": rel})
        # Prefer exact/typo/translation/alias over related
        rank = {
            "exact": 0,
            "translation": 0,
            "alias": 1,
            "typo": 1,
            "child": 2,
            "related": 3,
            "parent": 4,
        }
        hits.sort(key=lambda h: rank.get(str(h["relation"]), 9))
        return {
            "status": "ok",
            "mode": "focus_relation_to_concept",
            "match_count": len(hits),
            "primary": hits[0] if hits else None,
            "matches": hits[:8],
            "note": "Tam match_taught_concepts taraması yapılmadı (performans); odaklı kavram seti.",
        }
    except Exception as exc:
        return {"status": "olculemedi", "reason": str(exc)[:200]}


def probe_learned_registry_sample(memory_db: Path, query: str) -> dict[str, Any]:
    """Optional: SQL-only alias/canonical equality (no CLIP, no full fuzzy hang)."""
    if not memory_db.is_file():
        return {"status": "olculemedi"}
    try:
        from core.textile_terms import normalize_turkish

        qn = normalize_turkish(query)
        c = sqlite3.connect(str(memory_db))
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT id, canonical, aliases, parent, concept_type FROM concept_registry "
            "WHERE IFNULL(status,'active') NOT IN ('inactive','retired')"
        ).fetchall()
        c.close()
        hits = []
        for r in rows:
            can = str(r["canonical"] or "")
            cn = normalize_turkish(can)
            if cn == qn or qn in cn.split() or cn in qn.split():
                hits.append(
                    {
                        "canonical": can,
                        "relation": "exact",
                        "via": "canonical_norm",
                    }
                )
                continue
            try:
                aliases = json.loads(r["aliases"] or "[]")
            except Exception:
                aliases = []
            for a in aliases or []:
                if normalize_turkish(str(a)) == qn:
                    hits.append(
                        {
                            "canonical": can,
                            "relation": "alias",
                            "via": "alias_norm",
                        }
                    )
                    break
        return {
            "status": "ok",
            "mode": "sql_alias_exact",
            "match_count": len(hits),
            "primary": hits[0] if hits else None,
            "matches": hits[:8],
        }
    except Exception as exc:
        return {"status": "olculemedi", "reason": str(exc)[:200]}


def probe_color_db(patterns: Path) -> dict[str, Any]:
    if not patterns.is_file():
        return {"status": "olculemedi", "reason": "patterns.db yok"}
    c = sqlite3.connect(str(patterns))
    try:
        rgb = c.execute(
            "SELECT COUNT(*) FROM features WHERE dominant_colors IS NOT NULL "
            "AND dominant_colors NOT IN ('','[]')"
        ).fetchone()[0]
        named = ratio = user = 0
        n = 0
        for (tm,) in c.execute(
            "SELECT texture_map FROM features WHERE dominant_colors IS NOT NULL "
            "AND dominant_colors NOT IN ('','[]') LIMIT 5000"
        ):
            n += 1
            try:
                d = json.loads(tm or "{}")
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            if str(d.get("color_source") or "") in {"user", "teach_me", "manual_user"}:
                user += 1
            ev = d.get("color_evidence") if isinstance(d.get("color_evidence"), dict) else {}
            if ev.get("detected_colors"):
                named += 1
            if ev.get("ratio_confidence") == "cluster_weight":
                ratio += 1
        return {
            "status": "ok",
            "files_with_rgb": int(rgb),
            "sample": n,
            "sample_named_evidence": named,
            "sample_real_ratios": ratio,
            "sample_user_color_lock": user,
        }
    finally:
        c.close()


def probe_learning_authority(memory_db: Path) -> dict[str, Any]:
    if not memory_db.is_file():
        return {"status": "olculemedi"}
    c = sqlite3.connect(str(memory_db))
    c.row_factory = sqlite3.Row
    try:
        by_source = {
            str(r["source"]): int(r["n"])
            for r in c.execute(
                "SELECT coalesce(source,'') AS source, COUNT(*) AS n "
                "FROM concept_examples WHERE role='positive' GROUP BY source"
            )
        }
        focus = {}
        for can in (
            "Tiger",
            "Leopard",
            "Snake Skin",
            "Floral",
            "Rose",
            "dudak",
            "kalp",
        ):
            row = c.execute(
                """
                SELECT COUNT(*) AS n FROM concept_examples ce
                JOIN concept_registry cr ON cr.id=ce.concept_id
                WHERE ce.role='positive' AND cr.canonical=?
                """,
                (can,),
            ).fetchone()
            focus[can] = int(row["n"] if row else 0)
        events = int(c.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0])
        return {
            "status": "ok",
            "positives_by_source": by_source,
            "focus_positives": focus,
            "learning_events": events,
        }
    except Exception as exc:
        return {"status": "olculemedi", "reason": str(exc)[:200]}
    finally:
        c.close()


def probe_category_pool(patterns: Path) -> dict[str, Any]:
    if not patterns.is_file():
        return {"status": "olculemedi"}
    c = sqlite3.connect(str(patterns))
    try:
        total = int(c.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        clauses = {
            "all_files": "SELECT COUNT(*) FROM files",
            "has_thumb_or_preview": (
                "SELECT COUNT(*) FROM files WHERE "
                "coalesce(thumbnail_path,'')!='' OR coalesce(feature_preview_path,'')!=''"
            ),
            "searchable_visual_flag": (
                "SELECT COUNT(*) FROM files WHERE coalesce(is_searchable_visual,0)=1"
            ),
        }
        counts: dict[str, Any] = {}
        for name, sql in clauses.items():
            try:
                counts[name] = int(c.execute(sql).fetchone()[0])
            except Exception as exc:
                counts[name] = f"olculemedi:{exc}"
        same_pool_note = (
            "Kategori filtresi vs normal arama aynı SQL clause kullandığı kodda "
            "assert edildi (önceki Stage); bu audit yalnızca havuz büyüklüklerini ölçer."
        )
        return {
            "status": "ok",
            "total_files": total,
            "pool_counts": counts,
            "note": same_pool_note,
        }
    finally:
        c.close()


def static_double_score_audit() -> dict[str, Any]:
    import inspect

    from core import color_evidence as ce
    from core import search_intelligence_chain as sic
    from core import visual_variant_intel as vv

    chain_src = inspect.getsource(sic.apply_search_intelligence_chain)
    color_src = inspect.getsource(ce.apply_color_evidence_scoring)
    var_src = inspect.getsource(vv.annotate_visual_variants)
    findings = []
    if "apply_query_attribute_intel" in chain_src and "apply_color_evidence_scoring" in chain_src:
        guarded = "query_attribute_intel" in color_src and "matches" in color_src
        findings.append(
            {
                "opportunity": "attribute_color + color_evidence aynı zincirde",
                "guard_present": guarded,
                "severity_class": "kozmetik" if guarded else "orta",
            }
        )
    if "score_attributes_against_dna" in var_src:
        findings.append(
            {
                "opportunity": "visual_variant_intel attribute skoru yeniden hesaplıyor (annotate)",
                "score_change": False,
                "severity_class": "kozmetik",
            }
        )
    return {
        "status": "ok",
        "unified": "unified" in chain_src,
        "findings": findings,
    }


def evaluate_sibling(learned: dict[str, Any], query: str) -> dict[str, Any]:
    if learned.get("status") != "ok":
        return {"verdict": "olculemedi", "detail": learned.get("reason")}
    primary = learned.get("primary") or {}
    can = str(primary.get("canonical") or "").strip().lower()
    rel = str(primary.get("relation") or "")
    q = query.lower()
    verdict = "PASS"
    note = ""
    if not primary:
        return {"verdict": "olculemedi", "note": "learned match yok", "primary": None}
    if q in {"tiger", "kaplan"} and "leopard" in can and rel in {
        "exact",
        "alias",
        "translation",
        "typo",
    }:
        verdict, note = "FAIL", "Tiger→Leopard exact"
    elif q in {"leopard", "leopar"} and can == "tiger" and rel in {
        "exact",
        "alias",
        "translation",
        "typo",
    }:
        verdict, note = "FAIL", "Leopard→Tiger exact"
    elif q == "snake skin" and can in {"snake", "yılan", "yilan"} and rel == "exact":
        verdict, note = "FAIL", "Snake Skin exact→Snake"
    elif q == "snake" and "snake skin" in can and rel in {"exact", "alias", "translation"}:
        verdict, note = "WARN", "snake exact/alias→Snake Skin"
    return {"verdict": verdict, "note": note, "primary": primary}


def evaluate_tr_en_pair(a: dict, b: dict, label: str) -> dict[str, Any]:
    ma = ((a.get("meaning") or {}).get("attributes") or {}).get("motif")
    mb = ((b.get("meaning") or {}).get("attributes") or {}).get("motif")
    la = ((a.get("learned") or {}).get("primary") or {}).get("canonical")
    lb = ((b.get("learned") or {}).get("primary") or {}).get("canonical")
    if ma and mb and ma == mb:
        return {"verdict": "PASS", "via": "motif", "label": label, "motif": ma}
    if la and lb and str(la).lower() == str(lb).lower():
        return {"verdict": "PASS", "via": "learned", "label": label, "canonical": la}
    # Special: dudak/lips may only resolve via learned
    if label == "lips" and (la or lb or ma or mb):
        if (la and "dudak" in str(la).lower()) or (lb and "dudak" in str(lb).lower()):
            return {"verdict": "PASS", "via": "learned_partial", "label": label}
    if not la and not lb and not ma and not mb:
        return {"verdict": "olculemedi", "label": label}
    return {
        "verdict": "WARN",
        "label": label,
        "motif_a": ma,
        "motif_b": mb,
        "learned_a": la,
        "learned_b": lb,
    }


def probe_learned_file_sample(memory_db: Path, patterns: Path, canonical: str, *, limit: int = 10) -> dict[str, Any]:
    """Top user-positive files for a concept + pattern_family from index (not full search)."""
    if not memory_db.is_file() or not patterns.is_file():
        return {"status": "olculemedi"}
    try:
        mc = sqlite3.connect(str(memory_db))
        mc.row_factory = sqlite3.Row
        rows = mc.execute(
            """
            SELECT ce.file_id, ce.source, ce.file_path
            FROM concept_examples ce
            JOIN concept_registry cr ON cr.id=ce.concept_id
            WHERE cr.canonical=? AND ce.role='positive' AND ce.file_id>0
            ORDER BY CASE ce.source WHEN 'user' THEN 0 WHEN 'candidate' THEN 1 ELSE 2 END, ce.id
            LIMIT ?
            """,
            (canonical, limit),
        ).fetchall()
        mc.close()
        if not rows:
            return {"status": "ok", "canonical": canonical, "files": [], "note": "positive yok"}
        ids = [int(r["file_id"]) for r in rows]
        pc = sqlite3.connect(str(patterns))
        pc.row_factory = sqlite3.Row
        qmarks = ",".join("?" * len(ids))
        fmap = {
            int(r["id"]): r
            for r in pc.execute(
                f"SELECT id, filename, pattern_family, path FROM files WHERE id IN ({qmarks})",
                ids,
            )
        }
        # texture family hints
        tmap = {}
        for r in pc.execute(
            f"SELECT file_id, texture_map FROM features WHERE file_id IN ({qmarks})",
            ids,
        ):
            try:
                tm = json.loads(r["texture_map"] or "{}")
            except Exception:
                tm = {}
            tmap[int(r["file_id"])] = tm if isinstance(tm, dict) else {}
        pc.close()
        files = []
        for r in rows:
            fid = int(r["file_id"])
            fr = fmap.get(fid)
            tm = tmap.get(fid) or {}
            ev = tm.get("color_evidence") if isinstance(tm.get("color_evidence"), dict) else {}
            files.append(
                {
                    "file_id": fid,
                    "source": r["source"],
                    "filename": (fr["filename"] if fr else None),
                    "pattern_family": (fr["pattern_family"] if fr else None),
                    "tm_family": tm.get("pattern_family"),
                    "animal_print_type": tm.get("animal_print_type"),
                    "detected_colors": ev.get("detected_colors"),
                    "color_source": tm.get("color_source"),
                }
            )
        return {"status": "ok", "canonical": canonical, "files": files}
    except Exception as exc:
        return {"status": "olculemedi", "reason": str(exc)[:200]}
    t0 = time.perf_counter()
    try:
        from core.search_engine import SearchEngine
        from core.settings import AppSettings

        settings = AppSettings()
        settings.db_path = str(patterns)
        # Prefer not loading heavy AI if constructor allows; fall back to default.
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
            qa = dbg.get("query_attribute_intel") if isinstance(dbg.get("query_attribute_intel"), dict) else {}
            og = dbg.get("object_pattern_gate") if isinstance(dbg.get("object_pattern_gate"), dict) else {}
            ces = dbg.get("color_evidence_score") if isinstance(dbg.get("color_evidence_score"), dict) else {}
            chain = dbg.get("search_intelligence_chain") if isinstance(dbg.get("search_intelligence_chain"), dict) else {}
            color_attr = "color" in (qa.get("matches") or {})
            color_ev = bool(ces.get("applied")) and abs(float(ces.get("delta") or 0)) > 1e-9
            rows.append(
                {
                    "file_id": getattr(r, "file_id", None),
                    "filename": getattr(r, "filename", None),
                    "score": round(float(getattr(r, "score", 0) or 0), 4),
                    "pattern_family": getattr(r, "pattern_family", None),
                    "animal_print_type": getattr(r, "animal_print_type", None),
                    "learned_exact": bool(dbg.get("learned_concept_exact")),
                    "learned_canonical": dbg.get("learned_canonical"),
                    "user_taught": bool(dbg.get("user_taught_positive")),
                    "attr_matches": qa.get("matches"),
                    "object_gate": {"reason": og.get("reason"), "delta": og.get("delta")},
                    "color_evidence_score": {
                        "delta": ces.get("delta"),
                        "hits": ces.get("hits"),
                        "skipped": ces.get("skipped"),
                    },
                    "search_reason": dbg.get("search_reason"),
                    "double_color_score_risk": bool(color_attr and color_ev),
                    "chain_unified": bool(chain.get("unified")),
                }
            )
        return {
            "status": "ok",
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "count": len(rows),
            "top": rows,
        }
    except Exception as exc:
        return {
            "status": "olculemedi",
            "reason": str(exc)[:300],
            "trace_tail": traceback.format_exc()[-400:],
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        }


def classify_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in report.get("sibling_checks") or []:
        v = item.get("verdict")
        if v == "FAIL":
            out.append({"severity": "kritik", "area": "sibling", "detail": item})
        elif v == "WARN":
            out.append({"severity": "orta", "area": "sibling", "detail": item})
    for item in report.get("tr_en_pairs") or []:
        if item.get("verdict") == "WARN":
            out.append({"severity": "orta", "area": "tr_en", "detail": item})
        elif item.get("verdict") == "olculemedi":
            out.append({"severity": "veri_eksikligi", "area": "tr_en", "detail": item})

    color = report.get("color_db") or {}
    if color.get("status") == "ok":
        sample = max(1, int(color.get("sample") or 1))
        named = int(color.get("sample_named_evidence") or 0)
        if named / sample < 0.25:
            out.append(
                {
                    "severity": "veri_eksikligi",
                    "area": "color_evidence_coverage",
                    "detail": {
                        "sample_named": named,
                        "sample": sample,
                        "files_with_rgb": color.get("files_with_rgb"),
                    },
                }
            )

    for f in (report.get("static_double_score") or {}).get("findings") or []:
        out.append(
            {
                "severity": f.get("severity_class") or "kozmetik",
                "area": "static_chain",
                "detail": f,
            }
        )

    for q, live in (report.get("live_search") or {}).items():
        if not isinstance(live, dict) or "status" not in live:
            continue
        if live.get("status") == "olculemedi":
            out.append(
                {
                    "severity": "veri_eksikligi",
                    "area": "live_search",
                    "detail": {"query": q, "reason": live.get("reason")},
                }
            )
            continue
        risky = [r for r in (live.get("top") or []) if r.get("double_color_score_risk")]
        if risky:
            out.append(
                {
                    "severity": "orta",
                    "area": "double_color_score",
                    "detail": {"query": q, "risky_rows": len(risky)},
                }
            )
        if q in {"kaplan", "tiger", "çiçek", "flower"}:
            fams = [str(r.get("pattern_family") or "") for r in (live.get("top") or [])]
            if fams and all(f in {"object", "photo", ""} for f in fams):
                out.append(
                    {
                        "severity": "orta",
                        "area": "object_pattern",
                        "detail": {"query": q, "fams": fams[:5]},
                    }
                )
    return out


def run_audit(*, patterns: Path, live_search: bool, live_limit: int) -> dict[str, Any]:
    memory = _mem_path(patterns)
    report: dict[str, Any] = {
        "stage": 8,
        "mode": "read_only_quality_audit",
        "ts": datetime.now(timezone.utc).isoformat(),
        "patterns_db": str(patterns),
        "memory_db": str(memory),
        "behavior_changed": False,
        "queries": {},
        "tr_en_pairs": [],
        "sibling_checks": [],
        "typo_checks": [],
        "compound_checks": [],
        "color_db": {},
        "learning_authority": {},
        "category_pool": {},
        "static_double_score": {},
        "live_search": {},
        "findings": [],
        "suggested_fixes": [],
    }

    uniq: list[str] = []
    seen: set[str] = set()
    for qs in QUERY_GROUPS.values():
        for q in qs:
            if q not in seen:
                seen.add(q)
                uniq.append(q)

    for q in uniq:
        meaning = probe_query_meaning(q)
        learned = probe_learned_light(memory, q)
        sql_learned = probe_learned_registry_sample(memory, q)
        report["queries"][q] = {
            "group_tags": [g for g, qs in QUERY_GROUPS.items() if q in qs],
            "meaning": meaning,
            "learned": learned,
            "learned_sql_alias": sql_learned,
        }

    pairs = [
        ("kaplan", "tiger", "tiger"),
        ("leopar", "leopard", "leopard"),
        ("yılan", "snake", "snake"),
        ("çiçek", "flower", "flower"),
        ("dudak", "lips", "lips"),
    ]
    for a, b, label in pairs:
        report["tr_en_pairs"].append(
            evaluate_tr_en_pair(
                report["queries"].get(a, {}),
                report["queries"].get(b, {}),
                label,
            )
        )

    for q in QUERY_GROUPS["sibling"]:
        report["sibling_checks"].append(
            {
                "query": q,
                **evaluate_sibling(report["queries"].get(q, {}).get("learned") or {}, q),
            }
        )

    for q in QUERY_GROUPS["typo"]:
        learned = report["queries"].get(q, {}).get("learned") or {}
        primary = learned.get("primary")
        report["typo_checks"].append(
            {
                "query": q,
                "verdict": "PASS"
                if primary and str(primary.get("relation")) in {"exact", "typo", "alias", "translation", "related"}
                else ("olculemedi" if not primary else "WARN"),
                "primary": primary,
                "meaning_motif": ((report["queries"].get(q, {}).get("meaning") or {}).get("attributes") or {}).get(
                    "motif"
                ),
            }
        )

    for q in QUERY_GROUPS["compound"]:
        m = report["queries"].get(q, {}).get("meaning") or {}
        attrs = m.get("attributes") or {}
        report["compound_checks"].append(
            {
                "query": q,
                "motif": attrs.get("motif"),
                "scale": attrs.get("scale"),
                "density": attrs.get("density"),
                "colors": attrs.get("colors"),
                "concept_core": m.get("concept_core"),
                "learned_primary": (report["queries"].get(q, {}).get("learned") or {}).get("primary"),
            }
        )

    report["color_db"] = probe_color_db(patterns)
    report["learning_authority"] = probe_learning_authority(memory)
    report["category_pool"] = probe_category_pool(patterns)
    report["static_double_score"] = static_double_score_audit()
    report["learned_file_samples"] = {
        can: probe_learned_file_sample(memory, patterns, can, limit=10)
        for can in ("Tiger", "Leopard", "Snake Skin", "Floral", "dudak")
    }

    if live_search:
        for q in LIVE_SEARCH_QUERIES:
            report["live_search"][q] = live_search_top10(patterns, q, limit=live_limit)
    else:
        report["live_search"] = {
            "_note": (
                "Tam SearchEngine top-10 bu oturumda calistirilmadi "
                "(--live-search). Cold-start + 49k tarama uzun suruyor; "
                "sonuc uydurulmadi -> olculemedi."
            )
        }

    report["findings"] = classify_findings(report)
    sev = {"kritik": 0, "orta": 0, "kozmetik": 0, "veri_eksikligi": 0}
    for f in report["findings"]:
        k = str(f.get("severity"))
        sev[k] = sev.get(k, 0) + 1
    report["finding_counts"] = sev

    # Only necessary fix suggestions from measured findings
    tips = []
    if any(f.get("severity") == "kritik" for f in report["findings"]):
        tips.append("Kritik sibling FAIL varsa yalnızca concept_query_normalize / learned match path'e bak.")
    if any(f.get("area") == "color_evidence_coverage" for f in report["findings"]):
        tips.append(
            "Renk: davranış değil coverage — mevcut backfill_color_evidence.py batch ile devam (yeni motor yok)."
        )
    if any(f.get("area") == "double_color_score" for f in report["findings"]):
        tips.append("Çift renk skoru canlıda görüldüyse color_evidence skip guard'ını sıkılaştır.")
    if not tips:
        tips.append("Kritik FAIL yoksa yeni katman ekleme; coverage/backfill ve izleme yeterli.")
    report["suggested_fixes"] = tips
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(PATTERNS_DB))
    ap.add_argument("--live-search", action="store_true")
    ap.add_argument("--live-limit", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    report = run_audit(
        patterns=Path(args.db),
        live_search=bool(args.live_search),
        live_limit=max(1, int(args.live_limit)),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / (
        f"stage8_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "out": str(out),
        "finding_counts": report.get("finding_counts"),
        "tr_en_pairs": report.get("tr_en_pairs"),
        "sibling_checks": report.get("sibling_checks"),
        "typo_checks": report.get("typo_checks"),
        "compound_checks": report.get("compound_checks"),
        "color_db": report.get("color_db"),
        "learning_authority": report.get("learning_authority"),
        "category_pool": report.get("category_pool"),
        "static_double_score": report.get("static_double_score"),
        "learned_file_samples": {
            k: {
                "status": v.get("status"),
                "n": len(v.get("files") or []),
                "families": list(
                    {
                        str(f.get("pattern_family") or f.get("tm_family") or "")
                        for f in (v.get("files") or [])
                    }
                ),
            }
            for k, v in (report.get("learned_file_samples") or {}).items()
        },
        "live_search_status": {
            q: {"status": v.get("status"), "elapsed": v.get("elapsed_sec"), "n": v.get("count")}
            if isinstance(v, dict) and "status" in v
            else v
            for q, v in (report.get("live_search") or {}).items()
        },
        "findings": report.get("findings"),
        "suggested_fixes": report.get("suggested_fixes"),
    }
    text = json.dumps(summary, ensure_ascii=True, indent=2)
    try:
        print(text)
    except UnicodeEncodeError:
        import sys

        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
