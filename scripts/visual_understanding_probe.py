"""Measure query-side visual understanding. No index rebuild, no ranking change.

Parses each matrix query through NLQ + PatternIntent + v2 + UVI, then
(optionally) corpus filename/OCR counts. Visual CLIP ranking is not run.

  python scripts/visual_understanding_probe.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Expected slots: what a human-like parser should emit. Missing = capability gap.
MATRIX = [
    {
        "id": "object_leopard",
        "level": 1,
        "query": "leopar",
        "expect": {"object": "leopard"},
        "note": "Nesne — leopard desenleri",
    },
    {
        "id": "species_tiger",
        "level": 1,
        "query": "kaplan",
        "expect": {"object": "tiger"},
        "note": "Tür — leopard'tan ayırım",
    },
    {
        "id": "orientation_leopard",
        "level": 3,
        "query": "sağa bakan leopar",
        "expect": {"object": "leopard", "orientation": "right"},
        "note": "Yön",
    },
    {
        "id": "count_orientation_leopard",
        "level": 3,
        "query": "iki sağa bakan leopar",
        "expect": {"object": "leopard", "count": "2", "orientation": "right"},
        "note": "Sayı + yön",
    },
    {
        "id": "object_car",
        "level": 1,
        "query": "araba",
        "expect": {"object": "car", "category": "vehicle"},
        "note": "Nesne — gerçek araba vs baskı",
    },
    {
        "id": "brand_togg",
        "level": 5,
        "query": "Togg",
        "expect": {"brand": "togg", "category": "vehicle", "object": "car"},
        "note": "Marka",
    },
    {
        "id": "type_suv",
        "level": 5,
        "query": "SUV",
        "expect": {"type": "suv", "category": "vehicle"},
        "note": "Tip",
    },
    {
        "id": "brand_type_togg_suv",
        "level": 5,
        "query": "Togg SUV",
        "expect": {"brand": "togg", "type": "suv", "object": "car"},
        "note": "Marka + tip",
    },
    {
        "id": "attrs_blue_suv",
        "level": 2,
        "query": "mavi SUV",
        "expect": {"object": "car", "type": "suv", "color": "blue"},
        "note": "Nesne + tip + renk",
    },
    {
        "id": "hierarchy_blue_togg_suv",
        "level": 5,
        "query": "mavi Togg SUV",
        "expect": {"brand": "togg", "category": "vehicle", "type": "suv", "color": "blue"},
        "note": "Hiyerarşi",
    },
    {
        "id": "object_rose",
        "level": 1,
        "query": "gül",
        "expect": {"object": "rose"},
        "note": "Tür — generic çiçekten ayırım",
    },
    {
        "id": "color_rose",
        "level": 2,
        "query": "kırmızı gül",
        "expect": {"object": "rose", "color": "red"},
        "note": "Renk + nesne",
    },
    {
        "id": "relation_red_rose_textile",
        "level": 3,
        "query": "kırmızı gül desenli kumaş",
        "expect": {
            "object": "rose",
            "color": "red",
            "representation": "textile",
        },
        "note": "İlişki — motif + kumaş",
    },
    {
        "id": "composite_rose_leaf",
        "level": 3,
        "query": "kırmızı gül + yaprak",
        "expect": {"object": "rose", "object2": "leaf", "color": "red"},
        "note": "Composite",
    },
    {
        "id": "species_crow",
        "level": 1,
        "query": "karga",
        "expect": {"object": "crow", "family": "bird"},
        "note": "Tür — diğer kuşlardan ayırım",
    },
    {
        "id": "ocr_gucci",
        "level": 4,
        "query": "gucci",
        "expect": {"brand": "gucci", "ocr": True},
        "note": "OCR / marka yazısı",
    },
    {
        "id": "texture_denim",
        "level": 1,
        "query": "denim",
        "expect": {"texture": "denim"},
        "note": "Doku",
    },
    {
        "id": "texture_kot",
        "level": 1,
        "query": "kot",
        "expect": {"texture": "denim"},
        "note": "Doku alias",
    },
    {
        "id": "color_car",
        "level": 2,
        "query": "mavi araba",
        "expect": {"object": "car", "color": "blue"},
        "note": "Nesne + renk",
    },
    {
        "id": "identity_forbidden",
        "level": 0,
        "query": "bu Mehmet'e benziyor",
        "expect": {"identity": "forbidden"},
        "note": "Kimlik tanıma yapılmayacak",
    },
]


def _status_conf(status: str) -> int:
    s = (status or "").lower()
    if s == "supported":
        return 90
    if s == "experimental":
        return 61
    if s == "unsupported":
        return 0
    return 40


def _got_slots(nlq, intent, v2, uq) -> dict[str, str]:
    got: dict[str, str] = {}
    if uq.node_id:
        got["object"] = uq.node_id
    elif intent.motif:
        got["object"] = intent.motif
    elif nlq.motif:
        got["object"] = nlq.motif
    if uq.family:
        got["family"] = uq.family
    if uq.level:
        got["level"] = uq.level
    path = list(uq.path or [])
    if "vehicle" in path:
        got["category"] = "vehicle"
    if "car" in path and uq.node_id != "car":
        got["object"] = got.get("object") or "car"
    if uq.node_id in ("togg", "bmw", "mercedes", "audi", "gucci") or (nlq.brand or intent.brand):
        got["brand"] = uq.node_id if uq.level == "brand" else (nlq.brand or intent.brand or uq.node_id)
    color = intent.color or v2.color or nlq.color
    if color:
        got["color"] = color
    if v2.representation:
        got["representation"] = v2.representation
    if nlq.texture:
        got["texture"] = nlq.texture
    if v2.objects:
        got["v2_objects"] = ",".join(v2.objects)
    if v2.motifs:
        got["v2_motifs"] = ",".join(v2.motifs)
    if v2.composition:
        got["composition"] = v2.composition
    residual = (nlq.residual or "").strip()
    if residual:
        got["residual"] = residual
    return got


def _slot_verdict(key: str, expected: str, got: dict) -> dict:
    if key == "identity" and expected == "forbidden":
        return {
            "slot": key,
            "expected": expected,
            "parsed": "not_implemented",
            "ok": True,
            "source": "policy",
            "confidence": 100,
            "note": "face/identity yok — doğru sınır",
        }
    if key == "ocr":
        return {
            "slot": key,
            "expected": "visual_text",
            "parsed": "unmeasured",
            "ok": None,
            "source": "ocr_index",
            "confidence": None,
            "note": "sorgu parse değil; indeks OCR alanı ayrıca bakılır",
        }
    # Residual = unparsed leftover. Do not treat leftover tokens as understood slots.
    structured = {k: v for k, v in got.items() if k != "residual"}
    raw = (got.get(key) or "").lower()
    exp = str(expected).lower()
    aliases = {
        "red": ("red", "red_pink", "kirmizi"),
        "blue": ("blue", "mavi"),
        "textile": ("textile", "textile_pattern", "kumas"),
        "suv": ("suv",),
        "right": ("right", "sag", "sağa"),
        "2": ("2", "iki", "two"),
        "leaf": ("leaf", "yaprak"),
        "denim": ("denim", "kot"),
        "gucci": ("gucci",),
        "togg": ("togg",),
        "vehicle": ("vehicle", "car"),
        "car": ("car", "vehicle", "togg"),
        "rose": ("rose",),
        "leopard": ("leopard",),
        "tiger": ("tiger",),
        "crow": ("crow",),
        "bird": ("bird",),
    }
    ok = False
    if key in ("orientation", "count", "type"):
        ok = False
        raw = ""
        note_miss = {
            "orientation": "orientation slot yok; 'sağa/bakan' residual kaldı",
            "count": "count slot yok; 'iki' residual kaldı",
            "type": "SUV ontolojide yok; residual olarak kaldı",
        }[key]
        return {
            "slot": key,
            "expected": expected,
            "parsed": "—",
            "ok": False,
            "source": "missing",
            "confidence": 0,
            "note": note_miss,
        }
    if key == "object2":
        blob = " ".join(str(v) for v in structured.values()).lower()
        ok = "leaf" in blob
        raw = "leaf" if ok else ""
    else:
        toks = aliases.get(exp, (exp,))
        blob = " ".join(
            [
                raw,
                got.get("object", ""),
                got.get("brand", ""),
                got.get("family", ""),
                got.get("category", ""),
                got.get("v2_objects", ""),
                got.get("v2_motifs", ""),
            ]
        ).lower()
        ok = any(t in blob for t in toks)
    source = "ontology" if ok else "missing"
    conf = 90 if ok else 0
    if ok and key in ("brand",) and expected.lower() == "togg":
        conf = 61
        source = "experimental"
    return {
        "slot": key,
        "expected": expected,
        "parsed": raw or ("—" if not ok else str(expected)),
        "ok": ok,
        "source": source,
        "confidence": conf,
        "note": "" if ok else "parser slot yok",
    }


def probe_query(row: dict) -> dict:
    from core.natural_language_query import parse_natural_query
    from core.pattern_intelligence_v2 import parse_pattern_query_v2
    from core.semantic_pattern_intel import parse_pattern_intent
    from core.universal_visual_intel import ONTOLOGY, parse_universal_query

    q = row["query"]
    nlq = parse_natural_query(q)
    intent = parse_pattern_intent(q)
    v2 = parse_pattern_query_v2(q)
    uq = parse_universal_query(q)
    node = ONTOLOGY.get(uq.node_id) or {}
    got = _got_slots(nlq, intent, v2, uq)
    slots = [_slot_verdict(k, v, got) for k, v in row["expect"].items()]
    parsed_n = sum(1 for s in slots if s["ok"] is True)
    miss_n = sum(1 for s in slots if s["ok"] is False)
    unknown_n = sum(1 for s in slots if s["ok"] is None)
    if miss_n == 0 and parsed_n and not unknown_n:
        verdict = "PASS"
    elif parsed_n:
        verdict = "PARTIAL"
    elif unknown_n and not miss_n:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    if row["id"] == "identity_forbidden":
        verdict = "PASS"
    understood = [s["expected"] for s in slots if s["ok"] is True]
    missing = [s["slot"] for s in slots if s["ok"] is False]
    return {
        "id": row["id"],
        "level": row["level"],
        "query": q,
        "note": row["note"],
        "verdict": verdict,
        "understood": understood,
        "missing": missing,
        "slots": slots,
        "got": got,
        "uvi": {
            "node": uq.node_id,
            "path": uq.path,
            "level": uq.level,
            "status": uq.status or node.get("status", ""),
            "leaf": uq.leaf,
            "family": uq.family,
            "required": uq.required,
            "rivals": uq.rivals,
        },
        "v1": {
            "family": intent.family,
            "motif": intent.motif,
            "color": intent.color,
            "brand": intent.brand,
            "specificity": intent.specificity,
        },
        "v2": v2.to_dict(),
        "nlq": {
            "brand": nlq.brand,
            "color": nlq.color,
            "motif": nlq.motif,
            "texture": nlq.texture,
            "residual": nlq.residual,
            "components": nlq.components,
        },
        "node_status": uq.status or node.get("status", ""),
        "node_conf": _status_conf(uq.status or node.get("status", "")),
    }


def corpus_stats(db_path: str) -> dict:
    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    cur = con.cursor()
    n_files = cur.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    n_ocr = cur.execute(
        "SELECT COUNT(*) FROM files WHERE TRIM(COALESCE(ocr_text,'')) != ''"
    ).fetchone()[0]
    terms = [
        "gucci",
        "versace",
        "togg",
        "suv",
        "leopar",
        "leopard",
        "kaplan",
        "tiger",
        "gül",
        "gul",
        "rose",
        "karga",
        "crow",
        "denim",
        "kot",
        "araba",
    ]
    hits = {}
    for t in terms:
        like = f"%{t}%"
        fn = cur.execute(
            "SELECT COUNT(*) FROM files WHERE lower(filename) LIKE lower(?) OR lower(path) LIKE lower(?)",
            (like, like),
        ).fetchone()[0]
        ocr = cur.execute(
            "SELECT COUNT(*) FROM files WHERE lower(COALESCE(ocr_text,'')) LIKE lower(?)",
            (like,),
        ).fetchone()[0]
        ocr_not_fn = cur.execute(
            """
            SELECT COUNT(*) FROM files
            WHERE lower(COALESCE(ocr_text,'')) LIKE lower(?)
              AND lower(filename) NOT LIKE lower(?)
              AND lower(path) NOT LIKE lower(?)
            """,
            (like, like, like),
        ).fetchone()[0]
        hits[t] = {"filename_or_path": fn, "ocr": ocr, "ocr_only": ocr_not_fn}
    samples = {}
    for t in ("gucci", "versace", "togg"):
        rows = cur.execute(
            """
            SELECT filename, substr(ocr_text,1,80)
            FROM files
            WHERE lower(COALESCE(ocr_text,'')) LIKE lower(?)
            LIMIT 5
            """,
            (f"%{t}%",),
        ).fetchall()
        samples[t] = [{"filename": r[0], "ocr": r[1]} for r in rows]
    con.close()
    return {
        "files": n_files,
        "ocr_nonempty": n_ocr,
        "ocr_coverage_pct": round(100.0 * n_ocr / max(n_files, 1), 2),
        "terms": hits,
        "ocr_samples": samples,
    }


def main() -> None:
    from core.settings import AppSettings
    from core.universal_visual_intel import CAPABILITIES

    cards = [probe_query(row) for row in MATRIX]
    settings = AppSettings.load()
    corpus = corpus_stats(settings.db_path)
    gucci = corpus["terms"].get("gucci") or {}
    for c in cards:
        if c["id"] != "ocr_gucci":
            continue
        ocr_only = int(gucci.get("ocr_only") or 0)
        ocr_n = int(gucci.get("ocr") or 0)
        for s in c["slots"]:
            if s["slot"] != "ocr":
                continue
            s["parsed"] = f"ocr={ocr_n} ocr_only={ocr_only}"
            s["ok"] = ocr_only > 0
            s["confidence"] = 50 if ocr_only else 0
            s["note"] = (
                f"indekste {ocr_only} dosya OCR'da Gucci, dosya adında yok; arama sıralaması bu oturumda çalıştırılmadı"
            )
        parsed_n = sum(1 for s in c["slots"] if s["ok"] is True)
        miss_n = sum(1 for s in c["slots"] if s["ok"] is False)
        c["understood"] = [s["expected"] for s in c["slots"] if s["ok"] is True]
        c["missing"] = [s["slot"] for s in c["slots"] if s["ok"] is False]
        c["verdict"] = "PARTIAL" if parsed_n else "FAIL"
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for c in cards:
        counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "production_index_touched": False,
        "ranking_run": False,
        "clip_map_note": "visual CLIP ranking not measured here",
        "counts": counts,
        "capabilities_relevant": {
            k: CAPABILITIES[k]
            for k in (
                "hierarchical_query_parse",
                "fine_grained_clip_rival_gate",
                "composite_and_search",
                "representation_textile_vs_photo",
                "visual_dna_query_card",
                "face_recognition",
                "dedicated_togg_classifier",
                "dedicated_bird_species_classifier",
                "clip_zero_shot_species",
                "clip_zero_shot_car_brand",
                "spatial_object_boxes",
            )
            if k in CAPABILITIES
        },
        "corpus": corpus,
        "cards": cards,
    }
    dest = ROOT / "data" / "reports" / "visual_understanding_cards.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))
    print(f"ocr {corpus['ocr_nonempty']}/{corpus['files']} ({corpus['ocr_coverage_pct']}%)")
    for c in cards:
        miss = ",".join(c["missing"]) or "—"
        print(f"{c['verdict']:8} L{c['level']}  {c['query']!r:32} missing={miss}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
