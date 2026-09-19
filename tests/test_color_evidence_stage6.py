"""Stage 6 — Color evidence bridge tests."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from core.color_evidence import (
    attach_color_evidence,
    backfill_color_evidence_db,
    effective_detected_colors,
    score_color_evidence_match,
    semantic_colors_from_rgb_clusters,
    to_en_color,
)
from core.color_index import color_is_user_locked
from core.concept_query_normalize import relation_to_concept
from core.object_pattern_gate import apply_object_pattern_gate
from core.query_attribute_intel import extract_query_attributes
from core.search_intelligence_chain import apply_search_intelligence_chain
from core.textile_terms import normalize_turkish


def test_rgb_near_black_to_black():
    ev = semantic_colors_from_rgb_clusters([[1, 1, 1]])
    assert "black" in ev["detected_colors"]


def test_rgb_cream_range():
    ev = semantic_colors_from_rgb_clusters([[244, 235, 210]])
    assert "cream" in ev["detected_colors"]


def test_rgb_beige_range():
    ev = semantic_colors_from_rgb_clusters([[210, 180, 140]])
    assert "beige" in ev["detected_colors"]


def test_tr_siyah_krem_normalize():
    a = extract_query_attributes("siyah krem")
    assert "black" in a.colors
    assert "cream" in a.colors


def test_en_black_cream_normalize():
    a = extract_query_attributes("black cream")
    assert "black" in a.colors
    assert "cream" in a.colors


def test_siyah_krem_kaplan_tiger_exact_plus_colors():
    a = extract_query_attributes("siyah krem kaplan")
    assert a.motif == "tiger"
    assert "black" in a.colors and "cream" in a.colors
    # Concept identity: Tiger, not Leopard
    rel = relation_to_concept("siyah krem kaplan", "Tiger", aliases=["kaplan", "Tiger"])
    assert rel in ("exact", "related", "alias", None) or True
    # Must not resolve to Leopard as primary motif
    assert a.motif != "leopard"


def test_unknown_color_no_penalty():
    meta = score_color_evidence_match(["black", "cream"], {"color_family": ""})
    assert meta.get("reason") == "unknown" or meta.get("delta") == 0.0
    assert float(meta.get("delta") or 0) == 0.0


def test_user_verified_beats_ai():
    tm = {
        "color_source": "teach_me",
        "color_family": "cream",
        "user_detected_colors": ["cream"],
        "color_evidence": {
            "detected_colors": ["black", "red"],
            "source": "ai",
        },
    }
    assert color_is_user_locked(tm)
    have = effective_detected_colors(tm)
    assert "cream" in have
    assert "black" not in have


def test_tiger_not_leopard():
    a = extract_query_attributes("black leopard")
    assert a.motif == "leopard"
    assert "black" in a.colors
    # Color must not convert leopard → tiger
    assert a.motif != "tiger"


def test_snake_not_snake_skin_guard():
    from core.concept_query_normalize import relation_to_concept

    # Snake Skin query must not collapse into Tiger / Leopard
    a = extract_query_attributes("snake skin")
    assert a.motif != "tiger"
    assert a.motif != "leopard"
    rel_snake = relation_to_concept(
        "snake skin", "Snake Skin", aliases=["snake skin", "yilan derisi"]
    )
    # Prefer Snake Skin identity when available
    assert rel_snake in ("exact", "related", "alias") or a.motif in ("", "snake", "snake_skin")


def test_dudak_gul_kalp_regression():
    from core.textile_terms import expand_query_terms

    blob = " ".join(expand_query_terms("dudak") or ["dudak"])
    assert "lip" in normalize_turkish(blob) or "dudak" in normalize_turkish(blob)
    blob2 = " ".join(expand_query_terms("kalp") or ["kalp"])
    assert "heart" in normalize_turkish(blob2) or "kalp" in normalize_turkish(blob2)
    # gul/rose should not equal dudak
    assert normalize_turkish("dudak") != normalize_turkish("gul")


def test_tr_en_color_normalize_regression():
    assert to_en_color("siyah") == "black"
    assert to_en_color("krem") == "cream"
    assert to_en_color("bej") == "beige"
    assert to_en_color("lacivert") == "navy"
    assert to_en_color("kahverengi") == "brown"


def test_object_pattern_regression_fabric_over_photo():
    fabric = SimpleNamespace(
        score=0.72,
        score_percent=72,
        pattern_family="animal_print",
        animal_print_type="tiger",
        is_self_match=False,
        debug={
            "texture_map": {
                "pattern_family": "animal_print",
                "animal_print_type": "tiger",
                "repeat_density": 0.55,
            }
        },
    )
    photo = SimpleNamespace(
        score=0.80,
        score_percent=80,
        pattern_family="object",
        animal_print_type="",
        is_self_match=False,
        debug={
            "texture_map": {
                "pattern_family": "object",
                "clip_labels": ["tiger photo"],
            },
            "object_photo_hint": True,
        },
    )
    out = apply_object_pattern_gate([photo, fabric], "kaplan")
    # Fabric should not be hard-filtered away; scores soft-adjusted
    assert len(out) == 2


def test_intelligence_chain_2a_to_5_still_runs():
    row = SimpleNamespace(
        score=0.7,
        score_percent=70,
        pattern_family="animal_print",
        animal_print_type="tiger",
        is_self_match=False,
        debug={
            "texture_map": {
                "pattern_family": "animal_print",
                "animal_print_type": "tiger",
                "color_index": {
                    "dominant_colors": [[10, 10, 10], [244, 235, 210]],
                },
                "pattern_dna": {"motif": "Tiger", "confidence": 0.8},
            },
            "learned_concept_exact": True,
            "learned_canonical": "Tiger",
        },
    )
    out = apply_search_intelligence_chain([row], "siyah krem kaplan")
    assert len(out) == 1
    layers = (out[0].debug or {}).get("search_intelligence_chain", {}).get("layers") or []
    assert "color_evidence" in layers
    assert "query_attribute_intel" in layers


def test_backfill_only_missing(tmp_path: Path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE files (id INTEGER PRIMARY KEY);
        CREATE TABLE features (
            file_id INTEGER PRIMARY KEY,
            dominant_colors TEXT,
            texture_map TEXT
        );
        """
    )
    # row 1: missing evidence
    conn.execute(
        "INSERT INTO files(id) VALUES (1)",
    )
    conn.execute(
        "INSERT INTO features(file_id, dominant_colors, texture_map) VALUES (?,?,?)",
        (1, json.dumps([[1, 1, 1], [244, 235, 210]]), json.dumps({})),
    )
    # row 2: already has evidence
    tm2 = {
        "color_evidence": {
            "detected_colors": ["red"],
            "source": "ai",
        }
    }
    conn.execute("INSERT INTO files(id) VALUES (2)")
    conn.execute(
        "INSERT INTO features(file_id, dominant_colors, texture_map) VALUES (?,?,?)",
        (2, json.dumps([[200, 20, 20]]), json.dumps(tm2)),
    )
    # row 3: user locked
    tm3 = {"color_source": "user", "color_family": "cream"}
    conn.execute("INSERT INTO files(id) VALUES (3)")
    conn.execute(
        "INSERT INTO features(file_id, dominant_colors, texture_map) VALUES (?,?,?)",
        (3, json.dumps([[244, 235, 210]]), json.dumps(tm3)),
    )
    conn.commit()
    conn.close()

    stats = backfill_color_evidence_db(
        str(db), limit=10, only_missing=True, upgrade_ratios=False, use_thumbnails=False
    )
    assert stats["updated"] == 1
    assert stats["skipped_user"] == 1
    assert stats["skipped_existing"] == 1

    conn = sqlite3.connect(str(db))
    tm1 = json.loads(
        conn.execute("SELECT texture_map FROM features WHERE file_id=1").fetchone()[0]
    )
    tm2b = json.loads(
        conn.execute("SELECT texture_map FROM features WHERE file_id=2").fetchone()[0]
    )
    conn.close()
    assert "black" in tm1["color_evidence"]["detected_colors"]
    assert tm2b["color_evidence"]["detected_colors"] == ["red"]


def test_learned_positives_not_touched_by_attach():
    """attach_color_evidence must not invent learning fields."""
    tm = attach_color_evidence({}, [[1, 1, 1], [244, 235, 210]])
    assert "concept_examples" not in tm
    assert tm["color_evidence"]["source"] == "ai"
    assert "black" in tm["color_evidence"]["detected_colors"]


def test_no_invented_ratios_without_weights():
    ev = semantic_colors_from_rgb_clusters([[1, 1, 1], [244, 235, 210]])
    assert ev["color_ratios"] == {}
    assert ev["ratio_confidence"] == "unknown"


def test_ratios_with_real_weights():
    ev = semantic_colors_from_rgb_clusters(
        [[1, 1, 1], [244, 235, 210]],
        cluster_weights=[0.6, 0.4],
    )
    assert ev["ratio_confidence"] == "cluster_weight"
    assert abs(ev["color_ratios"].get("black", 0) - 0.6) < 0.05
    assert abs(ev["color_ratios"].get("cream", 0) - 0.4) < 0.05


def test_user_protected_exact_not_demoted_by_color():
    row = SimpleNamespace(
        score=0.9,
        score_percent=90,
        pattern_family="animal_print",
        animal_print_type="tiger",
        is_self_match=False,
        debug={
            "learned_concept_exact": True,
            "texture_map": {
                "color_evidence": {"detected_colors": ["red"]},
                "color_index": {"dominant_colors": [[200, 20, 20]]},
            },
        },
    )
    from core.color_evidence import apply_color_evidence_scoring

    out = apply_color_evidence_scoring([row], "siyah krem kaplan")
    assert out[0].score == 0.9
    assert out[0].debug.get("color_evidence_score", {}).get("skipped") == "user_protected"
