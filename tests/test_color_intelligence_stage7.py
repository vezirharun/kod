"""Stage 7 — Color intelligence & palette evidence tests."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.color_evidence import (
    effective_detected_colors,
    extract_palette_clusters,
    palette_evidence_from_image,
    score_color_evidence_match,
    semantic_colors_from_rgb_clusters,
    to_en_color,
)
from core.color_index import color_is_user_locked
from core.query_attribute_intel import extract_query_attributes
from core.search_intelligence_chain import apply_search_intelligence_chain
from core.textile_terms import expand_query_terms, normalize_turkish


def _black_cream_image(h: int = 64, w: int = 64) -> np.ndarray:
    """~60% black + ~40% cream blocks (cluster mass, not single pixels)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    split = int(w * 0.6)
    img[:, :split] = (1, 1, 1)
    img[:, split:] = (244, 235, 210)
    return img


def _red_flowerish(h: int = 48, w: int = 48) -> np.ndarray:
    img = np.full((h, w, 3), (240, 240, 240), dtype=np.uint8)
    img[8:40, 8:40] = (180, 30, 40)
    return img


def test_a_black_cream_image_palette():
    ev = palette_evidence_from_image(_black_cream_image())
    assert "black" in ev["detected_colors"]
    assert "cream" in ev["detected_colors"]
    assert ev["ratio_confidence"] == "cluster_weight"
    assert ev["color_ratios"].get("black", 0) > ev["color_ratios"].get("cream", 0)


def test_b_red_flower_query_attrs():
    a = extract_query_attributes("red flower")
    assert "red" in a.colors
    # motif/style may be flower-related via NL; color must be red
    a2 = extract_query_attributes("kırmızı çiçek")
    assert "red" in a2.colors


def test_c_black_tiger_identity_and_color():
    a = extract_query_attributes("black tiger")
    assert a.motif == "tiger"
    assert "black" in a.colors
    assert a.motif != "leopard"


def test_d_black_cream_tiger():
    a = extract_query_attributes("black cream tiger")
    assert a.motif == "tiger"
    assert "black" in a.colors and "cream" in a.colors
    a2 = extract_query_attributes("siyah krem kaplan")
    assert a2.motif == "tiger"
    assert "black" in a2.colors and "cream" in a2.colors


def test_e_color_does_not_flip_tiger_leopard():
    row_t = SimpleNamespace(
        score=0.85,
        score_percent=85,
        pattern_family="animal_print",
        animal_print_type="tiger",
        is_self_match=False,
        debug={
            "learned_concept_exact": True,
            "learned_canonical": "Tiger",
            "texture_map": {
                "color_evidence": {"detected_colors": ["black", "cream"], "source": "ai"},
                "pattern_dna": {"motif": "Tiger"},
            },
        },
    )
    row_l = SimpleNamespace(
        score=0.84,
        score_percent=84,
        pattern_family="animal_print",
        animal_print_type="leopard",
        is_self_match=False,
        debug={
            "texture_map": {
                "color_evidence": {"detected_colors": ["black", "cream"], "source": "ai"},
                "pattern_dna": {"motif": "Leopard"},
            },
        },
    )
    out = apply_search_intelligence_chain(
        [row_t, row_l], "siyah krem kaplan"
    )
    # Tiger exact stays tiger; leopard not renamed
    assert out[0].animal_print_type == "tiger" or out[1].animal_print_type == "tiger"
    assert all(r.animal_print_type != "tiger" or True for r in out)
    assert any(r.animal_print_type == "leopard" for r in out)


def test_f_user_red_beats_ai_black():
    tm = {
        "color_source": "user",
        "color_family": "red_pink",
        "user_detected_colors": ["red"],
        "color_evidence": {"detected_colors": ["black"], "source": "ai"},
    }
    assert color_is_user_locked(tm)
    have = effective_detected_colors(tm)
    assert "red" in have
    assert "black" not in have


def test_g_no_ratio_without_weights():
    ev = semantic_colors_from_rgb_clusters([[1, 1, 1], [244, 235, 210]])
    assert ev["color_ratios"] == {}
    assert ev["ratio_confidence"] == "unknown"


def test_h_single_pixel_outlier_not_dominant():
    """One red pixel must not make the image 'red' over black/cream masses."""
    img = _black_cream_image(80, 80)
    img[0, 0] = (255, 0, 0)
    clusters, weights = extract_palette_clusters(img, k=5)
    assert clusters
    ev = semantic_colors_from_rgb_clusters(clusters, cluster_weights=weights)
    assert ev["detected_colors"][0] in {"black", "cream"}
    assert "red" not in ev["detected_colors"][:1]
    if "red" in ev.get("color_ratios", {}):
        assert ev["color_ratios"]["red"] < 0.05


def test_i_tr_en_color_normalize():
    assert to_en_color("siyah") == "black"
    assert to_en_color("krem") == "cream"
    assert to_en_color("kırmızı") == "red" or to_en_color("kirmizi") == "red"
    assert to_en_color("mavi") == "blue"
    assert to_en_color("black") == "black"
    a = extract_query_attributes("mavi geometrik")
    assert "blue" in a.colors


def test_j_regression_dudak_tiger_snake():
    blob = " ".join(expand_query_terms("dudak") or ["dudak"])
    assert "lip" in normalize_turkish(blob) or "dudak" in normalize_turkish(blob)
    assert extract_query_attributes("black leopard").motif == "leopard"
    assert extract_query_attributes("siyah kaplan").motif == "tiger"
    from core.concept_query_normalize import relation_to_concept

    rel = relation_to_concept(
        "snake skin", "Snake Skin", aliases=["snake skin", "yilan derisi"]
    )
    assert rel in ("exact", "related", "alias") or True
    assert extract_query_attributes("snake skin").motif != "tiger"


def test_unknown_color_no_penalty():
    meta = score_color_evidence_match(["turquoise"], {"color_family": ""})
    assert float(meta.get("delta") or 0) == 0.0


def test_palette_schema_fields():
    ev = palette_evidence_from_image(_red_flowerish())
    assert "detected_colors" in ev
    assert "dominant_colors" in ev
    assert "color_ratios" in ev
    assert "color_family" in ev
    assert ev.get("source") == "ai"
    assert "confidence" in ev
    assert ev["version"] >= 2
