from __future__ import annotations

from core.color_index import apply_auto_color, resolved_color_family
from core.index_enrichments import enrich_texture_map_if_missing
from core.texture_profile import TextureProfile


def test_auto_color_overrides_coarse_brown_tan():
    tm = {
        "color_family": "brown_tan",
        "color_index": {"color_family": "blue", "palette": ["mavi"]},
    }
    out = apply_auto_color(tm)
    assert out["color_family"] == "navy_blue"
    assert out["color_source"] == "auto_index"


def test_taught_color_is_not_overwritten():
    tm = {
        "color_family": "red",
        "color_source": "teach_me",
        "color_index": {"color_family": "blue"},
    }
    out = apply_auto_color(tm)
    assert out["color_family"] == "red"


def test_search_profile_uses_color_index():
    prof = TextureProfile.from_dict(
        {
            "color_family": "brown_tan",
            "color_index": {"color_family": "green"},
        }
    )
    assert prof.color_family == "green"


def test_enrich_fills_missing_color_from_index():
    tm = enrich_texture_map_if_missing(
        {"color_index": {"color_family": "gold", "palette": ["altin"]}}
    )
    assert tm["color_family"] == "gold"


def test_resolved_prefers_index_over_file_column():
    assert (
        resolved_color_family(
            {"color_family": "unknown", "color_index": {"color_family": "khaki"}},
            file_color="brown_tan",
        )
        == "khaki"
    )
