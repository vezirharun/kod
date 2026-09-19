"""Entity/Object Intelligence v1 — açık kavram, kanıt ayrımı, regresyon."""
from __future__ import annotations

from types import SimpleNamespace

from core.entity_intelligence import (
    INDEX_ENTITY_OVERHEAD_MS,
    annotate_search_results,
    clip_is_not_detection,
    match_score,
    object_detection_status,
    parse_entity_query,
    ranking_channels_unchanged,
    records_from_file_meta,
)
from core.index_freeze import INDEX_FROZEN
from core.visual_concept import (
    DEFAULT_FUSION_WEIGHTS,
    VisualChannelScores,
    catalog_stats,
    compile_visual_query,
    expand_clip_prompt,
    fuse_visual_channels,
    load_open_concepts,
)


def _ids(q: str) -> list[str]:
    return [e.canonical_name for e in parse_entity_query(q).entities]


def test_open_catalog_grows_without_hardcoded_seven():
    st = catalog_stats()
    assert st["categories"] >= 15
    assert st["lemmas"] >= 40
    data = load_open_concepts()
    assert data["synonyms"]["avokado"] == "avocado"
    assert data["categories"]["avocado"] == "fruit"
    assert compile_visual_query("avokado").concepts[0].concept_id.endswith("avocado")


def test_turkish_to_english_resolution():
    assert parse_entity_query("çanta").canonical_ids() == ["handbag"]
    assert parse_entity_query("kolye").canonical_ids() == ["necklace"]
    assert parse_entity_query("kiraz").canonical_ids() == ["cherry"]
    assert parse_entity_query("araba").canonical_ids() == ["car"]
    assert parse_entity_query("bisiklet").canonical_ids() == ["bicycle"]
    assert parse_entity_query("dudak").canonical_ids() == ["lips"]
    q = parse_entity_query("çantalı kadın")
    ids = set(q.canonical_ids())
    assert "handbag" in ids
    assert "person" in ids
    q2 = parse_entity_query("arabası olan")
    assert "car" in q2.canonical_ids()


def test_short_clip_prompt_bag_is_valid():
    p = expand_clip_prompt("bag")
    assert "bag" in p
    assert "photograph" in p


def test_single_and_multi_entity():
    assert _ids("kiraz") == ["cherry"]
    ids = _ids("kiraz ve çanta")
    assert ids == ["cherry", "handbag"] or set(ids) == {"cherry", "handbag"}
    ids2 = set(_ids("kadın yüzü ve kolye"))
    assert "face" in ids2 or "person" in ids2
    assert "necklace" in ids2
    ids3 = set(_ids("çantalı kadın çiçekli desen"))
    assert {"handbag", "person", "flower"} <= ids3


def test_brand_plus_entity_and_motif_color():
    q = parse_entity_query("Amiri çiçek")
    assert q.brands
    assert "flower" in q.canonical_ids()
    assert not any("amiri" in x.lower() for x in q.canonical_ids())
    q2 = parse_entity_query("kırmızı çanta")
    assert "red" in q2.colors
    assert "handbag" in q2.canonical_ids()
    q3 = parse_entity_query("Amiri kırmızı çiçek")
    assert q3.brands
    assert "red" in q3.colors
    assert "flower" in q3.canonical_ids()


def test_clip_is_not_real_entity_evidence():
    assert clip_is_not_detection("openclip") is True
    assert clip_is_not_detection("object_detector") is False
    query = parse_entity_query("çanta")
    clip_only = records_from_file_meta(
        {
            "texture_map": {
                "global_object_intelligence": {
                    "concepts": [{"label": "handbag", "confidence": 0.91}],
                }
            }
        }
    )
    score, kind, _ = match_score(query, clip_only)
    assert kind == "openclip"
    assert score <= 0.55
    det = records_from_file_meta(
        {
            "texture_map": {
                "global_object_intelligence": {
                    "objects": [{"label": "handbag", "label_tr": "çanta", "confidence": 0.94}],
                }
            }
        }
    )
    dscore, dkind, hits = match_score(query, det)
    assert dkind == "object_detector"
    assert dscore >= 0.9
    assert hits[0].evidence_source == "object_detector"


def test_pattern_dna_not_rewritten_as_detector():
    recs = records_from_file_meta(
        {"texture_map": {"pattern_dna": {"family": "floral", "confidence": 0.8}}}
    )
    assert recs
    assert recs[0].canonical_name == "flower"
    assert recs[0].evidence_source == "pattern_dna"
    assert recs[0].evidence_source != "object_detector"


def test_dino_clip_fusion_weights_not_replaced():
    w = ranking_channels_unchanged()
    assert w["clip"] == DEFAULT_FUSION_WEIGHTS["clip"]
    assert w["dino"] == DEFAULT_FUSION_WEIGHTS["dino"]
    fused = fuse_visual_channels(VisualChannelScores(clip=0.8, dino=0.6, object_index=0.2))
    assert 0.0 < fused < 1.0


def test_faiss_module_not_imported_by_entity_layer():
    import core.entity_intelligence as ei
    import sys

    assert "core.faiss_store" not in sys.modules or ei.INDEX_ENTITY_OVERHEAD_MS == 0.0
    src = Path_read = __import__("inspect").getsource(ei)
    assert "FaissStore" not in src
    assert "rebuild" not in src.lower()


def test_index_overhead_is_zero():
    assert INDEX_ENTITY_OVERHEAD_MS == 0.0
    st = object_detection_status()
    assert st["index_overhead_ms"] == 0.0
    assert st["clip_is_not_detector"] is True
    assert "object_detection_available" in st


def test_search_freeze_flag_untouched():
    assert isinstance(INDEX_FROZEN, bool)


def test_annotate_sets_debug_counter_without_closing_channels():
    result = SimpleNamespace(
        score=0.41,
        pattern_family="",
        debug={"clip_score": 0.41, "dino_score": 0.33},
        breakdown={"clip": 0.41, "dino": 0.33},
    )
    annotate_search_results([result], "çanta")
    assert result.debug.get("entity_query") == ["handbag"]
    assert result.debug.get("clip_as_entity_detection") is False
    assert result.score == 0.41
    assert result.breakdown["clip"] == 0.41
    assert result.breakdown["dino"] == 0.33


def test_example_queries_resolve_via_open_concepts():
    samples = [
        "kiraz", "kolye", "çanta", "araba", "bisiklet", "dudak", "kadın yüzü",
        "çiçek", "kelebek", "kedi", "köpek", "ayakkabı", "şapka", "gözlük",
        "yüzük", "saat", "çiçek ve çanta", "kadın ve kolye", "çanta ve çiçek",
        "kadın yüzü ve kolye",
    ]
    for q in samples:
        parsed = parse_entity_query(q)
        assert parsed.entities, q
