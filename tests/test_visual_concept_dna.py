"""Görsel Nesne/Kavram DNA — index persist + search routing. No FAISS rebuild."""
from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from core.fusion_query_v2 import apply_fusion_query_v2, parse_fusion_query
from core.global_object_intelligence import DetectedObject
from core.index_freeze import INDEX_FROZEN
from core.index_object_evidence import attach_preview_object_evidence
from core.object_index import ObjectIndexStore
from core.visual_concept_dna import (
    build_visual_concept_dna,
    parse_search_bags,
)
from core.visual_pattern_query import apply_visual_pattern_priority, parse_visual_pattern_concepts


def _row(**kwargs):
    defaults = dict(
        filename="x.jpg",
        path="",
        color_family="",
        pattern_family="",
        score=0.50,
        score_percent=50.0,
        breakdown={},
        debug={},
        file_id=0,
        is_self_match=False,
    )
    defaults.update(kwargs)
    defaults["breakdown"] = dict(defaults["breakdown"])
    defaults["debug"] = dict(defaults["debug"])
    return SimpleNamespace(**defaults)


def test_index_frozen():
    assert INDEX_FROZEN is True


def test_bags_routing():
    assert parse_search_bags("ekose").bags == ("pattern",)
    assert "object" in parse_search_bags("kiraz").bags
    assert "pattern" in parse_search_bags("leopar").bags
    leo_d = parse_search_bags("leopar desen")
    assert leo_d.pattern_weighted is True
    assert parse_fusion_query("ekose").patterns
    assert parse_fusion_query("LV").brand or parse_fusion_query("Louis Vuitton").brand
    kyz = parse_search_bags("kadın yüz")
    assert "person" in kyz.bags
    barok = parse_search_bags("barok")
    assert "pattern" in barok.bags
    assert "baroque_pattern" not in barok.objects
    assert "object" not in barok.bags
    taki = parse_search_bags("takı")
    assert "jewelry" in taki.objects
    assert "pattern" not in taki.bags


def test_index_persists_concept_dna_not_patterns_db(tmp_path):
    preview = tmp_path / "feat.webp"
    Image.new("RGB", (64, 64), (180, 20, 20)).save(preview, "WEBP")
    obj_db = tmp_path / "object_index.db"

    def fake_detect(_path: str):
        return []

    goi = attach_preview_object_evidence(
        file_id=11,
        preview_path=str(preview),
        object_db_path=str(obj_db),
        detector=fake_detect,
        concepts=[
            {
                "label": "cherry",
                "label_tr": "kiraz",
                "canonical_name": "cherry",
                "confidence": 0.94,
                "category": "fruit",
            }
        ],
    )
    assert goi and goi["clip_as_detector"] is False
    dna = goi["visual_concept_dna"]
    assert "kiraz" in dna["summary"].lower() or "cherry" in str(dna).lower()
    assert dna["concepts"][0]["detected"] is False
    store = ObjectIndexStore(obj_db, readonly=True)
    rows = store.concepts_for_file(11)
    assert any(r["lemma"] == "cherry" for r in rows)
    assert "patterns.db" not in str(obj_db)


def test_kiraz_dna_found_without_filename():
    hit = _row(
        filename="scan_8841.jpg",
        score=0.31,
        breakdown={"clip": 0.31, "filename_score": 0.0},
        debug={
            "clip_score": 0.31,
            "texture_map": {
                "visual_concept_dna": build_visual_concept_dna(
                    concepts=[{
                        "lemma": "cherry",
                        "label": "cherry",
                        "label_tr": "kiraz",
                        "canonical_name": "cherry",
                        "confidence": 0.91,
                        "category": "fruit",
                    }]
                )
            },
        },
    )
    floral = _row(
        filename="red_floral.jpg",
        pattern_family="floral",
        score=0.70,
        breakdown={"clip": 0.70},
        debug={
            "clip_score": 0.70,
            "entity_evidence_kind": "openclip",
            "texture_map": {"pattern_dna": {"family": "floral"}},
        },
    )
    fillers = [
        _row(
            filename=f"style_{i}.jpg",
            score=0.70,
            breakdown={"clip": 0.70},
            debug={"clip_score": 0.70, "entity_evidence_kind": "openclip"},
        )
        for i in range(50)
    ]
    out = apply_fusion_query_v2(fillers + [floral, hit], "kiraz")
    assert hit in out[:50]
    assert floral not in out[:50]
    assert "Kanıt yok" not in str((hit.debug or {}).get("kanit_yok_label") or "")
    assert "tespit edildi" in str((hit.debug or {}).get("concept_found_label") or "")


def test_red_floral_clip_only_not_kiraz_top50():
    clips = [
        _row(
            filename=f"red_{i}.jpg",
            pattern_family="floral",
            score=0.70,
            breakdown={"clip": 0.70},
            debug={"clip_score": 0.70, "entity_evidence_kind": "openclip"},
        )
        for i in range(50)
    ]
    out = apply_fusion_query_v2(clips, "kiraz")
    assert clips[0] not in out[:50]


def test_leopar_desen_is_pattern_weighted():
    plaid_not = _row(
        filename="x.jpg",
        pattern_family="plaid_check",
        score=0.4,
        debug={"texture_map": {"pattern_dna": {"family": "plaid_check"}}},
    )
    leo = _row(
        filename="print.jpg",
        pattern_family="animal_print",
        score=0.4,
        breakdown={"family_score": 0.9, "dna_score": 0.88},
        debug={
            "texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "leopard"}},
            "animal_print_type": "leopard",
        },
    )
    cons = parse_visual_pattern_concepts("leopar desen")
    out = apply_visual_pattern_priority(
        [plaid_not, leo], "leopar desen", concepts=cons, reject_unmatched=True,
    )
    assert out[0] is leo
    plaid2 = _row(
        filename="x.jpg",
        pattern_family="plaid_check",
        score=0.4,
        debug={"texture_map": {"pattern_dna": {"family": "plaid_check"}}},
    )
    leo2 = _row(
        filename="print.jpg",
        pattern_family="animal_print",
        score=0.4,
        breakdown={"family_score": 0.9, "dna_score": 0.88},
        debug={
            "texture_map": {"pattern_dna": {"family": "animal_print", "animal_print_type": "leopard"}},
            "animal_print_type": "leopard",
        },
    )
    out2 = apply_fusion_query_v2([plaid2, leo2], "leopar")
    assert out2[0] is leo2


def test_kadin_without_person_out_of_top50():
    clips = [
        _row(
            filename=f"kumas_{i}.jpg",
            score=0.82,
            breakdown={"clip": 0.82},
            debug={"clip_score": 0.82, "entity_evidence_kind": "openclip"},
        )
        for i in range(50)
    ]
    out = apply_fusion_query_v2(clips, "kadın")
    assert all(c not in out[:50] for c in clips)


def test_ekose_still_pattern_lv_still_brand():
    plaid = _row(
        filename="d.jpg",
        pattern_family="plaid_check",
        score=0.4,
        breakdown={"family_score": 0.91, "dna_score": 0.86},
        debug={"texture_map": {"pattern_dna": {"family": "plaid_check", "motif": "tartan"}}},
    )
    named = _row(
        filename="ekose_look.jpg",
        pattern_family="floral",
        score=0.93,
        breakdown={"filename_score": 0.95},
        debug={"texture_map": {"pattern_dna": {"family": "floral"}}},
    )
    out = apply_fusion_query_v2([named, plaid], "ekose")
    assert out[0] is plaid

    lv = _row(
        filename="LV_logo.jpg",
        score=1.0,
        breakdown={"brand_alias_score": 0.95, "brand_evidence_hit": 1.0},
        debug={"brand_evidence": True},
    )
    junk = _row(
        filename="print.jpg",
        pattern_family="floral",
        score=0.74,
        breakdown={"clip": 0.74},
        debug={"clip_score": 0.74},
    )
    outb = apply_fusion_query_v2([junk, lv], "Louis Vuitton")
    assert outb[0] is lv
