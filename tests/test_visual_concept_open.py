"""Open visual concept compiler — not a 7-class catalog."""
from __future__ import annotations

from core.universal_visual_intel import (
    is_hard_object_query,
    node_clip_prompt,
    parse_universal_query,
)
from core.visual_concept import (
    VisualChannelScores,
    catalog_stats,
    compile_visual_query,
    expand_clip_prompt,
    fuse_visual_channels,
    prompt_for_open_id,
)

# Kabul örnekleri (kiraz/çanta/...) özel case değil; kategori tarama kümesi.
CATEGORY_QUERIES: dict[str, list[str]] = {
    "fruit": ["kiraz", "elma", "üzüm", "incir", "nar"],
    "vegetable": ["patates", "domates", "havuç", "soğan"],
    "plant": ["çiçek", "lale", "saksı", "ağaç"],
    "jewelry": ["kolye", "küpe", "yüzük", "broş"],
    "accessory": ["çanta", "gözlük", "şemsiye", "şapka"],
    "clothing": ["ayakkabı", "gömlek", "etek"],
    "vehicle": ["araba", "otomobil", "bisiklet", "tren", "helikopter"],
    "animal": ["kedi", "köpek", "kuş", "fil"],
    "body": ["dudak", "göz", "burun", "saç"],
    "person": ["kadın yüzü", "insan"],
    "furniture": ["sandalye", "masa", "koltuk"],
    "electronics": ["mikrofon", "kamera", "telefon", "yazıcı"],
    "household": ["kitap", "bardak", "anahtar"],
    "geometric": ["üçgen", "kare", "daire"],
    "textile": ["puantiye", "paisley"],
}


def test_catalog_is_broader_than_seven_concepts():
    st = catalog_stats()
    assert st["categories"] >= 12
    assert st["synonyms"] >= 80
    assert st["lemmas"] >= 40


def test_queries_span_many_categories():
    assert len(CATEGORY_QUERIES) >= 12
    n = sum(len(v) for v in CATEGORY_QUERIES.values())
    assert n >= 40
    seen_ids: set[str] = set()
    cats_hit: set[str] = set()
    for cat, queries in CATEGORY_QUERIES.items():
        for q in queries:
            compiled = compile_visual_query(q)
            assert compiled.concepts, q
            if compiled.concepts[0].category:
                cats_hit.add(compiled.concepts[0].category)
            uq = parse_universal_query(q)
            ident = (
                uq.node_id
                or (uq.required[0] if uq.required else "")
                or compiled.concepts[0].concept_id
            )
            assert ident, q
            seen_ids.add(ident)
            prompt = node_clip_prompt(ident)
            assert prompt.strip()
            assert "photograph" in prompt or " " in prompt or len(prompt) >= 3
    assert len(seen_ids) >= 25
    assert len(cats_hit) >= 10


def test_short_class_name_bag_is_valid_clip_prompt():
    p = expand_clip_prompt("bag")
    assert "bag" in p
    assert "photograph" in p
    uq = parse_universal_query("çanta")
    prompt = node_clip_prompt(uq.node_id)
    assert len(prompt) >= 3
    assert "bag" in prompt.lower() or "photograph" in prompt.lower()


def test_open_concepts_are_not_a_closed_seven_list():
    samples = [
        "kiraz",
        "kolye",
        "çanta",
        "otomobil",
        "araba",
        "bisiklet",
        "dudak",
        "kadın yüzü",
        "üzüm",
        "çekiç",
        "mikrofon",
        "gözlük",
        "saksı",
        "nar",
        "gitar",
        "tren",
        "şemsiye",
        "yüzük",
        "patates",
        "helikopter",
    ]
    ids = []
    for q in samples:
        uq = parse_universal_query(q)
        assert uq.node_id, q
        ids.append(uq.node_id)
        prompt = node_clip_prompt(uq.node_id)
        assert len(prompt) >= 3, q
        assert "if query" not in prompt
    assert len(set(ids)) >= 12


def test_unknown_token_still_becomes_clip_prompt():
    uq = parse_universal_query("zxqwidget")
    assert uq.node_id.startswith("open:")
    assert is_hard_object_query(uq)
    p = node_clip_prompt(uq.node_id)
    assert "zxqwidget" in p or "photograph" in p


def test_synonym_and_hypernym_otomobil_araba_arac():
    compiled = compile_visual_query("otomobil")
    assert compiled.concepts
    lemma = compiled.concepts[0].concept_id
    assert "car" in lemma
    assert "vehicle" in compiled.concepts[0].hypernyms or "arac" in compiled.concepts[0].hypernyms
    uq = parse_universal_query("araba")
    assert uq.node_id in {"car", "vehicle"} or uq.node_id.startswith("open:")


def test_multi_concept_and_color():
    q = compile_visual_query("kırmızı çanta")
    assert "red" in q.colors
    assert any("handbag" in c.concept_id or "bag" in c.concept_id for c in q.concepts)
    uq = parse_universal_query("kadın ve kolye")
    blob = " ".join([uq.node_id, *uq.required, *uq.open_ids])
    assert "female" in blob or "person" in blob or "kadın" in uq.raw
    assert any("necklace" in x or "kolye" in x for x in [uq.node_id, *uq.open_ids, *uq.required])


def test_textile_query_not_hijacked_into_open_object():
    uq = parse_universal_query("çiçekli desen")
    assert not any(str(x).startswith("open:") for x in uq.open_ids)
    assert not str(uq.node_id).startswith("open:")


def test_ranking_fusion_api_has_all_channels():
    s = VisualChannelScores(clip=0.8, dino=0.5, object_index=0.2, color=0.1, dna=0.3, ocr=0.0, face=0.4)
    fused = fuse_visual_channels(s)
    assert 0.0 < fused < 1.0
    assert hasattr(s, "bbox")
    assert hasattr(s, "relation")


def test_brand_token_is_not_an_open_visual_object():
    q = compile_visual_query("amiri")
    assert q.concepts == []
    q2 = compile_visual_query("amiri çiçek")
    ids = [c.concept_id for c in q2.concepts]
    assert not any("amiri" in x for x in ids)
    assert any("cicek" in x or "flower" in x or "floral" in x for x in ids)


def test_style_and_effect_are_not_object_photos():
    from core.visual_concept import english_lemma, prompt_for_lemma

    assert english_lemma("barok") == "baroque"
    assert english_lemma("ekose") == "plaid"
    assert english_lemma("takı") == "jewelry"
    p = prompt_for_lemma("baroque")
    assert "ornate" in p or "textile" in p
    assert "real object" not in p
    q = compile_visual_query("fırça etkisi")
    ids = [c.concept_id for c in q.concepts]
    assert ids == ["open:brushstroke"]
    uq = parse_universal_query("barok")
    assert uq.node_id == "baroque_pattern"
    uq2 = parse_universal_query("etnik")
    assert uq2.node_id == "ethnic_print"
