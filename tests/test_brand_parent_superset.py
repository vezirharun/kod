from core.brand_evidence import extract_brand_evidence
from core.category_tree import resolve_category_query
from core.text_index import text_search_score


def test_brand_parent_is_general_and_children_are_specific():
    assert resolve_category_query("marka").category_path == "Marka"
    assert resolve_category_query("dior").category_path == "Marka/Christian Dior"
    assert resolve_category_query("amiri").category_path == "Marka/Amiri"


def test_brand_evidence_is_shared_across_sources():
    dior = {
        "filename": "Christian-Dior-logo.png",
        "category_path": "Monogram Logo",
        "category_aliases": '["dior", "christian dior"]',
        "ocr_text": "",
        "text_search_blob": "",
        "texture_map": {"semantic_tags": {"brand_references": ["dior"]}},
    }
    amiri = {
        "filename": "amiri-monogram.jpg",
        "category_path": "Monogram Logo",
        "category_aliases": [],
        "ocr_text": "AMIRI",
        "text_search_blob": "",
        "texture_map": {},
    }
    assert "christian dior" in extract_brand_evidence(dior)
    assert "amiri" in extract_brand_evidence(amiri)


def test_dg_filename_is_dolce_gabbana_evidence():
    rec = {"filename": "5154MC fou D&G.jpg", "path": "F:/done/5154MC fou D&G.jpg"}
    assert "dolce gabbana" in extract_brand_evidence(rec)


def test_general_brand_query_gets_brand_evidence_score():
    rec = {
        "filename": "dior-oblique.jpg",
        "ocr_text": "",
        "category_path": "Monogram Logo",
        "manual_category_path": "",
        "category_aliases": '["dior"]',
        "texture_map": {},
    }
    score, breakdown, reasons = text_search_score("marka", rec)
    assert score >= 0.90
    assert breakdown["brand_alias_score"] >= 0.90
    assert "Marka kanıtı" in reasons


def test_brand_evidence_survives_manual_mark_brand_category_and_nested_metadata():
    from core.brand_evidence import extract_brand_evidence

    rec = {
        "filename": "generic-monogram.png",
        "category_path": "Marka/Amiri",
        "manual_category_path": "Marka/Amiri",
        "category_aliases": '[]',
        "texture_map": {
            "pattern_family": "monogram_logo",
            "category_path": "Marka/Amiri",
            "manual_category_path": "Marka/Amiri",
            "semantic_tags": {"brand_references": []},
        },
    }
    assert extract_brand_evidence(rec) == {"amiri"}


def test_brand_evidence_is_canonical_for_all_specific_queries():
    from core.brand_evidence import extract_brand_evidence

    records = [
        {"filename": "amiri_01.jpg", "category_path": "Marka/Amiri"},
        {"filename": "dior_01.jpg", "category_path": "Marka/Christian Dior"},
        {"filename": "lv_01.jpg", "category_path": "Marka/Louis Vuitton", "brand_name": "Louis Vuitton"},
    ]
    evidence = [extract_brand_evidence(r) for r in records]
    assert evidence[0] == {"amiri"}
    assert evidence[1] == {"christian dior"}
    assert evidence[2] == {"louis vuitton"}
    # Genel marka kümesi, tek tek marka kümelerinin birleşimidir.
    assert set().union(*evidence) == {"amiri", "christian dior", "louis vuitton"}
