from core.brand_aliases import brand_suggestions
from core.category_tree import infer_category_path


def test_loui_and_louise_suggest_louis_vuitton():
    for q in ("loui", "louise"):
        names = [name.lower() for name, score in brand_suggestions(q)]
        assert "louis vuitton" in names


def test_unknown_explicit_brand_reference_is_preserved():
    path = infer_category_path(
        {"semantic_tags": {"brand_references": ["Polo"]}},
        ocr_text="",
    )
    assert path == "Marka/Polo"
