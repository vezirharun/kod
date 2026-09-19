from pathlib import Path

from core.db import Database


def test_label_text_search_is_not_leopard_specific():
    src = Path(Database.__module__.replace(".", "/") + ".py")
    text = src.read_text(encoding="utf-8")
    assert "animal_print_type" in text
    assert "pattern_subtype" in text
    assert "texture_map" in text and "animal_print_type" in text
    assert "leopard" not in text.split("def _label_text_search", 1)[1].split("def _blob_like_search", 1)[0].lower()


def test_text_candidate_label_pool_is_wider_than_visible_limit():
    src = Path(Database.__module__.replace(".", "/") + ".py")
    text = src.read_text(encoding="utf-8")
    assert "_label_text_search(primary[:4]" in text
