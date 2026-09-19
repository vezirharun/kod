import sqlite3
from pathlib import Path

from core.category_tree import child_categories, resolve_category_query
from core.category_memory import register_category, category_suggestions
from core.entity_evidence import extract_entity_evidence
from core.text_index import build_text_search_blob


def test_person_and_vehicle_categories_are_searchable(tmp_path):
    db = str(tmp_path / "test.db")
    assert resolve_category_query("kadın", db_path=db).category_path == "Kişi/Kadın"
    assert resolve_category_query("T10X", db_path=db).category_path == "Araba Modeli/T10X"
    assert resolve_category_query("broş", db_path=db).category_path == "Aksesuar/Broş"


def test_dynamic_subcategory_search_and_persistence(tmp_path):
    db = str(tmp_path / "test.db")
    assert register_category(db, "Araba Modeli", "YeniModel", aliases=["ym"])
    assert "YeniModel" in child_categories("Araba Modeli", db)
    assert resolve_category_query("ym", db_path=db).category_path == "Araba Modeli/YeniModel"
    assert category_suggestions(db, "yenimodel")


def test_multiple_entity_attributes_persist_into_evidence_and_text():
    attrs = [
        {"path": "Araç Detayı/SUV", "parent": "Araç Detayı", "label": "SUV", "source": "SUV"},
        {"path": "Araba Markası/Togg", "parent": "Araba Markası", "label": "Togg", "source": "Togg"},
        {"path": "Araba Modeli/T10X", "parent": "Araba Modeli", "label": "T10X", "source": "T10X"},
    ]
    tm = {"entity_attributes": attrs}
    rec = {"texture_map": tm}
    ev = extract_entity_evidence(rec)
    assert "togg" in ev
    assert "t10x" in ev
    blob = build_text_search_blob(filename="x.jpg", texture_map=tm)
    assert "togg" in blob and "t10x" in blob and "suv" in blob


if __name__ == "__main__":
    print("test_global_category_assignment_v2: OK")
