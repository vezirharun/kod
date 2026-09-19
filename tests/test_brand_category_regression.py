from pathlib import Path

from core.category_tree import (
    infer_category_path,
    resolve_category_query,
)
from core.db import Database


def test_brand_ocr_overrides_generic_monogram_family():
    tm = {"pattern_family": "monogram_logo", "pattern_subtype": "repeated_logo"}
    assert (
        infer_category_path(
            tm,
            filename="christian-dior-logo.png",
            ocr_text="CHRISTIAN DIOR",
        )
        == "Marka/Christian Dior"
    )
    # Filename alone is not brand proof.
    assert (
        infer_category_path(
            tm,
            filename="dior.png",
            ocr_text="",
        )
        == "Monogram Logo"
    )


def test_brand_query_resolves_to_brand_leaf():
    assert resolve_category_query("marka").category_path == "Marka"
    assert resolve_category_query("dior").category_path == "Marka/Christian Dior"
    assert resolve_category_query("amiri").category_path == "Marka/Amiri"


def test_brand_category_filter_finds_existing_ocr_rows(tmp_path):
    db = Database(tmp_path / "brand_filter.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO files(path, filename, status, ocr_text, category_path, manual_category_path) "
            "VALUES (?, ?, 'indexed', ?, '', '')",
            ("F:/patterns/a.png", "a.png", "CHRISTIAN DIOR"),
        )
        conn.execute(
            "INSERT INTO features(file_id, texture_map) VALUES (?, ?)",
            (1, '{"semantic_tags":{"brand_references":["dior"]}}'),
        )
        conn.commit()

    assert db.list_file_ids_by_category_path("Marka/Christian Dior") == [1]
    assert db.list_file_ids_by_category_path("Marka") == [1]


def test_brand_text_search_scores_legacy_ocr_row_without_category_path():
    from core.text_index import text_search_score

    rec = {
        "filename": "generic-logo.png",
        "ocr_text": "CHRISTIAN DIOR",
        "category_path": "",
        "manual_category_path": "",
        "texture_map": {"pattern_family": "monogram_logo"},
    }
    score, breakdown, reasons = text_search_score("marka", rec)
    assert score >= 0.90
    assert breakdown["family_score"] >= 0.90
    assert any("Marka kanıtı" in r for r in reasons)


def test_brand_category_filter_finds_manual_learning_when_category_column_is_stale(tmp_path):
    db = Database(tmp_path / "brand_manual_stale.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO files(path, filename, status, ocr_text, category_path, manual_category_path, category_aliases) "
            "VALUES (?, ?, 'indexed', ?, '', '', ?)",
            ("F:/patterns/amiri-01.jpg", "amiri-01.jpg", "", '["amiri"]'),
        )
        conn.execute(
            "INSERT INTO features(file_id, texture_map) VALUES (?, ?)",
            (1, '{"brand_name":"Amiri","semantic_tags":{"brand_references":["amiri"]}}'),
        )
        conn.execute(
            "INSERT INTO user_feedback(query_file_id, query_path, result_file_id, action, label, note, created_at) "
            "VALUES (0, '', 1, 'custom_tag', 'Amiri', '', '')"
        )
        conn.commit()

    assert db.list_file_ids_by_category_path("Marka/Amiri") == [1]
    assert db.list_file_ids_by_category_path("Marka") == [1]


def test_brand_category_filter_includes_text_search_blob_for_monogram_legacy(tmp_path):
    db = Database(tmp_path / "brand_text_blob.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO files(path, filename, status, ocr_text, category_path, manual_category_path, text_search_blob) "
            "VALUES (?, ?, 'indexed', ?, 'Monogram Logo', '', ?)",
            (
                "F:/patterns/dior-monogram.jpg",
                "dior-monogram.jpg",
                "",
                "monogram logo christian dior dior",
            ),
        )
        conn.execute(
            "INSERT INTO features(file_id, texture_map) VALUES (?, ?)",
            (1, '{"pattern_family":"monogram_logo"}'),
        )
        conn.commit()

    assert db.list_file_ids_by_category_path("Marka/Christian Dior") == [1]
    assert db.list_file_ids_by_category_path("Marka") == [1]


def test_person_print_category_resolves_and_infers_from_ocr():
    assert resolve_category_query("Marilyn Monroe").category_path == "Kişi / Ünlü/Marilyn Monroe"
    assert resolve_category_query("Kenan İmirzalıoğlu").category_path == "Kişi / Ünlü/Kenan İmirzalıoğlu"
    assert infer_category_path(
        {"pattern_family": "typography_text", "semantic_tags": {"motifs": ["Marilyn Monroe"]}},
        ocr_text="MARILYN MONROE",
    ) == "Kişi / Ünlü/Marilyn Monroe"


def test_person_category_filter_includes_legacy_text_blob(tmp_path):
    db = Database(tmp_path / "person_filter.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO files(path, filename, status, ocr_text, category_path, manual_category_path, text_search_blob) "
            "VALUES (?, ?, 'indexed', ?, 'Drawing Illustration', '', ?)",
            ("F:/patterns/marilyn.jpg", "marilyn.jpg", "", "portrait marilyn monroe celebrity"),
        )
        conn.execute("INSERT INTO features(file_id, texture_map) VALUES (?, ?)", (1, '{"pattern_family":"typography_text"}'))
        conn.commit()
    assert db.list_file_ids_by_category_path("Kişi / Ünlü/Marilyn Monroe") == [1]
    assert db.list_file_ids_by_category_path("Kişi / Ünlü") == [1]


def test_lightweight_indexed_files_keeps_brand_category_metadata(tmp_path):
    db = Database(tmp_path / "brand_lightweight.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO files(path, filename, status, category_path, manual_category_path, category_aliases, text_search_blob) "
            "VALUES (?, ?, 'indexed', ?, ?, ?, ?)",
            (
                "F:/patterns/dior-monogram.jpg",
                "dior-monogram.jpg",
                "Marka/Christian Dior",
                "",
                '["dior","christian dior"]',
                "christian dior dior monogram",
            ),
        )
        conn.commit()
    rows = db.get_indexed_files(lightweight=True)
    assert rows and rows[0]["category_path"] == "Marka/Christian Dior"
    assert rows[0]["category_aliases"] == '["dior","christian dior"]'
    assert "christian dior" in rows[0]["text_search_blob"]


def test_brand_category_parent_is_superset_of_every_brand_child(tmp_path):
    db = Database(tmp_path / "brand_parent_superset_db.db")
    rows = [
        ("F:/patterns/amiri.png", "amiri.png", "AMIRI", "Monogram Logo", "", '["amiri"]', "amiri monogram"),
        ("F:/patterns/dior.png", "dior.png", "CHRISTIAN DIOR", "Monogram Logo", "", '["dior"]', "christian dior dior"),
        ("F:/patterns/lv.png", "lv.png", "", "Marka/Louis Vuitton", "", '["lv","louis vuitton"]', "louis vuitton lv"),
    ]
    with db.connect() as conn:
        for path, filename, ocr, cat, manual, aliases, blob in rows:
            conn.execute(
                "INSERT INTO files(path, filename, status, ocr_text, category_path, manual_category_path, category_aliases, text_search_blob) "
                "VALUES (?, ?, 'indexed', ?, ?, ?, ?, ?)",
                (path, filename, ocr, cat, manual, aliases, blob),
            )
            conn.execute(
                "INSERT INTO features(file_id, texture_map) VALUES (last_insert_rowid(), ?)",
                ('{}',),
            )
        conn.commit()

    parent = set(db.list_file_ids_by_category_path("Marka", limit=0))
    amiri = set(db.list_file_ids_by_category_path("Marka/Amiri", limit=0))
    dior = set(db.list_file_ids_by_category_path("Marka/Christian Dior", limit=0))
    lv = set(db.list_file_ids_by_category_path("Marka/Louis Vuitton", limit=0))
    assert amiri <= parent
    assert dior <= parent
    assert lv <= parent
    assert parent == {1, 2, 3}
