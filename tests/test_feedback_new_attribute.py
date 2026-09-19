import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.category_tree import resolve_category_query, child_categories
from ui.feedback_family_dialog import FeedbackFamilyDialog


def _app():
    app = QApplication.instance()
    return app or QApplication([])


def test_unknown_attribute_can_be_created_under_selected_parent(tmp_path):
    _app()
    db = str(tmp_path / "feedback.db")
    dlg = FeedbackFamilyDialog(db_path=db)

    idx = dlg.cmb_parent.findData("Marble Abstract")
    assert idx >= 0
    dlg.cmb_parent.setCurrentIndex(idx)

    dlg.txt_attribute_search.setText("Boya Lekesi")
    dlg._add_attribute_from_search()

    assert dlg.btn_new_attribute.isVisible()
    assert "Boya Lekesi" in dlg.btn_new_attribute.text()

    dlg._add_typed_attribute()

    attrs = dlg.entity_attributes()
    assert attrs
    assert attrs[-1]["path"] == "Nitelik/Boya Lekesi"
    assert "Boya Lekesi" in child_categories("Nitelik", db)
    assert resolve_category_query("boya lekesi", db_path=db).category_path == "Nitelik/Boya Lekesi"


def test_multiple_new_attributes_are_independent(tmp_path):
    _app()
    db = str(tmp_path / "feedback2.db")
    dlg = FeedbackFamilyDialog(db_path=db)
    dlg.cmb_parent.setCurrentIndex(dlg.cmb_parent.findData("Soyut"))

    for value in ("Boya Lekesi", "Sıçrama"):
        dlg.txt_attribute_search.setText(value)
        dlg._add_attribute_from_search()
        assert dlg.btn_new_attribute.isVisible()
        dlg._add_typed_attribute()

    paths = [x["path"] for x in dlg.entity_attributes()]
    assert paths == ["Nitelik/Boya Lekesi", "Nitelik/Sıçrama"]


def test_soyut_search_selects_parent_without_marble(tmp_path):
    _app()
    db = str(tmp_path / "feedback3.db")
    dlg = FeedbackFamilyDialog(db_path=db)

    dlg.txt_category_search.setText("soyut")
    dlg._find_and_select_category()

    assert dlg.parent_category() == "Soyut"
    assert dlg.child_category() == ""
    assert dlg.category_path_value() == "Soyut"


def test_attribute_can_be_added_without_parent(tmp_path):
    _app()
    db = str(tmp_path / "feedback4.db")
    dlg = FeedbackFamilyDialog(db_path=db)

    # Kategori seçmeden de bağımsız nitelik öğretilebilir.
    dlg.txt_attribute_search.setText("fırça")
    dlg._add_attribute_from_search()
    assert dlg.btn_new_attribute.isVisible()
    dlg._add_typed_attribute()

    assert dlg.entity_attributes()[-1]["path"] == "Nitelik/fırça"
