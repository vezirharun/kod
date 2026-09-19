from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ui" / "feedback_family_dialog.py"


def test_new_main_category_and_attribute_controls_exist():
    text = SRC.read_text(encoding="utf-8")
    assert "register_root_category" in text
    assert "btn_new_category" in text
    assert "_add_typed_category" in text
    assert "btn_new_attribute" in text
    assert "_add_typed_attribute" in text
    assert "Yeni ana kategori olarak ekle" in text
    assert "yeni nitelik olarak ekle" in text
