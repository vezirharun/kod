from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ui" / "feedback_family_dialog.py"


def test_mixed_animal_picker_exists():
    text = SRC.read_text(encoding="utf-8")
    assert "_refresh_mixed_animal_picker" in text
    assert "Karışım bileşenleri" in text
    assert '"Animal Print"' in text
    assert '"Mixed Animal Print"' in text


def test_mixed_animal_components_are_saved_as_attributes():
    text = SRC.read_text(encoding="utf-8")
    assert 'path = f"Animal Print/{child}"' in text
    assert 'self._add_attribute_path(path, item.text(), silent=True)' in text
    assert 'self._entity_attributes.append({' in text


def test_mixed_animal_component_can_be_removed():
    text = SRC.read_text(encoding="utf-8")
    assert "def _remove_attribute_path" in text
    assert 'self._remove_attribute_path(path)' in text
