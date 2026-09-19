
import ast
from pathlib import Path


def _source():
    return Path(__file__).parents[1].joinpath("core", "face_scanner.py").read_text(encoding="utf-8")


def test_face_scanner_has_no_direct_source_imread():
    tree = ast.parse(_source())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "imread":
                calls.append(node)
    assert not calls, "Face Scanner kaynak dosyayı cv2.imread ile açmamalı"


def test_face_scanner_uses_preview_loader():
    src = _source()
    assert "Thumbnailer.load_image" in src
    assert "feature_preview_path" in src
    assert "thumbnail_path" in src
    assert "cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)" in src


def test_face_scanner_does_not_fallback_to_source_path():
    src = _source()
    assert "preview_path = feature_preview or thumbnail" in src
    assert "preview_path = path" not in src
