"""Responsive dialog fit — availableGeometry caps, no hardcoded resolutions."""

from __future__ import annotations

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_fit_dialog_caps_to_mocked_small_screen(monkeypatch):
    _app()
    from ui import responsive_dialog as rd

    small = QRect(0, 0, 800, 500)
    monkeypatch.setattr(rd, "available_screen_geometry", lambda _w=None: small)

    dlg = QDialog()
    dlg.setMinimumWidth(440)
    dlg.setMinimumHeight(600)
    lay = QVBoxLayout(dlg)
    for i in range(20):
        lay.addWidget(QLabel(f"row {i}"))

    avail = rd.fit_dialog_to_available_screen(
        dlg, margin=24, min_width=360, min_height=200, prefer_width=480, prefer_height=700
    )
    assert avail == small
    max_w = 800 - 48
    max_h = 500 - 48
    assert dlg.maximumWidth() == max_w
    assert dlg.maximumHeight() == max_h
    assert dlg.width() <= max_w
    assert dlg.height() <= max_h
    assert dlg.minimumHeight() <= max_h


def test_result_metadata_dialog_respects_screen_cap(monkeypatch):
    _app()
    from ui import responsive_dialog as rd
    from ui.result_metadata_dialog import ResultMetadataDialog
    from ui.teach_me_panel import TeachClassifyDialog

    tiny = QRect(10, 20, 900, 480)
    monkeypatch.setattr(rd, "available_screen_geometry", lambda _w=None: tiny)

    edit = ResultMetadataDialog()
    assert edit.height() <= tiny.height() - 48
    assert edit.maximumHeight() <= tiny.height() - 48
    assert getattr(edit, "_body_layout", None) is not None
    assert edit.btn_save is not None
    assert edit.btn_cancel is not None

    teach = TeachClassifyDialog(file_count=2, concepts=["Zincir"])
    assert teach.height() <= tiny.height() - 48
    assert teach.maximumHeight() <= tiny.height() - 48
    assert teach.txt_concept is not None
    assert teach.btn_save.text() == "Kaydet"


def test_available_geometry_uses_widget_screen():
    _app()
    from ui.responsive_dialog import available_screen_geometry

    dlg = QDialog()
    geo = available_screen_geometry(dlg)
    assert geo.width() > 0
    assert geo.height() > 0
