"""Bana Öğret kart layout — IconMode overlap olmamalı."""
from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import QApplication, QListWidget, QListWidgetItem

from core.teach_me import TeachMeCard
from ui.teach_me_panel import (
    _THUMB_EDGE,
    _card_item_size_hint,
    _card_list_text,
    _items_overlap,
    _preview_pix,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def _card(**kwargs) -> TeachMeCard:
    base = dict(
        file_id=1,
        filename="x.png",
        path="/x.png",
        preview_path="",
        pool="undefined",
        reason="Anlamlı bir sınıf yok",
        guess="Belirsiz",
        confidence=0.5,
    )
    base.update(kwargs)
    return TeachMeCard(**base)


def _fill_icon_list(n: int, *, long_name: bool = False, long_reason: bool = False) -> QListWidget:
    w = QListWidget()
    w.setViewMode(QListWidget.ViewMode.IconMode)
    w.setResizeMode(QListWidget.ResizeMode.Adjust)
    w.setIconSize(QSize(_THUMB_EDGE, _THUMB_EDGE))
    w.setWordWrap(True)
    w.setSpacing(14)
    w.setUniformItemSizes(False)
    w.setMovement(QListWidget.Movement.Static)
    w.resize(900, 700)
    for i in range(n):
        name = (
            "WOMENS KIKO PRINT OVERVIEW-1-32 arka sol.tif"
            if long_name
            else f"file_{i}.png"
        )
        reason = (
            "Birden fazla güçlü tahmin: Small Floral / yaprak / Mixed Floral · "
            "Sistem ayrım yapamadı ve kullanıcı onayı bekliyor"
            if long_reason
            else f"reason {i}"
        )
        card = _card(
            file_id=i + 1,
            filename=name,
            guess="Small Floral / yaprak / Mixed Floral",
            confidence=0.92,
            reason=reason,
        )
        text = _card_list_text(card)
        item = QListWidgetItem(text)
        item.setSizeHint(_card_item_size_hint(text))
        item.setIcon(__import__("PySide6.QtGui", fromlist=["QIcon"]).QIcon(_preview_pix("")))
        w.addItem(item)
    w.show()
    w.doItemsLayout()
    QApplication.processEvents()
    return w


def _visible_rects(w: QListWidget) -> list[QRect]:
    out = []
    for i in range(w.count()):
        item = w.item(i)
        r = w.visualItemRect(item)
        out.append(QRect(r))
    return out


def test_size_hint_taller_than_thumb_alone(qapp):
    text = _card_list_text(
        _card(
            filename="WOMENS KIKO PRINT OVERVIEW-1-32 arka sol.tif",
            guess="Small Floral / yaprak / Mixed Floral",
            reason="Birden fazla güçlü tahmin: A / B / C",
            confidence=0.92,
        )
    )
    hint = _card_item_size_hint(text)
    assert hint.height() > _THUMB_EDGE + 40
    assert hint.width() >= _THUMB_EDGE


def test_ten_cards_no_overlap(qapp):
    w = _fill_icon_list(10, long_name=True, long_reason=True)
    assert not _items_overlap(_visible_rects(w))


def test_eighty_cards_no_overlap(qapp):
    w = _fill_icon_list(80, long_name=True)
    assert w.count() == 80
    assert not _items_overlap(_visible_rects(w))


def test_long_filename_no_overlap(qapp):
    w = _fill_icon_list(12, long_name=True)
    assert not _items_overlap(_visible_rects(w))


def test_long_reason_no_overlap(qapp):
    w = _fill_icon_list(12, long_reason=True)
    assert not _items_overlap(_visible_rects(w))


def test_placeholder_same_geometry_contract(qapp):
    pix = _preview_pix("")
    assert pix.width() == _THUMB_EDGE
    assert pix.height() == _THUMB_EDGE
    text = _card_list_text(_card())
    h1 = _card_item_size_hint(text).height()
    assert h1 > _THUMB_EDGE


def test_narrow_and_wide_window_no_overlap(qapp):
    for width in (420, 700, 1100, 1400):
        w = _fill_icon_list(24, long_name=True, long_reason=True)
        w.resize(width, 800)
        w.doItemsLayout()
        QApplication.processEvents()
        assert not _items_overlap(_visible_rects(w)), f"overlap at width={width}"


def test_scroll_content_covers_last_item(qapp):
    w = _fill_icon_list(40, long_name=True)
    last = w.item(w.count() - 1)
    rect = w.visualItemRect(last)
    # İçerik yüksekliği son kartın alt kenarını kapsamalı
    assert rect.bottom() <= w.contentsSize().height() + w.spacing()
    assert rect.bottom() > _THUMB_EDGE
