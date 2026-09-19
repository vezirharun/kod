"""Screen-aware dialog sizing — availableGeometry, never hardcoded resolutions."""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

DEFAULT_MARGIN = 24
DEFAULT_MIN_WIDTH = 360
DEFAULT_MIN_HEIGHT = 200


def available_screen_geometry(widget: QWidget | None = None) -> QRect:
    """Taskbar/dock-aware available geometry for the widget's screen."""
    screen = None
    if widget is not None:
        try:
            win = widget.window()
            handle = win.windowHandle() if win is not None else None
            if handle is not None:
                screen = handle.screen()
        except Exception:
            screen = None
        if screen is None:
            try:
                screen = widget.screen()
            except Exception:
                screen = None
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    if screen is None:
        return QRect(0, 0, 1024, 768)
    return screen.availableGeometry()


def fit_dialog_to_available_screen(
    dialog: QDialog,
    *,
    margin: int = DEFAULT_MARGIN,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
    prefer_width: int | None = None,
    prefer_height: int | None = None,
) -> QRect:
    """Cap max size to available − margin; resize/center within that rect."""
    avail = available_screen_geometry(dialog)
    max_w = max(min_width, avail.width() - 2 * margin)
    max_h = max(min_height, avail.height() - 2 * margin)
    dialog.setMaximumSize(max_w, max_h)

    cur_min_w = dialog.minimumWidth()
    cur_min_h = dialog.minimumHeight()
    use_min_w = min(max(cur_min_w if cur_min_w > 0 else min_width, min_width), max_w)
    use_min_h = min(max(cur_min_h if cur_min_h > 0 else min_height, min_height), max_h)
    if cur_min_w > max_w:
        dialog.setMinimumWidth(max_w)
        use_min_w = max_w
    if cur_min_h > max_h:
        dialog.setMinimumHeight(max_h)
        use_min_h = max_h

    hint = dialog.sizeHint()
    w = prefer_width if prefer_width is not None else max(use_min_w, hint.width())
    h = prefer_height if prefer_height is not None else max(use_min_h, hint.height())
    w = min(max(w, use_min_w), max_w)
    h = min(max(h, use_min_h), max_h)
    dialog.resize(w, h)

    x = avail.x() + max(0, (avail.width() - w) // 2)
    y = avail.y() + max(0, (avail.height() - h) // 2)
    dialog.move(x, y)
    return avail


def make_content_scroll_area(content: QWidget) -> QScrollArea:
    """Vertical AsNeeded scroll; avoid horizontal scroll when possible."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    scroll.setWidget(content)
    return scroll


def apply_responsive_dialog(
    dialog: QDialog,
    *,
    margin: int = DEFAULT_MARGIN,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
    prefer_width: int | None = None,
    prefer_height: int | None = None,
    defer: bool = True,
) -> None:
    """Fit immediately; optionally again after show so the correct screen is used."""

    def _fit() -> None:
        fit_dialog_to_available_screen(
            dialog,
            margin=margin,
            min_width=min_width,
            min_height=min_height,
            prefer_width=prefer_width,
            prefer_height=prefer_height,
        )

    _fit()
    if defer:
        QTimer.singleShot(0, _fit)
