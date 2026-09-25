"""Profesyonel dashboard teması."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QScrollArea

# Shared surface tokens (keep dock/scroll fills aligned with APP_STYLESHEET).
COLOR_BG = "#1a1d23"
COLOR_SURFACE = "#1c2129"
COLOR_FG = "#e8eaed"
COLOR_MUTED = "#94a3b8"
COLOR_BORDER = "#252b36"

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_FG};
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}}
QDockWidget {{
    color: {COLOR_FG};
    background-color: {COLOR_BG};
    border: 1px solid {COLOR_BORDER};
}}
QDockWidget::title {{
    background: {COLOR_SURFACE};
    color: {COLOR_MUTED};
    padding: 6px 8px;
    border-bottom: 1px solid {COLOR_BORDER};
}}
QDockWidget > QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_FG};
}}
QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    background: {COLOR_SURFACE};
}}
QTabBar::tab {{
    background: #222730;
    color: #b0b8c4;
    padding: 7px 12px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{
    background: #2a3340;
    color: #ffffff;
    font-weight: bold;
}}
QGroupBox {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    margin-top: 8px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {COLOR_MUTED};
}}
QPushButton {{
    background-color: #2a3340;
    border: 1px solid #343d4c;
    border-radius: 4px;
    padding: 5px 11px;
    min-height: 26px;
}}
QPushButton:hover {{
    background-color: #343d4c;
}}
QPushButton:pressed {{
    background-color: #1f2630;
}}
QPushButton#primaryBtn {{
    background-color: #2563eb;
    border-color: #3b82f6;
    font-weight: bold;
}}
QPushButton#dangerBtn {{
    background-color: #7f1d1d;
    border-color: #991b1b;
}}
QLineEdit, QComboBox {{
    background: #222730;
    border: 1px solid #343d4c;
    border-radius: 4px;
    padding: 5px 8px;
    min-height: 26px;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: #343d4c;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -4px 0;
    background: #60a5fa;
    border-radius: 7px;
}}
QScrollArea {{
    border: none;
    background-color: {COLOR_BG};
}}
QScrollArea > QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_FG};
}}
QSplitter::handle {{
    background: {COLOR_BORDER};
    width: 3px;
    height: 3px;
}}
QProgressBar {{
    border: 1px solid #343d4c;
    border-radius: 4px;
    text-align: center;
    background: #222730;
}}
QProgressBar::chunk {{
    background: #2563eb;
    border-radius: 3px;
}}
QTableWidget {{
    gridline-color: {COLOR_BORDER};
    background: {COLOR_SURFACE};
    alternate-background-color: #20252e;
}}
QTreeWidget, QTreeView {{
    background: {COLOR_SURFACE};
    color: {COLOR_FG};
    border: 1px solid {COLOR_BORDER};
    outline: none;
}}
QTreeWidget::item, QTreeView::item {{
    color: {COLOR_FG};
    padding: 2px 4px;
}}
QTreeWidget::item:selected, QTreeView::item:selected {{
    background: #2a3340;
    color: #ffffff;
}}
QHeaderView::section {{
    background: #2a3340;
    padding: 6px;
    border: none;
}}
"""


def configure_dock_scroll_area(scroll: QScrollArea) -> None:
    """Opaque scroll/viewport fill so floating docks never show a blank white client."""
    bg = QColor(COLOR_BG)
    surface = QColor(COLOR_SURFACE)
    fg = QColor(COLOR_FG)

    scroll.setAutoFillBackground(True)
    spal = scroll.palette()
    spal.setColor(QPalette.ColorRole.Window, bg)
    spal.setColor(QPalette.ColorRole.Base, surface)
    spal.setColor(QPalette.ColorRole.WindowText, fg)
    spal.setColor(QPalette.ColorRole.Text, fg)
    scroll.setPalette(spal)

    vp = scroll.viewport()
    if vp is not None:
        vp.setAutoFillBackground(True)
        vpal = vp.palette()
        vpal.setColor(QPalette.ColorRole.Window, bg)
        vpal.setColor(QPalette.ColorRole.Base, surface)
        vpal.setColor(QPalette.ColorRole.WindowText, fg)
        vpal.setColor(QPalette.ColorRole.Text, fg)
        vp.setPalette(vpal)

    child = scroll.widget()
    if child is not None:
        child.setAutoFillBackground(True)
        cpal = child.palette()
        cpal.setColor(QPalette.ColorRole.Window, bg)
        cpal.setColor(QPalette.ColorRole.WindowText, fg)
        child.setPalette(cpal)

CLUSTER_BADGE_LABELS = {
    "exact_same": "Aynı",
    "exact": "Aynı",
    "format_resolution": "Format",
    "format_variant": "Format",
    "resolution_variant": "Çözünürlük",
    "crop_variant": "Varyant",
    "pattern_variant": "Varyant",
    "color_variant": "Renk",
    "same_family_close": "Benzer",
    "same_family_style": "Tarz",
    "related_family": "Yakın aile",
    "leopard_brown_similar": "Benzer",
    "leopard_other_color": "Renk",
    "related_animal_print": "Hayvan",
    "animal_print": "Animal print",
    "far_texture": "Uzak",
    "distant_texture": "Uzak",
    "textile_texture": "Metraj",
    "weak": "Zayıf",
    "unrelated": "Alakasız",
}
CLUSTER_BADGE_COLORS = {
    "exact_same": "#16a34a",
    "exact": "#16a34a",
    "format_resolution": "#22c55e",
    "format_variant": "#22c55e",
    "resolution_variant": "#22c55e",
    "crop_variant": "#84cc16",
    "pattern_variant": "#84cc16",
    "color_variant": "#ea580c",
    "same_family_close": "#ca8a04",
    "same_family_style": "#d97706",
    "related_family": "#f59e0b",
    "leopard_brown_similar": "#ca8a04",
    "leopard_other_color": "#ea580c",
    "related_animal_print": "#f59e0b",
    "animal_print": "#f59e0b",
    "far_texture": "#64748b",
    "distant_texture": "#64748b",
    "textile_texture": "#64748b",
    "style": "#64748b",
    "weak": "#475569",
    "unrelated": "#334155",
}


def score_color(percent: float) -> str:
    if percent >= 90:
        return "#22c55e"
    if percent >= 70:
        return "#84cc16"
    if percent >= 50:
        return "#f59e0b"
    return "#94a3b8"


def apply_theme(app: QApplication | None = None) -> None:
    target = app or QApplication.instance()
    if target:
        target.setStyleSheet(APP_STYLESHEET)
