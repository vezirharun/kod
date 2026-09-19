"""Seçilen desen paneli — sorgu görselini sonuçlardan ayrı tutar."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QSizePolicy, QVBoxLayout, QWidget

from ui.closable_bar import ClosableBar
from ui.query_panel import QueryPanel


class QueryImagePanel(QWidget):
    """Seçilen Desen — query görseli/crop ve arama butonları ayrı panelde."""

    close_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("QueryImagePanel")
        self.setMinimumHeight(360)
        self.setMinimumWidth(260)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)

        self._title_bar = ClosableBar("Seçilen Desen")
        self._title_bar.closed.connect(self.close_requested.emit)
        layout.addWidget(self._title_bar)

        group = QGroupBox()
        group.setObjectName("QueryImageGroup")
        group.setFlat(True)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(group, stretch=1)

        g_l = QVBoxLayout(group)
        g_l.setContentsMargins(10, 14, 10, 10)
        g_l.setSpacing(8)
        self.query_panel = QueryPanel()
        g_l.addWidget(self.query_panel, stretch=1)

    def setAcceptDrops(self, on: bool) -> None:
        super().setAcceptDrops(on)
        self.query_panel.setAcceptDrops(on)
