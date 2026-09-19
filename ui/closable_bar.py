"""Kapatılabilir panel başlığı — X ile gizle, menüden tekrar aç."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class ClosableBar(QWidget):
    closed = Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ClosableBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 6, 4)
        lay.setSpacing(6)
        self.lbl_title = QLabel(f"<b>{title}</b>")
        self.lbl_title.setStyleSheet("color:#e2e8f0;")
        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedSize(26, 26)
        self.btn_close.setToolTip("Paneli kapat (Paneller menüsünden tekrar açın)")
        self.btn_close.setStyleSheet(
            "QPushButton { background:#2d3544; border:1px solid #3d4654; border-radius:4px; }"
            "QPushButton:hover { background:#3d4654; }"
        )
        self.btn_close.clicked.connect(self.closed.emit)
        lay.addWidget(self.lbl_title, stretch=1)
        lay.addWidget(self.btn_close)
