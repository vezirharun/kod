"""Alt durum çubuğu — FPS + kuyruk metrikleri."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget


class StatusBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet(
            "background: #15181e; border-top: 1px solid #2d323c; padding: 0 10px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        self.lbl_message = QLabel("Hazır")
        self.lbl_perf = QLabel("")
        self.lbl_perf.setStyleSheet("color:#64748b;font-size:11px;")
        self.lbl_index = QLabel("")
        self.lbl_index.setStyleSheet("color: #94a3b8;")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(120)
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        layout.addWidget(self.lbl_message, stretch=1)
        layout.addWidget(self.lbl_perf)
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_index)

    def set_message(self, text: str) -> None:
        self.lbl_message.setText(text)

    def set_perf_metrics(
        self,
        *,
        ui_fps: float = 60.0,
        search_queue: int = 0,
        preview_queue: int = 0,
        thumbnails_per_sec: float = 0.0,
    ) -> None:
        self.lbl_perf.setText(
            f"UI FPS: {ui_fps:.0f}  |  "
            f"Search Queue: {search_queue}  |  "
            f"Preview Queue: {preview_queue}  |  "
            f"Küçük görsel/sn: {thumbnails_per_sec:.0f}"
        )

    def set_index_hint(self, text: str) -> None:
        self.lbl_index.setText(text)

    def set_simple_summary(self, text: str) -> None:
        """Basit Mod — tek satır özet (DINO/FAISS yok)."""
        self.lbl_message.setText(text)
        self.lbl_index.setText("")

    def set_advanced_index_hint(self, text: str) -> None:
        """Gelişmiş Mod — teknik index satırı."""
        self.lbl_index.setText(text)

    def set_indexing_progress(self, percent: int, detail: str = "") -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(max(0, min(100, percent)))
        self.progress.setTextVisible(True)
        self.progress.setFormat(f"%p% {detail}".strip())

    def clear_indexing_progress(self) -> None:
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)

    def set_busy(self, busy: bool) -> None:
        if busy:
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.progress.setTextVisible(False)
        else:
            self.clear_indexing_progress()
