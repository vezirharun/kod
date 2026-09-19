"""Sabit konumlu hover önizleme — sağ panel üstü."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class FixedHoverPreviewPanel(QFrame):
    """Sonuç Detayı panelinin üstünde sabit büyük önizleme."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FixedHoverPreview")
        self.setFixedHeight(0)
        self.setVisible(False)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(175)
        self._hide_timer.timeout.connect(self._fade_out)
        self._opacity = 1.0
        self.setStyleSheet(
            "QFrame#FixedHoverPreview {"
            "background:#1c2129;border-bottom:1px solid #252b36;"
            "border-radius:0;"
            "}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self.lbl_caption = QLabel("")
        self.lbl_caption.setStyleSheet("color:#94a3b8;font-size:11px;")
        self.lbl_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumHeight(380)
        self.lbl_image.setMaximumHeight(480)
        self.lbl_image.setStyleSheet(
            "background:#232a35;border:1px solid #2a3340;border-radius:6px;color:#7c8698;"
        )
        layout.addWidget(self.lbl_caption)
        layout.addWidget(self.lbl_image, stretch=1)
        self._hover_path = ""
        self._hover_name = ""
        self._hover_token = 0
        from ui.thumbnail_scheduler import get_thumbnail_scheduler

        get_thumbnail_scheduler().thumbnail_ready.connect(self._on_thumb_ready)

    def _on_thumb_ready(self, file_id: int, image, size: int) -> None:
        if int(file_id) != int(self._hover_token):
            return
        if image is None:
            self.lbl_image.clear()
            self.lbl_image.setText("!")
            self.lbl_caption.setText(self._hover_name)
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            self.lbl_image.clear()
            self.lbl_image.setText("!")
            return
        target_width = min(620, max(480, self.width() - 20))
        scaled = pix.scaled(
            target_width,
            460,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_image.setPixmap(scaled)
        self.lbl_caption.setText(self._hover_name)
        if self.height() == 0:
            self.setFixedHeight(510)
        if not self.isVisible():
            self.setWindowOpacity(0.0)
            self.setVisible(True)
            self._animate_in()
        else:
            self.setWindowOpacity(1.0)
            self.show()

    def show_thumbnail(
        self,
        file_id: int,
        thumbnail_path: str,
        filename: str = "",
        *,
        source_path: str = "",
    ) -> None:
        """Kart ile aynı file_id / scheduler kaynağı."""
        self._hide_timer.stop()
        if not thumbnail_path and not source_path:
            return
        self._hover_path = thumbnail_path
        self._hover_name = filename or Path(thumbnail_path or source_path).name
        self._hover_token = int(file_id)
        from ui.thumbnail_scheduler import get_thumbnail_scheduler

        sched = get_thumbnail_scheduler()
        cached = sched.peek_image(self._hover_token)
        if cached is not None and not cached.isNull():
            pix = QPixmap.fromImage(cached)
            if not pix.isNull():
                target_width = min(620, max(480, self.width() - 20))
                scaled = pix.scaled(
                    target_width,
                    460,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
                self.lbl_image.setPixmap(scaled)
                self.lbl_image.setText("")
            else:
                self.lbl_image.clear()
                self.lbl_image.setText("···")
        else:
            self.lbl_image.clear()
            self.lbl_image.setText("···")
        self.lbl_caption.setText(self._hover_name)

        sched.request(
            self._hover_token,
            thumbnail_path,
            460,
            priority=0,
            source_path=source_path,
            filename=self._hover_name,
        )

    def schedule_hide(self) -> None:
        self._hide_timer.start()

    def cancel_hide(self) -> None:
        self._hide_timer.stop()

    def _animate_in(self) -> None:
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(160)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._fade_anim = anim

    def _fade_out(self) -> None:
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(140)
        anim.setStartValue(self.windowOpacity())
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InCubic)
        anim.finished.connect(self._on_hidden)
        anim.start()
        self._fade_anim = anim

    def _on_hidden(self) -> None:
        self.setVisible(False)
        self.setFixedHeight(0)
        self.lbl_image.clear()
        self.lbl_caption.clear()


class InspectorDockContent(QWidget):
    """Hover preview + inspector — tek sağ dock içeriği."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from ui.inspector_panel import InspectorPanel

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.hover_preview = FixedHoverPreviewPanel()
        self.inspector_panel = InspectorPanel()
        layout.addWidget(self.hover_preview)
        layout.addWidget(self.inspector_panel, stretch=1)
