"""Result Detail üst preview — liste hover aynı canvas'ta (overlay yok)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import configure_dock_scroll_area

# Full canvas contain-fit (liste hover + selection).
_CANVAS_FILL = 1.0


class FixedHoverPreviewPanel(QFrame):
    """Sağ dock üstündeki LARGE_PREVIEW_CANVAS.

    Liste hover → aynı canvas içinde görsel (geçici).
    Selection → kalıcı; leave sonrası restore.
    Ayrı floating overlay YOK.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FixedHoverPreview")
        self.setMinimumHeight(220)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMouseTracking(True)
        self._source_pix: QPixmap | None = None
        self._list_hide_timer = QTimer(self)
        self._list_hide_timer.setSingleShot(True)
        self._list_hide_timer.setInterval(175)
        self._list_hide_timer.timeout.connect(self._restore_selection)
        self.setStyleSheet(
            "QFrame#FixedHoverPreview {"
            "background:#1c2129;border-bottom:1px solid #252b36;"
            "border-radius:0;"
            "}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        self.lbl_caption = QLabel("Sonuç seçin…")
        self.lbl_caption.setStyleSheet("color:#94a3b8;font-size:11px;")
        self.lbl_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_caption.setWordWrap(True)
        self.lbl_caption.setMinimumWidth(0)
        self.lbl_image = QLabel("Önizleme")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(0, 160)
        self.lbl_image.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.lbl_image.setMouseTracking(True)
        self.lbl_image.setStyleSheet(
            "background:#232a35;border:1px solid #2a3340;border-radius:6px;color:#7c8698;"
        )
        layout.addWidget(self.lbl_caption)
        layout.addWidget(self.lbl_image, stretch=1)

        self._hover_path = ""
        self._hover_name = ""
        self._hover_token = 0
        self._selection_token = 0
        self._selection_name = ""
        self._selection_pix: QPixmap | None = None
        self._list_hovering = False

        from ui.thumbnail_scheduler import get_thumbnail_scheduler

        get_thumbnail_scheduler().thumbnail_ready.connect(self._on_thumb_ready)
        self.setVisible(True)

    # --- Compatibility aliases (older list-hover API) ---
    @property
    def _hide_timer(self) -> QTimer:
        return self._list_hide_timer

    def _avail_box(self) -> tuple[int, int]:
        """Usable size of LARGE_PREVIEW_CANVAS (lbl_image)."""
        w = max(1, int(self.lbl_image.width() or self.width() or 1) - 4)
        h = max(1, int(self.lbl_image.height() or self.height() or 1) - 4)
        p = self.parent()
        while p is not None:
            if isinstance(p, QScrollArea) and p.viewport() is not None:
                vw = int(p.viewport().width())
                if vw > 1:
                    w = min(w, max(1, vw - 24))
                break
            p = p.parent()
        return w, h

    def _apply_fit_image(
        self,
        pix: QPixmap,
        *,
        smooth: bool = True,
        fill: float | None = None,
    ) -> tuple[int, int]:
        """Maximize contain-fit into the full detail preview canvas (no crop, no file open)."""
        if pix.isNull():
            return (0, 0)
        self._source_pix = QPixmap(pix)
        nw, nh = int(pix.width()), int(pix.height())
        box_w, box_h = self._avail_box()
        f = _CANVAS_FILL if fill is None else float(fill)
        f = max(0.05, min(1.0, f))
        box_w = max(1, int(box_w * f))
        box_h = max(1, int(box_h * f))
        # Maximize inside canvas (allow upscale of small thumbs to fill red area).
        scale = min(box_w / max(nw, 1), box_h / max(nh, 1))
        tw = max(1, int(nw * scale))
        th = max(1, int(nh * scale))
        # Never exceed canvas.
        tw = min(tw, box_w)
        th = min(th, box_h)
        mode = (
            Qt.TransformationMode.SmoothTransformation
            if smooth
            else Qt.TransformationMode.FastTransformation
        )
        self.lbl_image.setText("")
        self.lbl_image.setPixmap(
            pix.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio, mode)
        )
        return tw, th

    def set_selection_pixmap(
        self,
        pix: QPixmap,
        *,
        file_id: int = 0,
        filename: str = "",
        tooltip: str = "",
        smooth: bool = True,
    ) -> tuple[int, int]:
        self._selection_token = int(file_id or self._selection_token or 0)
        if filename:
            self._selection_name = filename
        if pix.isNull():
            return (0, 0)
        self._selection_pix = QPixmap(pix)
        if not self._list_hovering:
            self.lbl_caption.setText(self._selection_name or "Önizleme")
            if tooltip:
                self.lbl_image.setToolTip(tooltip)
            return self._apply_fit_image(pix, smooth=smooth, fill=_CANVAS_FILL)
        return (0, 0)

    def clear_selection(self) -> None:
        self._selection_token = 0
        self._selection_name = ""
        self._selection_pix = None
        if not self._list_hovering:
            self.lbl_image.clear()
            self.lbl_image.setText("Önizleme")
            self.lbl_image.setToolTip("")
            self.lbl_caption.setText("Sonuç seçin…")

    def set_selection_pending(self, file_id: int, filename: str) -> None:
        self._selection_token = int(file_id)
        self._selection_name = filename or ""
        if not self._list_hovering:
            self.lbl_caption.setText(self._selection_name or "Yükleniyor…")
            self.lbl_image.setText("···")

    def set_selection_failed(self, file_id: int, filename: str, reason: str) -> None:
        if int(file_id) != int(self._selection_token):
            return
        if not self._list_hovering:
            self.lbl_image.clear()
            self.lbl_image.setText("!")
            self.lbl_image.setToolTip(reason)
            self.lbl_caption.setText(filename or self._selection_name)

    def _on_thumb_ready(self, file_id: int, image, size: int) -> None:
        if not self._list_hovering:
            return
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
        self._apply_fit_image(pix, smooth=True, fill=_CANVAS_FILL)
        self.lbl_caption.setText(self._hover_name)

    def show_thumbnail(
        self,
        file_id: int,
        thumbnail_path: str,
        filename: str = "",
        *,
        source_path: str = "",
    ) -> None:
        """Liste hover: hover_result → FULL detail canvas (selection değişmez)."""
        self._list_hide_timer.stop()
        if not thumbnail_path and not source_path:
            return
        self._list_hovering = True
        self._hover_path = thumbnail_path
        self._hover_name = filename or Path(thumbnail_path or source_path).name
        self._hover_token = int(file_id)
        from ui.thumbnail_scheduler import get_thumbnail_scheduler

        sched = get_thumbnail_scheduler()
        cached = sched.peek_image(self._hover_token)
        if cached is not None and not cached.isNull():
            pix = QPixmap.fromImage(cached)
            if not pix.isNull():
                self._apply_fit_image(pix, smooth=False, fill=_CANVAS_FILL)
                self.lbl_image.setText("")
            else:
                self.lbl_image.clear()
                self.lbl_image.setText("···")
        else:
            self.lbl_image.clear()
            self.lbl_image.setText("···")
        self.lbl_caption.setText(self._hover_name)
        # Request enough pixels for the dock canvas; no original AI/CDR/TIFF open.
        box_w, box_h = self._avail_box()
        req = max(460, int(max(box_w, box_h)))
        sched.request(
            self._hover_token,
            thumbnail_path,
            req,
            priority=0,
            source_path=source_path,
            filename=self._hover_name,
        )

    def schedule_hide(self) -> None:
        self._list_hide_timer.start()

    def cancel_hide(self) -> None:
        self._list_hide_timer.stop()

    def _restore_selection(self) -> None:
        self._list_hovering = False
        self._hover_token = 0
        if self._selection_pix is not None and not self._selection_pix.isNull():
            self.lbl_caption.setText(self._selection_name or "Önizleme")
            self._apply_fit_image(self._selection_pix, smooth=True, fill=_CANVAS_FILL)
        elif self._selection_token:
            self.lbl_caption.setText(self._selection_name or "Yükleniyor…")
            self.lbl_image.setText("···")
        else:
            self.clear_selection()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        pix = getattr(self, "_source_pix", None)
        if pix is not None and not pix.isNull():
            self._apply_fit_image(pix, smooth=True, fill=_CANVAS_FILL)


class InspectorDockContent(QWidget):
    """Üstte sabit ana preview + altta kaydırılabilir Sonuç Detayı."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from ui.inspector_panel import InspectorPanel

        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.hover_preview = FixedHoverPreviewPanel()
        self.inspector_panel = InspectorPanel()
        self.inspector_panel.set_preview_host(self.hover_preview)

        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.detail_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        configure_dock_scroll_area(self.detail_scroll)
        self.detail_scroll.setWidget(self.inspector_panel)

        layout.addWidget(self.hover_preview, stretch=2)
        layout.addWidget(self.detail_scroll, stretch=3)
