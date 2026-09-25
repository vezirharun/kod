"""Result Detail üst preview + ikinci seviye hover magnify overlay."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import COLOR_BG, COLOR_BORDER, COLOR_SURFACE, configure_dock_scroll_area

# Dock canvas contain-fit (liste hover + selection). Overlay ayrı katman.
_CANVAS_FILL = 1.0


def _find_main_window(widget: QWidget | None) -> QMainWindow | None:
    w = widget
    while w is not None:
        if isinstance(w, QMainWindow):
            return w
        w = w.parentWidget()
    return None


class DetailPreviewHoverOverlay(QFrame):
    """Ana pencere floating magnify — normal preview geometry'sine dokunmaz."""

    def __init__(self, host: "FixedHoverPreviewPanel", parent: QWidget):
        super().__init__(parent)
        self._host = host
        self.setObjectName("DetailPreviewHoverOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.hide()
        self.setStyleSheet(
            f"QFrame#DetailPreviewHoverOverlay {{"
            f"background:{COLOR_SURFACE};"
            f"border:1px solid {COLOR_BORDER};"
            f"border-radius:6px;"
            f"}}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(0, 0)
        self.lbl_image.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.lbl_image.setStyleSheet(f"background:{COLOR_BG};border-radius:4px;")
        self.lbl_image.setMouseTracking(True)
        lay.addWidget(self.lbl_image)

    def enterEvent(self, event) -> None:  # noqa: N802
        super().enterEvent(event)
        self._host._overlay_zone_enter()

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        self._host._overlay_zone_leave()

    def set_pixmap_contain(self, pix: QPixmap, box: QRect) -> tuple[int, int]:
        """Contain-fit into max box; return display size (no file open)."""
        if pix.isNull() or box.width() < 32 or box.height() < 32:
            self.hide()
            return (0, 0)
        pad = 16
        aw = max(1, box.width() - pad)
        ah = max(1, box.height() - pad)
        nw, nh = int(pix.width()), int(pix.height())
        scale = min(aw / max(nw, 1), ah / max(nh, 1))
        tw = max(1, min(aw, int(nw * scale)))
        th = max(1, min(ah, int(nh * scale)))
        self.lbl_image.setPixmap(
            pix.scaled(
                tw,
                th,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.setFixedSize(tw + 16, th + 16)
        return tw, th


class FixedHoverPreviewPanel(QFrame):
    """Sağ dock LARGE_PREVIEW_CANVAS + detail-hover magnify overlay.

    STATE 1: selection/normal canvas
    STATE 2: liste hover → aynı canvas (selection değişmez)
    STATE 3: canvas hover → ayrı DetailPreviewHoverOverlay (canvas geometry sabit)
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
        self._overlay_hide_timer = QTimer(self)
        self._overlay_hide_timer.setSingleShot(True)
        self._overlay_hide_timer.setInterval(120)
        self._overlay_hide_timer.timeout.connect(self._hide_overlay)
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
        self._overlay: DetailPreviewHoverOverlay | None = None
        self._overlay_visible = False

        from ui.thumbnail_scheduler import get_thumbnail_scheduler

        get_thumbnail_scheduler().thumbnail_ready.connect(self._on_thumb_ready)
        self.setVisible(True)

    @property
    def _hide_timer(self) -> QTimer:
        return self._list_hide_timer

    def _avail_box(self) -> tuple[int, int]:
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
        """Maximize contain-fit into dock canvas only (never opens files)."""
        if pix.isNull():
            return (0, 0)
        self._source_pix = QPixmap(pix)
        nw, nh = int(pix.width()), int(pix.height())
        box_w, box_h = self._avail_box()
        f = _CANVAS_FILL if fill is None else float(fill)
        f = max(0.05, min(1.0, f))
        box_w = max(1, int(box_w * f))
        box_h = max(1, int(box_h * f))
        scale = min(box_w / max(nw, 1), box_h / max(nh, 1))
        tw = max(1, min(box_w, int(nw * scale)))
        th = max(1, min(box_h, int(nh * scale)))
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
            result = self._apply_fit_image(pix, smooth=smooth, fill=_CANVAS_FILL)
            if self._overlay_visible:
                self._show_overlay()
            return result
        return (0, 0)

    def clear_selection(self) -> None:
        self._selection_token = 0
        self._selection_name = ""
        self._selection_pix = None
        self._hide_overlay(immediate=True)
        if not self._list_hovering:
            self.lbl_image.clear()
            self.lbl_image.setText("Önizleme")
            self.lbl_image.setToolTip("")
            self.lbl_caption.setText("Sonuç seçin…")

    def set_selection_pending(self, file_id: int, filename: str) -> None:
        self._selection_token = int(file_id)
        self._selection_name = filename or ""
        self._hide_overlay(immediate=True)
        if not self._list_hovering:
            self.lbl_caption.setText(self._selection_name or "Yükleniyor…")
            self.lbl_image.setText("···")

    def set_selection_failed(self, file_id: int, filename: str, reason: str) -> None:
        if int(file_id) != int(self._selection_token):
            return
        self._hide_overlay(immediate=True)
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
        if self._overlay_visible:
            self._show_overlay()

    def show_thumbnail(
        self,
        file_id: int,
        thumbnail_path: str,
        filename: str = "",
        *,
        source_path: str = "",
    ) -> None:
        """STATE 2 — liste hover → full detail canvas (selection değişmez)."""
        self._list_hide_timer.stop()
        self._hide_overlay(immediate=True)
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

    def _overlay_source_pix(self) -> QPixmap | None:
        """Currently displayed preview (list hover or selection)."""
        pix = self._source_pix
        if pix is not None and not pix.isNull():
            return pix
        if self._selection_pix is not None and not self._selection_pix.isNull():
            return self._selection_pix
        return None

    def _ensure_overlay(self) -> DetailPreviewHoverOverlay | None:
        if self._overlay is not None:
            return self._overlay
        mw = _find_main_window(self)
        if mw is None:
            return None
        self._overlay = DetailPreviewHoverOverlay(self, mw)
        return self._overlay

    def _overlay_anchor_and_box(self) -> tuple[QPoint, QRect] | None:
        mw = _find_main_window(self)
        if mw is None:
            return None
        client = mw.rect()
        central = mw.centralWidget()
        if central is not None:
            top_left = central.mapTo(mw, QPoint(0, 0))
            client = QRect(top_left, central.size())
        anchor = self.mapTo(mw, QPoint(0, 0))
        return anchor, client

    def _compute_overlay_geometry(self, fw: int, fh: int) -> QRect | None:
        placed = self._overlay_anchor_and_box()
        if placed is None:
            return None
        anchor, client = placed
        gap = 8
        # Prefer LEFT of normal preview (over results); never grow the dock.
        x = anchor.x() - fw - gap
        y = anchor.y()
        if x < client.left() + gap:
            x = client.left() + gap
        if y < client.top() + gap:
            y = client.top() + gap
        if x + fw > client.right() - gap:
            x = max(client.left() + gap, client.right() - gap - fw)
        if y + fh > client.bottom() - gap:
            y = max(client.top() + gap, client.bottom() - gap - fh)
        x = min(max(x, client.left()), max(client.left(), client.right() - fw))
        y = min(max(y, client.top()), max(client.top(), client.bottom() - fh))
        return QRect(x, y, fw, fh)

    def _show_overlay(self) -> None:
        """STATE 3 — large overlay only; normal canvas geometry untouched."""
        pix = self._overlay_source_pix()
        if pix is None or pix.isNull():
            return
        overlay = self._ensure_overlay()
        if overlay is None:
            return
        placed = self._overlay_anchor_and_box()
        if placed is None:
            return
        anchor, client = placed
        gap = 8
        space_left = max(0, anchor.x() - client.left() - gap * 2)
        # Prefer space left of preview; if dock-only / cramped, use most of main window.
        max_w = max(space_left, int(client.width() * 0.55), 240)
        max_h = max(160, int(client.height() * 0.70))
        max_w = min(max_w, max(160, client.width() - 16))
        max_h = min(max_h, max(160, client.height() - 16))
        tw, th = overlay.set_pixmap_contain(pix, QRect(0, 0, max_w, max_h))
        if tw <= 0:
            return
        fw, fh = overlay.width(), overlay.height()
        geo = self._compute_overlay_geometry(fw, fh)
        if geo is None:
            return
        overlay.setGeometry(geo)
        overlay.raise_()
        overlay.show()
        self._overlay_visible = True

    def _hide_overlay(self, *, immediate: bool = False) -> None:
        self._overlay_hide_timer.stop()
        if self._overlay is not None:
            self._overlay.hide()
        self._overlay_visible = False

    def _overlay_zone_enter(self) -> None:
        self._overlay_hide_timer.stop()

    def _overlay_zone_leave(self) -> None:
        self._overlay_hide_timer.start()

    def enterEvent(self, event) -> None:  # noqa: N802
        super().enterEvent(event)
        self._overlay_hide_timer.stop()
        # Do NOT resize/refit normal canvas here.
        self._show_overlay()

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        # Delay: mouse may move onto large overlay (hover zone).
        self._overlay_hide_timer.start()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        pix = getattr(self, "_source_pix", None)
        if pix is not None and not pix.isNull():
            self._apply_fit_image(pix, smooth=True, fill=_CANVAS_FILL)
        if self._overlay_visible:
            self._show_overlay()


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
