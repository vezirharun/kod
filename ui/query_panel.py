"""Sorgu görseli önizleme ve crop seçimi."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

MIN_CROP_PX = 24


class CropImageLabel(QLabel):
    """Mouse ile dikdörtgen alan seçimi yapılabilen görsel preview."""

    crop_changed = Signal(object)  # tuple[int, int, int, int] | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background: #1e1e1e; border: 1px solid #3d4654; border-radius: 4px;"
        )
        self._pixmap: QPixmap | None = None
        self._image_path: str = ""
        self._origin = QPoint()
        self._current = QPoint()
        self._selecting = False
        self._crop_rect: QRect | None = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self.setText("Görsel yükleyin veya sürükleyip bırakın")

    def set_image(self, path: str | None) -> None:
        self._image_path = path or ""
        self._crop_rect = None
        if not path:
            self._pixmap = None
            self.setText("Görsel yükleyin veya sürükleyip bırakın")
            self.update()
            return
        from core.preview_renderer import load_display_qpixmap

        pix, err = load_display_qpixmap(path)
        if pix.isNull():
            self._pixmap = None
            tip = err or "önizleme üretilemedi"
            self.setText(f"Yüklenemedi: {Path(path).name}\n{tip}")
            self.update()
            return
        self._pixmap = pix
        self.setText("")
        self._fit_image()
        self.update()

    def _fit_image(self) -> None:
        if not self._pixmap:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        lw, lh = max(1, self.width()), max(1, self.height())
        self._scale = min(lw / pw, lh / ph, 1.0)
        disp_w, disp_h = int(pw * self._scale), int(ph * self._scale)
        self._offset = QPoint((lw - disp_w) // 2, (lh - disp_h) // 2)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_image()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._pixmap:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        pw, ph = self._pixmap.width(), self._pixmap.height()
        disp = QRect(
            self._offset.x(),
            self._offset.y(),
            int(pw * self._scale),
            int(ph * self._scale),
        )
        painter.drawPixmap(disp, self._pixmap)
        if self._crop_rect and not self._crop_rect.isNull():
            painter.fillRect(self._crop_rect, QColor(0, 200, 255, 45))
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2, Qt.PenStyle.SolidLine))
            painter.drawRect(self._crop_rect)

    def _widget_to_image_rect(self, rect: QRect) -> tuple[int, int, int, int] | None:
        if not self._pixmap or rect.isNull() or self._scale <= 0:
            return None
        x1 = (rect.left() - self._offset.x()) / self._scale
        y1 = (rect.top() - self._offset.y()) / self._scale
        x2 = (rect.right() - self._offset.x()) / self._scale
        y2 = (rect.bottom() - self._offset.y()) / self._scale
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ix = max(0, int(min(x1, x2)))
        iy = max(0, int(min(y1, y2)))
        iw = min(pw - ix, int(abs(x2 - x1)))
        ih = min(ph - iy, int(abs(y2 - y1)))
        if iw < MIN_CROP_PX or ih < MIN_CROP_PX:
            return None
        return ix, iy, iw, ih

    def image_crop_rect(self) -> tuple[int, int, int, int] | None:
        if not self._crop_rect:
            return None
        return self._widget_to_image_rect(self._crop_rect.normalized())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap:
            self._selecting = True
            self._origin = event.position().toPoint()
            self._current = self._origin
            self._crop_rect = QRect(self._origin, self._current)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._selecting:
            self._current = event.position().toPoint()
            self._crop_rect = QRect(self._origin, self._current).normalized()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._selecting:
            self._selecting = False
            self._current = event.position().toPoint()
            self._crop_rect = QRect(self._origin, self._current).normalized()
            img_rect = self.image_crop_rect()
            if img_rect is None and self._crop_rect.width() > 0:
                QMessageBox.warning(
                    self, "Uyarı", f"Seçili alan çok küçük (min {MIN_CROP_PX}px)."
                )
                self._crop_rect = None
            self.crop_changed.emit(img_rect)
            self.update()
        super().mouseReleaseEvent(event)

    def reset_crop(self) -> None:
        self._crop_rect = None
        self.crop_changed.emit(None)
        self.update()

    @property
    def image_path(self) -> str:
        return self._image_path


class QueryPanel(QWidget):
    search_full = Signal()
    search_crop = Signal()
    change_image_clicked = Signal()
    clear_image_clicked = Signal()
    image_selected = Signal(str)
    crop_region_selected = Signal()
    crop_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("QueryPanel")
        self._build_ui()
        self.crop_label.crop_changed.connect(self._on_crop_changed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.crop_label = CropImageLabel()
        layout.addWidget(self.crop_label, stretch=1)

        self._thumb_host = QWidget()
        self._thumb_row = QHBoxLayout(self._thumb_host)
        self._thumb_row.setContentsMargins(0, 0, 0, 0)
        self._thumb_row.setSpacing(4)
        self._thumb_scroll = QScrollArea()
        self._thumb_scroll.setWidgetResizable(True)
        self._thumb_scroll.setFixedHeight(76)
        self._thumb_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._thumb_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._thumb_scroll.setWidget(self._thumb_host)
        self._thumb_scroll.setVisible(False)
        layout.addWidget(self._thumb_scroll)

        info_box = QFrame()
        info_box.setObjectName("queryInfoBox")
        info_box.setStyleSheet(
            "QFrame#queryInfoBox { background:#20242c; border:1px solid #2d3544; border-radius:4px; }"
        )
        info_l = QVBoxLayout(info_box)
        info_l.setContentsMargins(10, 6, 10, 6)
        info_l.setSpacing(4)
        self.lbl_meta = QLabel("—")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        self.lbl_mode = QLabel("")
        self.lbl_mode.setWordWrap(True)
        self.lbl_mode.setStyleSheet("color: #8bc34a; font-size: 12px; font-weight:600;")
        info_l.addWidget(self.lbl_meta)
        info_l.addWidget(self.lbl_mode)
        layout.addWidget(info_box)

        btn_grid = QGridLayout()
        btn_grid.setHorizontalSpacing(6)
        btn_grid.setVerticalSpacing(6)
        self.btn_change = QPushButton("Görseli Değiştir")
        self.btn_clear_image = QPushButton("Görseli Kaldır")
        self.btn_full = QPushButton("Görsel Ara")
        self.btn_crop = QPushButton("Alan Seç")
        self.btn_reset = QPushButton("Alanı Sıfırla")
        for btn, tip in (
            (self.btn_change, "Başka bir görsel seç"),
            (self.btn_clear_image, "Yüklenen sorgu görselini kaldır"),
            (self.btn_full, "Tüm görselle ara. Birden fazla görsel varsa hepsini sırayla arar."),
            (self.btn_crop, "Seçili alanla ara"),
            (self.btn_reset, "Seçim alanını temizle"),
        ):
            btn.setMinimumHeight(32)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setToolTip(tip)
        self.btn_change.clicked.connect(self.change_image_clicked.emit)
        self.btn_clear_image.clicked.connect(self.clear_image_clicked.emit)
        self.btn_full.clicked.connect(self.search_full.emit)
        self.btn_crop.clicked.connect(self.search_crop.emit)
        self.btn_reset.clicked.connect(self.crop_label.reset_crop)
        btn_grid.addWidget(self.btn_change, 0, 0)
        btn_grid.addWidget(self.btn_clear_image, 0, 1)
        btn_grid.addWidget(self.btn_full, 1, 0)
        btn_grid.addWidget(self.btn_crop, 1, 1)
        btn_grid.addWidget(self.btn_reset, 2, 0, 1, 2)
        layout.addLayout(btn_grid)

    def set_query_thumbs(self, paths: list[str]) -> None:
        """Çoklu bırakılan sorgu görsellerinin hepsini şeritte gösterir. Aramayı değiştirmez."""
        while self._thumb_row.count():
            item = self._thumb_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        clean = [str(p) for p in (paths or []) if str(p or "").strip()]
        if len(clean) <= 1:
            self._thumb_scroll.setVisible(False)
            return
        from core.preview_renderer import load_display_qpixmap

        for path in clean:
            btn = QToolButton()
            btn.setFixedSize(64, 64)
            btn.setAutoRaise(True)
            btn.setToolTip(Path(path).name)
            pix, _err = load_display_qpixmap(path)
            if pix is not None and not pix.isNull():
                btn.setIcon(
                    QIcon(
                        pix.scaled(
                            56,
                            56,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                )
                btn.setIconSize(QSize(56, 56))
            else:
                btn.setText(Path(path).stem[:6] or "?")
            btn.clicked.connect(lambda _checked=False, p=path: self.set_image(p))
            self._thumb_row.addWidget(btn)
        self._thumb_row.addStretch()
        self._thumb_scroll.setVisible(True)

    def set_image(self, path: str | None) -> None:
        if not path:
            self.set_query_thumbs([])
        self.crop_label.set_image(path)
        if path and Path(path).exists():
            p = Path(path)
            from core.preview_renderer import load_display_qpixmap

            pix, _err = load_display_qpixmap(path)
            if pix.isNull():
                w, h = 0, 0
            else:
                w, h = pix.width(), pix.height()
            self.lbl_meta.setText(
                f"<b>{p.name}</b><br>{w}×{h} | {p.suffix.upper().lstrip('.') or '?'}"
            )
            self.image_selected.emit(path)
        else:
            self.lbl_meta.setText("—")
        self.lbl_mode.setText("")

    def set_load_status(self, text: str) -> None:
        self.lbl_mode.setText(text)
        if text:
            self.lbl_mode.setStyleSheet(
                "color: #8bc34a; font-size: 12px; font-weight:600;"
            )

    def set_search_mode_label(self, text: str) -> None:
        self.lbl_mode.setText(text)

    def crop_rect(self) -> tuple[int, int, int, int] | None:
        return self.crop_label.image_crop_rect()

    def image_path(self) -> str:
        return self.crop_label.image_path

    def _on_crop_changed(self, rect: tuple[int, int, int, int] | None) -> None:
        if rect:
            x, y, w, h = rect
            self.lbl_mode.setText("Seçili alanla aranıyor")
            self.lbl_mode.setStyleSheet(
                "color: #fbbf24; font-size: 12px; font-weight:600;"
            )
            self.crop_region_selected.emit()
        else:
            self.lbl_mode.setText("")
            self.lbl_mode.setStyleSheet(
                "color: #8bc34a; font-size: 12px; font-weight:600;"
            )
            self.crop_cleared.emit()
