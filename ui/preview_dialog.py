"""Önizleme ve karşılaştırma diyalogu."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout

from core.search_engine import SearchResult


class ImagePreviewDialog(QDialog):
    """Tek görsel büyük önizleme. Oran korunur; EPS/TIFF için cache/preview yolu kullanılır."""

    def __init__(self, path: str, title: str = "Önizleme", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title or "Önizleme")
        self.resize(960, 720)
        self._path = str(path or "")
        self._pix = QPixmap()
        layout = QVBoxLayout(self)
        hint = QLabel("Kapatmak için Esc veya pencereyi kapatın.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Responsive floor (small screens); preferred dialog size is still ~960×720.
        self.lbl_image.setMinimumSize(320, 240)
        self.lbl_image.setStyleSheet("background: #1e1e1e; border: 1px solid #444;")
        layout.addWidget(self.lbl_image, 1)
        self._load()
        from ui.responsive_dialog import apply_responsive_dialog

        apply_responsive_dialog(
            self, min_width=400, min_height=320, prefer_width=960, prefer_height=720
        )

    def _load(self) -> None:
        if self._path:
            self._pix = QPixmap(self._path)
        if self._pix.isNull():
            self.lbl_image.setText("Önizleme yüklenemedi")
            return
        self._rescale()

    def _rescale(self) -> None:
        if self._pix.isNull():
            return
        target = self.lbl_image.size()
        w = max(400, target.width())
        h = max(400, target.height())
        self.lbl_image.setPixmap(
            self._pix.scaled(
                w,
                h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()


class PreviewDialog(QDialog):
    def __init__(
        self,
        query_path: str | None,
        result: SearchResult,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Önizleme — %{result.score_percent} benzerlik")
        self.resize(900, 500)
        self._build_ui(query_path, result)
        from ui.responsive_dialog import apply_responsive_dialog

        apply_responsive_dialog(
            self, min_width=480, min_height=320, prefer_width=900, prefer_height=500
        )

    def _build_ui(self, query_path: str | None, result: SearchResult) -> None:
        layout = QVBoxLayout(self)

        compare = QHBoxLayout()

        if query_path:
            q_box = QVBoxLayout()
            q_box.addWidget(QLabel("Sorgu Görseli"))
            q_img = self._make_image_label(query_path)
            q_box.addWidget(q_img)
            compare.addLayout(q_box)

        r_box = QVBoxLayout()
        r_box.addWidget(QLabel(f"Sonuç — {result.filename}"))
        r_img = self._make_image_label(result.thumbnail_path or result.path)
        r_box.addWidget(r_img)
        compare.addLayout(r_box)

        layout.addLayout(compare)

        info = QLabel(
            f"Müşteri: {result.customer}\n"
            f"Yol: {result.path}\n"
            f"Benzerlik: %{result.score_percent}\n"
            f"Skor detayı: {result.breakdown}"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

    @staticmethod
    def _make_image_label(path: str) -> QLabel:
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumSize(400, 400)
        lbl.setStyleSheet("background: #1e1e1e; border: 1px solid #444;")
        if path:
            pix = QPixmap(path)
            if not pix.isNull():
                lbl.setPixmap(
                    pix.scaled(
                        400,
                        400,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                lbl.setText("Önizleme yüklenemedi")
        return lbl
