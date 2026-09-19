"""Arama özeti — okunabilir istatistik paneli."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from core.search_models import SearchStats


class SearchSummaryPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("searchSummary")
        self.setStyleSheet(
            "#searchSummary { background: #232830; border: 1px solid #2d3544; "
            "border-radius: 6px; padding: 4px; }"
        )
        grid = QGridLayout(self)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        self._labels: dict[str, QLabel] = {}
        keys = [
            ("total_indexed", "Toplam index"),
            ("selected_source_files", "Seçili kaynak"),
            ("supported_image_files", "Desteklenen görsel"),
            ("thumbnail_files", "Thumbnail"),
            ("feature_files", "Feature"),
            ("candidates_evaluated", "Skorlanan aday"),
            ("above_threshold", "Eşik üstü"),
            ("displayed", "Gösterilen"),
            ("near_below", "Eşik altı yakın"),
            ("search_ms", "Arama süresi"),
        ]
        for i, (key, title) in enumerate(keys):
            title_lbl = QLabel(f"<b>{title}:</b>")
            val_lbl = QLabel("—")
            val_lbl.setStyleSheet("color: #93c5fd;")
            self._labels[key] = val_lbl
            row, col = divmod(i, 5)
            grid.addWidget(title_lbl, row * 2, col)
            grid.addWidget(val_lbl, row * 2 + 1, col)
        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet(
            "color: #fbbf24; background: #2d2818; padding: 8px; border-radius: 4px; "
            "border: 1px solid #92400e;"
        )
        self.lbl_hint.setVisible(False)
        grid.addWidget(self.lbl_hint, 4, 0, 1, 5)
        self.clear()

    def clear(self) -> None:
        for key, lbl in self._labels.items():
            if key == "search_ms":
                lbl.setText("—")
            else:
                lbl.setText("0")
        self.set_threshold_hint("")

    def update_stats(
        self,
        stats: SearchStats | None,
        threshold: float,
        near_below_count: int = 0,
    ) -> None:
        if not stats:
            self.clear()
            return
        mapping = {
            "total_indexed": stats.total_indexed,
            "selected_source_files": stats.selected_source_files,
            "supported_image_files": stats.supported_image_files,
            "thumbnail_files": stats.thumbnail_files,
            "feature_files": stats.feature_files,
            "candidates_evaluated": stats.candidates_evaluated,
            "above_threshold": stats.above_threshold,
            "displayed": stats.displayed,
            "near_below": near_below_count,
            "search_ms": f"{stats.search_ms:.0f} ms (eşik %{int(threshold * 100)})",
        }
        for key, val in mapping.items():
            if key == "search_ms":
                self._labels[key].setText(str(val))
            else:
                self._labels[key].setText(f"{int(val):,}")

    def set_threshold_hint(self, text: str) -> None:
        self.lbl_hint.setText(text)
        self.lbl_hint.setVisible(bool(text))
