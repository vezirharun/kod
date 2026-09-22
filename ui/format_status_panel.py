"""Format durumu paneli — format + bağımlılık denetimi."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.capability_check import probe_format_dependencies
from core.format_audit import audit_format_counts


class FormatStatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._db_path = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        grp = QGroupBox("Format Durumu")
        layout.addWidget(grp)
        inner = QVBoxLayout(grp)
        row = QHBoxLayout()
        self.lbl_summary = QLabel("Kaynak seçin veya yenileyin.")
        row.addWidget(self.lbl_summary, stretch=1)
        self.btn_refresh = QPushButton("Yenile")
        self.btn_refresh.clicked.connect(self.refresh)
        row.addWidget(self.btn_refresh)
        inner.addLayout(row)

        inner.addWidget(QLabel("<b>Format desteği</b>"))
        self.tbl_formats = QTableWidget(0, 2)
        self.tbl_formats.setHorizontalHeaderLabels(["Format / Araç", "Durum"])
        self.tbl_formats.horizontalHeader().setStretchLastSection(True)
        inner.addWidget(self.tbl_formats)

        inner.addWidget(QLabel("<b>Arşiv istatistikleri</b>"))
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Format",
                "Bulunan",
                "Önizleme hazır",
                "Küçük görsel",
                "Aranabilir",
                "Eksik dosya",
                "Vektör",
                "Yapay zekâ analizi",
                "Desteklenmiyor",
                "Bağımlılık eksik",
                "Hatalı",
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        inner.addWidget(self.table)

    def set_db_path(self, db_path: str) -> None:
        new = str(db_path or "")
        if new == self._db_path and self.table.rowCount() > 0:
            return
        self._db_path = new
        if new:
            self.refresh()

    def refresh(self) -> None:
        self._refresh_dependencies()
        if not self._db_path:
            return
        rows = audit_format_counts(self._db_path)
        self.table.setRowCount(len(rows))
        total_found = sum(r["found"] for r in rows)
        total_search = sum(r["searchable"] for r in rows)
        total_missing = sum(int(r.get("missing") or 0) for r in rows)
        self.lbl_summary.setText(
            f"Toplam: {total_found:,} | Aranabilir: {total_search:,} | Eksik: {total_missing:,}"
        )
        for i, r in enumerate(rows):
            for j, key in enumerate(
                (
                    "format",
                    "found",
                    "preview_ok",
                    "thumbnail_ok",
                    "searchable",
                    "missing",
                    "embedding_ok",
                    "ai_analyzed",
                    "unsupported",
                    "dependency_missing",
                    "failed",
                )
            ):
                self.table.setItem(i, j, QTableWidgetItem(str(r.get(key, 0))))

    def _refresh_dependencies(self) -> None:
        deps = probe_format_dependencies()
        rows = [
            ("TIF / TIFF", deps.get("tif")),
            ("PSD", deps.get("psd")),
            ("AI / EPS", deps.get("ai_eps")),
            ("PDF", deps.get("pdf")),
            ("SVG", deps.get("svg")),
            ("CDR", deps.get("cdr")),
            ("DXF", deps.get("dxf")),
            ("DWG (ODA)", deps.get("dwg")),
            ("PLT / HPGL", deps.get("plt")),
            ("Embroidery (pyembroidery)", deps.get("embroidery")),
            ("Ghostscript", deps.get("ghostscript")),
            ("Poppler (pdftoppm)", deps.get("poppler")),
            ("psd-tools", deps.get("psd_tools")),
            ("PyMuPDF", deps.get("pymupdf")),
            ("pyvips", deps.get("pyvips")),
        ]
        self.tbl_formats.setRowCount(len(rows))
        for i, (label, ok) in enumerate(rows):
            self.tbl_formats.setItem(i, 0, QTableWidgetItem(label))
            status = QTableWidgetItem("Hazır" if ok else "Eksik")
            if not ok:
                status.setForeground(Qt.GlobalColor.red)
            else:
                status.setForeground(Qt.GlobalColor.green)
            self.tbl_formats.setItem(i, 1, status)
