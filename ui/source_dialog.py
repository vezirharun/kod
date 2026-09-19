"""Kaynak ekleme/düzenleme diyalogu."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

from core.sources import SOURCE_TYPE_LABELS, SourceType, detect_source_type


class SourceDialog(QDialog):
    def __init__(self, parent=None, source: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Kaynak Düzenle" if source else "Kaynak Ekle")
        self._source = source or {}
        self._build_ui()
        if source:
            self._load(source)
        from ui.responsive_dialog import apply_responsive_dialog

        apply_responsive_dialog(self, min_width=400, prefer_width=460)

    def _build_ui(self) -> None:
        layout = QFormLayout(self)

        self.txt_name = QLineEdit()
        layout.addRow("Ad:", self.txt_name)

        path_row = QHBoxLayout()
        self.txt_path = QLineEdit()
        btn_browse = QPushButton("Gözat…")
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self.txt_path)
        path_row.addWidget(btn_browse)
        layout.addRow("Kök yol:", path_row)

        self.cmb_type = QComboBox()
        for st in SourceType:
            self.cmb_type.addItem(SOURCE_TYPE_LABELS[st], st.value)
        layout.addRow("Kaynak türü:", self.cmb_type)

        self.chk_active = QCheckBox("Aktif")
        self.chk_active.setChecked(True)
        layout.addRow("", self.chk_active)

        self.spin_quick = QSpinBox()
        self.spin_quick.setRange(1, 168)
        self.spin_quick.setValue(6)
        self.spin_quick.setSuffix(" saat")
        layout.addRow("Hızlı tarama sıklığı:", self.spin_quick)

        self.spin_deep = QSpinBox()
        self.spin_deep.setRange(1, 90)
        self.spin_deep.setValue(7)
        self.spin_deep.setSuffix(" gün")
        layout.addRow("Derin kontrol sıklığı:", self.spin_deep)

        self.lbl_hint = QLabel(
            "Orijinal TIF dosyaları kopyalanmaz.\n"
            "Sadece merkezi cache'e küçük thumbnail ve embedding yazılır."
        )
        self.lbl_hint.setWordWrap(True)
        layout.addRow(self.lbl_hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Kaynak Kök Klasörü")
        if path:
            self.txt_path.setText(path)
            if not self.txt_name.text():
                import os

                self.txt_name.setText(os.path.basename(path.rstrip("\\/")) or path)
            idx = self.cmb_type.findData(detect_source_type(path))
            if idx >= 0:
                self.cmb_type.setCurrentIndex(idx)
            if not self._source:
                self.accept()

    def _load(self, source: dict) -> None:
        self.txt_name.setText(source.get("name", ""))
        self.txt_path.setText(source.get("root_path", ""))
        idx = self.cmb_type.findData(source.get("source_type", "local_pc"))
        if idx >= 0:
            self.cmb_type.setCurrentIndex(idx)
        self.chk_active.setChecked(bool(source.get("is_active", 1)))
        self.spin_quick.setValue(int(source.get("scan_interval_hours", 6)))
        self.spin_deep.setValue(int(source.get("deep_scan_interval_days", 7)))

    def get_data(self) -> dict:
        return {
            "id": self._source.get("id"),
            "name": self.txt_name.text().strip(),
            "root_path": self.txt_path.text().strip(),
            "source_type": self.cmb_type.currentData(),
            "is_active": 1 if self.chk_active.isChecked() else 0,
            "scan_interval_hours": self.spin_quick.value(),
            "deep_scan_interval_days": self.spin_deep.value(),
        }
