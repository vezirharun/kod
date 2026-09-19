"""İki satırlı üst arama çubuğu."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.settings import AppSettings
from core.sources import SEARCH_SCOPE_LABELS


class SearchToolbar(QWidget):
    search_clicked = Signal()
    pick_image_clicked = Signal()
    quick_folder_clicked = Signal()
    clear_clicked = Signal()
    cancel_search_clicked = Signal()
    filter_changed = Signal()
    filter_research = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 4)
        root.setSpacing(6)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText(
            "Metin: leopard, animal print, çiçek desen… (görselle birlikte kullanılabilir)"
        )
        self.txt_search.setMinimumWidth(200)
        self.btn_search = QPushButton("Ara")
        self.btn_search.setObjectName("primaryBtn")
        self.btn_search.setMinimumWidth(72)
        self.btn_pick_image = QPushButton("Görsel Seç")
        self.btn_pick_image.setMinimumWidth(100)
        self.btn_quick_folder = QPushButton("Bu Klasörde Hızlı Ara")
        self.btn_quick_folder.setMinimumWidth(150)
        self.btn_clear = QPushButton("Aramayı Temizle")
        self.btn_clear.setMinimumWidth(120)
        self.btn_cancel = QPushButton("İptal")
        self.btn_cancel.setObjectName("dangerBtn")
        self.btn_cancel.setMinimumWidth(64)
        self.btn_cancel.setVisible(False)
        row1.addWidget(self.txt_search, stretch=1)
        row1.addWidget(self.btn_search)
        row1.addWidget(self.btn_pick_image)
        row1.addWidget(self.btn_quick_folder)
        row1.addWidget(self.btn_clear)
        row1.addWidget(self.btn_cancel)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(QLabel("Eşik"))
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(30, 100)
        self.slider_threshold.setValue(int(self._settings.similarity_threshold * 100))
        self.slider_threshold.setMinimumWidth(120)
        self.lbl_threshold = QLabel(f"%{self.slider_threshold.value()}")
        self.lbl_threshold.setMinimumWidth(36)
        self.lbl_threshold_count = QLabel("")
        row2.addWidget(self.slider_threshold, stretch=1)
        row2.addWidget(self.lbl_threshold)
        row2.addWidget(self.lbl_threshold_count)

        row2.addWidget(QLabel("Sonuç"))
        self.cmb_limit = QComboBox()
        self.cmb_limit.setMinimumWidth(90)
        self.cmb_limit.addItem("Sınırsız", 0)
        row2.addWidget(self.cmb_limit)

        row2.addWidget(QLabel("Kapsam"))
        self.cmb_scope = QComboBox()
        self.cmb_scope.setMinimumWidth(140)
        for scope, label in SEARCH_SCOPE_LABELS.items():
            self.cmb_scope.addItem(label, scope.value)
        idx = self.cmb_scope.findData(self._settings.search_scope)
        if idx >= 0:
            self.cmb_scope.setCurrentIndex(idx)
        row2.addWidget(self.cmb_scope)

        row2.addWidget(QLabel("Müşteri"))
        self.cmb_customer = QComboBox()
        self.cmb_customer.setMinimumWidth(90)
        self.cmb_customer.addItem("Tümü", "")
        row2.addWidget(self.cmb_customer)

        row2.addWidget(QLabel("Mod"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.setMinimumWidth(130)
        self.cmb_mode.addItem("Birebir aynı dosya", "exact")
        self.cmb_mode.addItem("Aynı desen / varyant", "similar")
        self.cmb_mode.addItem("Benzer doku", "style")
        self.cmb_mode.addItem("Daha kapsamlı bul", "comprehensive")
        idx_m = self.cmb_mode.findData(self._settings.search_mode)
        if idx_m >= 0:
            self.cmb_mode.setCurrentIndex(idx_m)
        row2.addWidget(self.cmb_mode)

        row2.addWidget(QLabel("Renk"))
        self.cmb_color = QComboBox()
        self.cmb_color.setMinimumWidth(120)
        self.cmb_color.addItem("Renk normal", "normal")
        self.cmb_color.addItem("Renk önemli", "important")
        self.cmb_color.addItem("Renk önemsiz", "ignore")
        idx_c = self.cmb_color.findData(self._settings.color_weight_mode)
        if idx_c >= 0:
            self.cmb_color.setCurrentIndex(idx_c)
        row2.addWidget(self.cmb_color)

        self.chk_near_below = QCheckBox("Eşik altı yakınlar")
        self.chk_near_below.setChecked(self._settings.show_near_below_threshold)
        row2.addWidget(self.chk_near_below)
        root.addLayout(row2)

        self.btn_search.clicked.connect(self.search_clicked.emit)
        self.btn_pick_image.clicked.connect(self.pick_image_clicked.emit)
        self.btn_quick_folder.clicked.connect(self.quick_folder_clicked.emit)
        self.btn_clear.clicked.connect(self.clear_clicked.emit)
        self.btn_cancel.clicked.connect(self.cancel_search_clicked.emit)
        self.slider_threshold.valueChanged.connect(self._on_threshold)
        for w in (
            self.cmb_limit,
            self.cmb_scope,
            self.cmb_customer,
            self.cmb_mode,
            self.cmb_color,
        ):
            w.currentIndexChanged.connect(lambda *_: self.filter_research.emit())
        self.chk_near_below.stateChanged.connect(self.filter_changed.emit)

    def _on_threshold(self, value: int) -> None:
        self.lbl_threshold.setText(f"%{value}")
        self.filter_changed.emit()

    def set_searching(self, active: bool) -> None:
        self.btn_cancel.setVisible(active)
        self.btn_search.setEnabled(not active)
