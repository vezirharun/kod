"""Arama ayarları — filtreler ve istatistik özeti."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.settings import AppSettings
from core.sources import SEARCH_SCOPE_LABELS
from ui.closable_bar import ClosableBar
from ui.search_summary_panel import SearchSummaryPanel


class FilterPanel(QWidget):
    filter_changed = Signal()
    filter_research = Signal()
    close_requested = Signal()
    reindex_requested = Signal()
    ai_enable_warning = Signal()
    panel_settings_changed = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._ai_settings = None  # lazy — ilk AI sekmesi açılışında
        self._ai_tab_index = -1
        self._build_ui()

    @property
    def ai_settings(self):
        self._ensure_ai_settings()
        return self._ai_settings

    def set_advanced_mode(self, advanced: bool) -> None:
        if self._ai_settings is not None:
            self._ai_settings.set_advanced_mode(advanced)

    def _ensure_ai_settings(self):
        if self._ai_settings is not None:
            return self._ai_settings
        from ui.ai_settings_panel import AiSettingsPanel

        idx = self._ai_tab_index if self._ai_tab_index >= 0 else self.tabs.count()
        old = self.tabs.widget(idx) if idx < self.tabs.count() else None
        self._ai_settings = AiSettingsPanel(self._settings)
        self._ai_settings.settings_changed.connect(self.panel_settings_changed.emit)
        self._ai_settings.reindex_requested.connect(self.reindex_requested.emit)
        self._ai_settings.ai_enable_warning.connect(self.ai_enable_warning.emit)
        self._ai_settings.set_advanced_mode(self._settings.ui_mode == "advanced")
        if old is not None:
            self.tabs.removeTab(idx)
            if old is not self._ai_settings:
                old.deleteLater()
        self._ai_tab_index = self.tabs.insertTab(idx, self._ai_settings, "AI & Index")
        self.tabs.setCurrentIndex(self._ai_tab_index)
        return self._ai_settings

    def _on_tab_changed(self, index: int) -> None:
        # AI sekmesine geçince paneli oluştur (UI donmasın diye lazy)
        if index == self._ai_tab_index and self._ai_settings is None:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, self._ensure_ai_settings)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 2, 8, 2)
        root.setSpacing(0)
        self._title_bar = ClosableBar("Arama & AI Ayarları")
        self._title_bar.closed.connect(self.close_requested.emit)
        root.addWidget(self._title_bar)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        search_tab = QWidget()
        search_layout = QVBoxLayout(search_tab)
        group = QGroupBox()
        group.setFlat(True)
        search_layout.addWidget(group)
        inner = QVBoxLayout(group)
        inner.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        row = 0
        grid.addWidget(QLabel("Eşik"), row, 0)
        thresh_row = QHBoxLayout()
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(30, 100)
        self.slider_threshold.setValue(int(self._settings.similarity_threshold * 100))
        self.slider_threshold.setMinimumWidth(140)
        self.lbl_threshold = QLabel(f"%{self.slider_threshold.value()}")
        self.lbl_threshold.setMinimumWidth(40)
        self.lbl_threshold_count = QLabel("")
        thresh_row.addWidget(self.slider_threshold, stretch=1)
        thresh_row.addWidget(self.lbl_threshold)
        thresh_row.addWidget(self.lbl_threshold_count)
        grid.addLayout(thresh_row, row, 1, 1, 5)
        row += 1

        grid.addWidget(QLabel("Sonuç"), row, 0)
        self.cmb_limit = QComboBox()
        self.cmb_limit.setMinimumWidth(90)
        self.cmb_limit.addItem("Sınırsız", 0)
        grid.addWidget(self.cmb_limit, row, 1)

        grid.addWidget(QLabel("Kapsam"), row, 2)
        self.cmb_scope = QComboBox()
        self.cmb_scope.setMinimumWidth(130)
        for scope, label in SEARCH_SCOPE_LABELS.items():
            self.cmb_scope.addItem(label, scope.value)
        idx = self.cmb_scope.findData(self._settings.search_scope)
        if idx >= 0:
            self.cmb_scope.setCurrentIndex(idx)
        grid.addWidget(self.cmb_scope, row, 3)

        grid.addWidget(QLabel("Müşteri"), row, 4)
        self.cmb_customer = QComboBox()
        self.cmb_customer.setMinimumWidth(100)
        self.cmb_customer.addItem("Tümü", "")
        grid.addWidget(self.cmb_customer, row, 5)
        row += 1

        grid.addWidget(QLabel("Mod"), row, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.setMinimumWidth(150)
        self.cmb_mode.addItem("Birebir", "exact")
        self.cmb_mode.addItem("Aynı desen / varyant", "similar")
        self.cmb_mode.addItem("Benzer doku", "style")
        self.cmb_mode.addItem("Daha kapsamlı bul", "comprehensive")
        idx_m = self.cmb_mode.findData(self._settings.search_mode)
        if idx_m >= 0:
            self.cmb_mode.setCurrentIndex(idx_m)
        grid.addWidget(self.cmb_mode, row, 1, 1, 2)

        grid.addWidget(QLabel("Renk modu"), row, 3)
        self.cmb_color = QComboBox()
        self.cmb_color.setMinimumWidth(150)
        self.cmb_color.addItem("Renk normal", "normal")
        self.cmb_color.addItem("Renk önemli", "important")
        self.cmb_color.addItem("Renk önemsiz / doku", "ignore")
        idx_c = self.cmb_color.findData(self._settings.color_weight_mode)
        if idx_c >= 0:
            self.cmb_color.setCurrentIndex(idx_c)
        grid.addWidget(self.cmb_color, row, 4, 1, 2)
        row += 1

        grid.addWidget(QLabel("Yöntem"), row, 0)
        self.cmb_search_method = QComboBox()
        self.cmb_search_method.setMinimumWidth(180)
        from core.search_method_modes import (
            DEFAULT_SEARCH_METHOD,
            primary_search_method_labels,
        )

        for key, label in primary_search_method_labels():
            self.cmb_search_method.addItem(label, key)
        method_key = getattr(self._settings, "search_method_mode", DEFAULT_SEARCH_METHOD)
        idx_method = self.cmb_search_method.findData(method_key)
        if idx_method >= 0:
            self.cmb_search_method.setCurrentIndex(idx_method)
        grid.addWidget(self.cmb_search_method, row, 1, 1, 2)
        row += 1

        self.chk_near_below = QCheckBox("Eşik altı yakınları göster")
        self.chk_near_below.setChecked(self._settings.show_near_below_threshold)
        grid.addWidget(self.chk_near_below, row, 0, 1, 6)
        inner.addLayout(grid)

        self.chk_semantic_search = QCheckBox("Semantik metin arama kullan")
        self.chk_semantic_search.setChecked(
            bool(getattr(self._settings, "semantic_text_search_enabled", False))
        )
        self.lbl_semantic_status = QLabel("")
        self.lbl_semantic_status.setWordWrap(True)
        self.lbl_semantic_status.setStyleSheet("color:#fbbf24;font-size:11px;")
        inner.addWidget(self.chk_semantic_search)
        inner.addWidget(self.lbl_semantic_status)
        self._refresh_semantic_status()

        self.summary_panel = SearchSummaryPanel()
        inner.addWidget(self.summary_panel)

        self.tabs.addTab(search_tab, "Arama")

        # AI sekmesi lazy — placeholder; ilk tıklamada AiSettingsPanel yaratılır
        placeholder = QWidget()
        ph_l = QVBoxLayout(placeholder)
        ph_lbl = QLabel("AI ayarları yükleniyor…")
        ph_lbl.setStyleSheet("color:#94a3b8;")
        ph_l.addWidget(ph_lbl)
        ph_l.addStretch()
        self._ai_tab_index = self.tabs.addTab(placeholder, "AI & Index")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.slider_threshold.valueChanged.connect(self._on_threshold)
        for w in (
            self.cmb_limit,
            self.cmb_scope,
            self.cmb_customer,
            self.cmb_mode,
            self.cmb_color,
            self.cmb_search_method,
        ):
            w.currentIndexChanged.connect(lambda *_: self.filter_research.emit())
        self.chk_near_below.stateChanged.connect(self.filter_changed.emit)
        self.chk_semantic_search.toggled.connect(self._on_semantic_toggled)

    def _on_threshold(self, value: int) -> None:
        self.lbl_threshold.setText(f"%{value}")
        self.filter_changed.emit()

    def _on_semantic_toggled(self, checked: bool) -> None:
        self._settings.semantic_text_search_enabled = bool(checked)
        self._settings.save()
        self._refresh_semantic_status()
        self.panel_settings_changed.emit()

    def _refresh_semantic_status(self) -> None:
        if self.chk_semantic_search.isChecked():
            self.lbl_semantic_status.setText(
                "Semantik arama açık. Eksik etiketler Kaynak & Index panelinden tamamlanabilir."
            )
            self.lbl_semantic_status.setStyleSheet("color:#a3e635;font-size:11px;")
        else:
            self.lbl_semantic_status.setText(
                "Semantik arama kapalı; mevcut dosya adı ve kayıtlı etiket araması kullanılır."
            )
            self.lbl_semantic_status.setStyleSheet("color:#fbbf24;font-size:11px;")
