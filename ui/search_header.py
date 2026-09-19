"""Üst arama alanı — metin kutusu, müşteri/klasör akıllı arama ve dosya seçimleri."""

from __future__ import annotations

from PySide6.QtCore import QStringListModel, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class SearchHeader(QWidget):
    search_clicked = Signal()
    pick_image_clicked = Signal()
    pick_files_clicked = Signal()
    pick_folder_clicked = Signal()
    quick_folder_clicked = Signal()
    clear_clicked = Signal()
    cancel_search_clicked = Signal()
    toggle_filters_clicked = Signal()
    toggle_sources_clicked = Signal()
    toggle_mode_clicked = Signal()
    reset_layout_clicked = Signal()
    preset_changed = Signal(str)
    scope_changed = Signal(str)
    customer_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SearchHeader")
        self._simple_mode = True
        self._has_image = False
        self._customer_labels: list[str] = []
        self._selected_customer = ""
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 2)
        root.setSpacing(0)

        group = QGroupBox("Arama")
        group.setObjectName("SearchHeaderGroup")
        root.addWidget(group)

        lay = QVBoxLayout(group)
        lay.setContentsMargins(14, 14, 14, 10)
        lay.setSpacing(8)

        row_query = QHBoxLayout()
        row_query.setSpacing(10)
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText(
            "Desen ara: leopard, çiçek, ekose, tavşan, kamuflaj…"
        )
        self.txt_search.setMinimumWidth(280)
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.textChanged.connect(self._update_search_button)

        # Müşteri / Klasör — akıllı arama (path keşfi + fuzzy)
        self.txt_customer = QLineEdit()
        self.txt_customer.setObjectName("CustomerSmartSearch")
        self.txt_customer.setPlaceholderText("Müşteri / Klasör")
        self.txt_customer.setMinimumWidth(180)
        self.txt_customer.setMaximumWidth(280)
        self.txt_customer.setClearButtonEnabled(True)
        self.txt_customer.setToolTip(
            "Aramayı yalnızca seçilen müşteri/klasör kapsamına alır. "
            "Boş bırakılırsa tüm indeks aranır."
        )
        self._customer_model = QStringListModel(self)
        self._customer_completer = QCompleter(self._customer_model, self)
        self._customer_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._customer_completer.setCompletionMode(
            QCompleter.CompletionMode.UnfilteredPopupCompletion
        )
        self._customer_completer.setMaxVisibleItems(10)
        self.txt_customer.setCompleter(self._customer_completer)
        self.txt_customer.textEdited.connect(self._on_customer_text_edited)
        self.txt_customer.editingFinished.connect(self._on_customer_editing_finished)
        self._customer_completer.activated.connect(self._on_customer_activated)

        self.btn_search = QPushButton("Ara")
        self.btn_search.setObjectName("primaryBtn")
        self.btn_search.setMinimumWidth(120)

        self.btn_reset_layout = QPushButton("Panel Sıfırla")
        self.btn_reset_layout.setMinimumWidth(110)
        self.btn_reset_layout.setToolTip("Panel düzenini varsayılanlara döndür")

        row_query.addWidget(self.txt_search, stretch=1)
        row_query.addWidget(self.txt_customer, stretch=0)
        row_query.addWidget(self.btn_reset_layout)
        row_query.addWidget(self.btn_search)
        lay.addLayout(row_query)

        self.lbl_search_summary = QLabel("")
        self.lbl_search_summary.setStyleSheet("color:#94a3b8;font-size:11px;")
        self.lbl_scope = QLabel("")
        self.lbl_scope.setStyleSheet("color:#60a5fa;font-size:11px;")
        lay.addWidget(self.lbl_search_summary)
        lay.addWidget(self.lbl_scope)

        row_simple = QHBoxLayout()
        row_simple.setSpacing(8)
        self.btn_pick_image = QPushButton("Görsel Seç")
        self.btn_clear = QPushButton("Temizle")
        self.btn_filters = QPushButton("Filtreler")
        self.btn_panels = QToolButton()
        self.btn_panels.setText("Paneller")
        self.btn_panels.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_panels.setMinimumWidth(90)
        self.btn_panels.setToolTip("Panelleri aç/kapat")
        self.panels_menu = QMenu(self.btn_panels)
        self.btn_panels.setMenu(self.panels_menu)
        self.btn_mode = QPushButton("Gelişmiş Mod")
        self.btn_cancel = QPushButton("İptal")
        self.btn_cancel.setObjectName("dangerBtn")
        self.btn_cancel.setVisible(False)
        self.cmb_scope = QComboBox()
        self.cmb_scope.setMinimumWidth(180)
        self.cmb_scope.addItem("Tüm aktif kaynaklar", "all")
        self.cmb_scope.addItem("Seçili kaynaklar", "selected_sources")
        self.cmb_scope.addItem("Sadece bu klasör", "selected_folder")
        for btn, width in (
            (self.btn_pick_image, 100),
            (self.btn_filters, 90),
            (self.btn_panels, 90),
            (self.btn_mode, 120),
            (self.btn_cancel, 70),
        ):
            btn.setMinimumWidth(width)
        row_simple.addWidget(self.btn_pick_image)
        row_simple.addWidget(self.btn_filters)
        row_simple.addWidget(self.btn_panels)
        row_simple.addWidget(self.btn_mode)
        row_simple.addWidget(self.btn_cancel)
        row_simple.addStretch(1)
        self._row_simple = row_simple
        lay.addLayout(row_simple)

        row_advanced = QHBoxLayout()
        row_advanced.setSpacing(8)
        row_advanced.addWidget(QLabel("Arama Türü"))
        self.cmb_preset = QComboBox()
        self.cmb_preset.setMinimumWidth(200)
        from core.search_presets import SEARCH_PRESETS

        for preset in SEARCH_PRESETS:
            self.cmb_preset.addItem(preset.label, preset.key)
        self.btn_pick_files = QPushButton("Dosyalar Seç")
        self.btn_pick_folder = QPushButton("Klasör Seç")
        self.btn_quick_folder = QPushButton("Bu Klasörde Hızlı Ara")
        self.btn_sources = QPushButton("Kaynak / Index")
        self.btn_clear_adv = self.btn_clear
        for btn, width in (
            (self.btn_pick_files, 110),
            (self.btn_pick_folder, 100),
            (self.btn_quick_folder, 160),
            (self.btn_sources, 120),
            (self.btn_clear, 80),
        ):
            btn.setMinimumWidth(width)
        row_advanced.addWidget(self.cmb_preset)
        row_advanced.addWidget(self.btn_pick_files)
        row_advanced.addWidget(self.btn_pick_folder)
        row_advanced.addWidget(self.btn_quick_folder)
        row_advanced.addWidget(self.btn_sources)
        row_advanced.addWidget(QLabel("Kapsam"))
        row_advanced.addWidget(self.cmb_scope)
        row_advanced.addWidget(self.btn_clear)
        row_advanced.addStretch(1)
        self._row_advanced = row_advanced
        self._advanced_widgets: list[QWidget] = []
        for i in range(row_advanced.count()):
            item = row_advanced.itemAt(i)
            if item and item.widget():
                self._advanced_widgets.append(item.widget())
        lay.addLayout(row_advanced)

        self.btn_search.clicked.connect(self.search_clicked.emit)
        self.btn_pick_image.clicked.connect(self._on_pick_image)
        self.btn_pick_files.clicked.connect(self.pick_files_clicked.emit)
        self.btn_pick_folder.clicked.connect(self.pick_folder_clicked.emit)
        self.btn_quick_folder.clicked.connect(self.quick_folder_clicked.emit)
        self.btn_filters.clicked.connect(self.toggle_filters_clicked.emit)
        self.btn_sources.clicked.connect(self.toggle_sources_clicked.emit)
        self.btn_mode.clicked.connect(self.toggle_mode_clicked.emit)
        self.btn_reset_layout.clicked.connect(self.reset_layout_clicked.emit)
        self.btn_clear.clicked.connect(self.clear_clicked.emit)
        self.btn_cancel.clicked.connect(self.cancel_search_clicked.emit)
        self.cmb_preset.currentIndexChanged.connect(
            lambda *_: self.preset_changed.emit(
                str(self.cmb_preset.currentData() or "")
            )
        )
        self.cmb_scope.currentIndexChanged.connect(
            lambda *_: self.scope_changed.emit(
                str(self.cmb_scope.currentData() or "all")
            )
        )
        self.set_simple_mode(True)

    def _on_pick_image(self) -> None:
        self._has_image = True
        self._update_search_button()
        self.pick_image_clicked.emit()

    def set_has_image(self, has_image: bool) -> None:
        self._has_image = has_image
        self._update_search_button()

    def _update_search_button(self) -> None:
        text = self.txt_search.text().strip()
        if self._has_image and text:
            self.btn_search.setText("Görsel + Metin")
            self.btn_search.setToolTip("Görsel ve metinle birlikte ara")
        elif text and not self._has_image:
            self.btn_search.setText("Metinle Ara")
            self.btn_search.setToolTip("Sadece metinle ara")
        else:
            self.btn_search.setText("Ara")
            self.btn_search.setToolTip("Görsel veya metinle ara")

    def set_simple_mode(self, simple: bool) -> None:
        self._simple_mode = simple
        for w in self._advanced_widgets:
            w.setVisible(not simple)
        self.btn_reset_layout.setVisible(not simple)
        self.lbl_search_summary.setVisible(not simple)
        self.lbl_scope.setVisible(not simple)

    def set_search_summary(self, text: str) -> None:
        self.lbl_search_summary.setText(text)

    def set_scope_hint(self, text: str) -> None:
        self.lbl_scope.setText(text)

    def set_searching(self, active: bool) -> None:
        self.btn_cancel.setVisible(active)
        self.btn_search.setEnabled(not active)
        self.btn_pick_image.setEnabled(not active)
        self.btn_pick_files.setEnabled(not active)
        self.btn_pick_folder.setEnabled(not active)
        self.btn_quick_folder.setEnabled(not active)
        self.btn_filters.setEnabled(not active)
        self.txt_customer.setEnabled(not active)
        self.btn_sources.setEnabled(True)
        self.btn_mode.setEnabled(True)

    def set_customer_suggestions(self, labels: list[str]) -> None:
        """Fuzzy/keşif öneri listesini güncelle (orijinal müşteri adları)."""
        self._customer_labels = list(labels or [])
        self._customer_model.setStringList(self._customer_labels)

    def selected_customer(self) -> str:
        return (self._selected_customer or self.txt_customer.text() or "").strip()

    def set_selected_customer(self, name: str) -> None:
        name = (name or "").strip()
        self._selected_customer = name
        if self.txt_customer.text() != name:
            self.txt_customer.blockSignals(True)
            self.txt_customer.setText(name)
            self.txt_customer.blockSignals(False)

    def _on_customer_text_edited(self, text: str) -> None:
        # Yazarken seçimi henüz kilitleme — boşaltılırsa filtre kalkar
        if not (text or "").strip():
            if self._selected_customer:
                self._selected_customer = ""
                self.customer_changed.emit("")
            return
        # Canlı öneri: dışarıdan model güncellenir (main_window)

    def _on_customer_activated(self, text: str) -> None:
        name = str(text or "").strip()
        self._selected_customer = name
        self.txt_customer.setText(name)
        self.customer_changed.emit(name)

    def _on_customer_editing_finished(self) -> None:
        text = self.txt_customer.text().strip()
        if not text:
            if self._selected_customer:
                self._selected_customer = ""
                self.customer_changed.emit("")
            return
        # Tam eşleşen öneri varsa kilitle; yoksa serbest metni aday olarak bırak
        match = ""
        for label in self._customer_labels:
            if label.casefold() == text.casefold():
                match = label
                break
        if match:
            self._selected_customer = match
            if self.txt_customer.text() != match:
                self.txt_customer.setText(match)
            self.customer_changed.emit(match)
        else:
            # Belirsiz — otomatik zorla eşleme yok; yine de metni scope adayı yap
            self._selected_customer = text
            self.customer_changed.emit(text)
