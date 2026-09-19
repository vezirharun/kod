"""Kaynak yönetim paneli — çok kaynaklı merkezi index."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.sources import SOURCE_TYPE_LABELS_TR


class SourcesPanel(QWidget):
    add_source_clicked = Signal()

    edit_source_clicked = Signal(int)

    toggle_active_clicked = Signal(int)

    detach_source_clicked = Signal(int)

    purge_source_clicked = Signal(int)

    purge_orphans_clicked = Signal()

    scan_source_clicked = Signal(int, str)

    scan_all_clicked = Signal(str)

    scan_due_clicked = Signal()

    source_selection_changed = Signal(int)

    source_checks_changed = Signal()

    def __init__(self, parent=None):

        super().__init__(parent)

        self._sources: list[dict] = []
        self._source_checks_initialized = False

        self._build_ui()

    def _build_ui(self) -> None:

        layout = QVBoxLayout(self)

        group = QGroupBox("Index Kaynakları")

        g_layout = QVBoxLayout(group)

        self.table = QTableWidget(0, 7)

        self.table.setHorizontalHeaderLabels(
            [
                "Ad",
                "Kaynak Yolu",
                "Tür",
                "Dosya",
                "Hata",
                "Cache",
                "Son Tarama",
            ]
        )

        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )

        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        self.table.itemSelectionChanged.connect(self._on_selection)

        g_layout.addWidget(self.table)

        btn_row1 = QHBoxLayout()

        self.btn_add = QPushButton("Kaynak Ekle")

        self.btn_edit = QPushButton("Düzenle")

        self.btn_toggle = QPushButton("Aktif/Pasif")

        self.btn_detach = QPushButton("Kaynağı Kaldır")

        self.btn_purge = QPushButton("Sil ve Index Temizle")

        self.btn_purge.setToolTip(
            "Index kayıtlarını temizler; orijinal dosyalar silinmez"
        )

        self.btn_purge_orphans = QPushButton("Yetim Kayıtları Temizle")
        self.btn_purge_orphans.setToolTip(
            "sources tablosunda olmayan hayalet index kayıtlarını siler"
        )

        self.btn_add.clicked.connect(self.add_source_clicked.emit)

        self.btn_edit.clicked.connect(self._emit_edit)

        self.btn_toggle.clicked.connect(self._emit_toggle)

        self.btn_detach.clicked.connect(self._emit_detach)

        self.btn_purge.clicked.connect(self._emit_purge)

        self.btn_purge_orphans.clicked.connect(self.purge_orphans_clicked.emit)

        for btn in (self.btn_add, self.btn_edit, self.btn_toggle):
            btn.setMinimumHeight(32)
            btn_row1.addWidget(btn)

        g_layout.addLayout(btn_row1)

        btn_row1b = QHBoxLayout()

        for btn in (self.btn_detach, self.btn_purge):
            btn.setMinimumHeight(32)
            btn_row1b.addWidget(btn)

        btn_row1b.addStretch(1)

        g_layout.addLayout(btn_row1b)

        btn_row1c = QHBoxLayout()
        self.btn_purge_orphans.setMinimumHeight(32)
        btn_row1c.addWidget(self.btn_purge_orphans)
        btn_row1c.addStretch(1)
        g_layout.addLayout(btn_row1c)

        btn_row2 = QHBoxLayout()

        self.cmb_scan_mode = QComboBox()

        self.cmb_scan_mode.addItem("Hızlı güncelleme", "quick")

        self.cmb_scan_mode.addItem("Derin kontrol (haftalık)", "deep")

        self.btn_scan_sel = QPushButton("Seçili Kaynağı Tara")

        self.btn_scan_all = QPushButton("Tüm Aktif Kaynaklar")

        self.btn_scan_due = QPushButton("Bekleyen Güncellemeler")

        self.btn_scan_sel.clicked.connect(self._emit_scan_selected)

        self.btn_scan_all.clicked.connect(
            lambda: self.scan_all_clicked.emit(self.cmb_scan_mode.currentData())
        )

        self.btn_scan_due.clicked.connect(self.scan_due_clicked.emit)

        btn_row2.addWidget(self.cmb_scan_mode, stretch=1)

        g_layout.addLayout(btn_row2)

        btn_row2b = QHBoxLayout()

        for btn in (self.btn_scan_sel, self.btn_scan_all, self.btn_scan_due):
            btn.setMinimumHeight(32)
            btn_row2b.addWidget(btn)

        g_layout.addLayout(btn_row2b)

        self.chk_auto_scan = QCheckBox("Açılışta bekleyen güncellemeleri otomatik tara")

        self.chk_auto_scan.setChecked(False)

        g_layout.addWidget(self.chk_auto_scan)

        scope_group = QGroupBox("Arama Kaynakları (çoklu seçim)")

        scope_layout = QVBoxLayout(scope_group)

        self._source_checks: dict[int, QCheckBox] = {}

        self._scope_container = QVBoxLayout()

        scope_layout.addLayout(self._scope_container)

        sel_row = QHBoxLayout()

        self.btn_sel_all = QPushButton("Tümünü seç")

        self.btn_sel_none = QPushButton("Tümünü kaldır")

        self.btn_sel_all.clicked.connect(self._select_all_sources)

        self.btn_sel_none.clicked.connect(self._deselect_all_sources)

        sel_row.addWidget(self.btn_sel_all)

        sel_row.addWidget(self.btn_sel_none)

        scope_layout.addLayout(sel_row)

        self.lbl_scope_hint = QLabel("Arama kapsamında 'Seçili kaynaklarda ara' seçin.")

        self.lbl_scope_hint.setWordWrap(True)

        self.lbl_scope_hint.setStyleSheet("color: #888; font-size: 11px;")

        scope_layout.addWidget(self.lbl_scope_hint)

        layout.addWidget(scope_group)

        self.lbl_summary = QLabel("Kaynak: 0")

        g_layout.addWidget(self.lbl_summary)

        layout.addWidget(group)

    def set_auto_scan(self, enabled: bool) -> None:

        self.chk_auto_scan.setChecked(enabled)

    def is_auto_scan_enabled(self) -> bool:

        return self.chk_auto_scan.isChecked()

    def set_sources(self, sources: list[dict]) -> None:

        self._sources = sources

        self.table.setRowCount(len(sources))

        for row, src in enumerate(sources):
            active = bool(src.get("is_active"))

            name = src.get("name", "")

            if not active:
                name = f"⏸ {name}"

            self.table.setItem(row, 0, QTableWidgetItem(name))

            root_path = str(src.get("root_path") or "—")
            path_item = QTableWidgetItem(root_path)
            path_item.setToolTip(root_path)
            self.table.setItem(row, 1, path_item)

            stype = src.get("source_type", "")

            type_label = SOURCE_TYPE_LABELS_TR.get(stype, stype)

            self.table.setItem(row, 2, QTableWidgetItem(type_label))

            self.table.setItem(row, 3, QTableWidgetItem(str(src.get("file_count", 0))))

            self.table.setItem(row, 4, QTableWidgetItem(str(src.get("error_count", 0))))

            cache = src.get("cache_status", "—")

            self.table.setItem(row, 5, QTableWidgetItem(cache))

            last = (src.get("last_scan_at") or "—")[:16].replace("T", " ")

            self.table.setItem(row, 6, QTableWidgetItem(last))

            for col in range(7):
                item = self.table.item(row, col)

                if item:
                    item.setData(Qt.ItemDataRole.UserRole, src.get("id"))

                    if not active:
                        item.setForeground(Qt.GlobalColor.gray)

        active_count = sum(1 for s in sources if s.get("is_active"))

        inactive = len(sources) - active_count

        self.lbl_summary.setText(
            f"Toplam: {len(sources)} | Aktif: {active_count} | Pasif: {inactive}"
        )

        self._rebuild_source_checks(sources)

    def _rebuild_source_checks(self, sources: list[dict]) -> None:
        # previous only — asla "first_build → hepsini işaretle".
        # İşaretli yok = selected_source_ids=[] = tüm arşiv (orphan dahil).
        # Checkbox dolu = yalnız o source_id'ler (orphan sızamaz).
        previous_ids = set(self.selected_source_ids())

        while self._scope_container.count():
            item = self._scope_container.takeAt(0)

            if item.widget():
                item.widget().deleteLater()

        self._source_checks.clear()

        for src in sources:
            if not src.get("is_active"):
                continue

            sid = int(src.get("id", 0))

            if not sid:
                continue

            label = f"{src.get('name', '')}  —  {src.get('root_path', '')}"

            chk = QCheckBox(label)

            chk.setChecked(sid in previous_ids)

            chk.setProperty("source_id", sid)

            chk.stateChanged.connect(self._on_source_check_changed)

            self._source_checks[sid] = chk

            self._scope_container.addWidget(chk)

        self._source_checks_initialized = True
        self._update_scope_hint()

    def selected_source_ids(self) -> list[int]:

        return [sid for sid, chk in self._source_checks.items() if chk.isChecked()]

    def set_checked_source_ids(self, ids: list[int]) -> None:

        id_set = set(ids)

        for sid, chk in self._source_checks.items():
            chk.blockSignals(True)
            chk.setChecked(sid in id_set)
            chk.blockSignals(False)
        self._update_scope_hint()

    def select_all_sources(self) -> None:

        self._select_all_sources()

    def clear_source_checks(self) -> None:

        self._deselect_all_sources()

    def _select_all_sources(self) -> None:

        for chk in self._source_checks.values():
            chk.blockSignals(True)
            chk.setChecked(True)
            chk.blockSignals(False)

        self._update_scope_hint()
        self.source_checks_changed.emit()

    def _deselect_all_sources(self) -> None:

        for chk in self._source_checks.values():
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)

        self._update_scope_hint()
        self.source_checks_changed.emit()

    def _on_source_check_changed(self, _state: int) -> None:
        self._update_scope_hint()
        self.source_checks_changed.emit()

    def _update_scope_hint(self) -> None:
        selected_ids = set(self.selected_source_ids())
        active = [src for src in self._sources if src.get("is_active")]
        selected = [
            str(src.get("name") or src.get("root_path") or src.get("id"))
            for src in active
            if int(src.get("id") or 0) in selected_ids
        ]
        if selected:
            names = ", ".join(selected[:4])
            if len(selected) > 4:
                names += f" +{len(selected) - 4}"
            self.lbl_scope_hint.setText(
                f"Seçili arama kaynağı: {len(selected)}/{len(active)} — {names}"
            )
            self.lbl_scope_hint.setStyleSheet(
                "color:#60a5fa;font-size:11px;font-weight:600;"
            )
        else:
            self.lbl_scope_hint.setText(
                "Kaynak seçili değil → index kapsamı: Tüm arşiv (kayıtlı + orphan)."
            )
            self.lbl_scope_hint.setStyleSheet(
                "color:#f59e0b;font-size:11px;font-weight:600;"
            )

    def selected_source_id(self) -> int:

        rows = self.table.selectionModel().selectedRows()

        if not rows:
            return 0

        item = self.table.item(rows[0].row(), 0)

        return int(item.data(Qt.ItemDataRole.UserRole)) if item else 0

    def _on_selection(self) -> None:

        sid = self.selected_source_id()

        if sid:
            self.source_selection_changed.emit(sid)

    def _emit_edit(self) -> None:

        sid = self.selected_source_id()

        if sid:
            self.edit_source_clicked.emit(sid)

    def _emit_toggle(self) -> None:

        sid = self.selected_source_id()

        if sid:
            self.toggle_active_clicked.emit(sid)

    def _emit_detach(self) -> None:

        sid = self.selected_source_id()

        if not sid:
            QMessageBox.information(self, "Bilgi", "Önce bir kaynak seçin.")

            return

        self.detach_source_clicked.emit(sid)

    def _emit_purge(self) -> None:

        sid = self.selected_source_id()

        if not sid:
            QMessageBox.information(self, "Bilgi", "Önce bir kaynak seçin.")

            return

        self.purge_source_clicked.emit(sid)

    def _emit_scan_selected(self) -> None:

        sid = self.selected_source_id()

        if not sid:
            QMessageBox.information(self, "Bilgi", "Önce bir kaynak seçin.")

            return

        self.scan_source_clicked.emit(sid, self.cmb_scan_mode.currentData())
