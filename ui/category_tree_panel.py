"""Sol panel — hiyerarşik kategori filtresi."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.category_tree import build_category_tree_hierarchy, category_path


class CategoryTreePanel(QWidget):
    category_filter_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_path = ""
        self._db_path = ""
        self._tree_source_mtime = 0.0
        self._build_ui()
        self._populate_tree()

    def is_tree_ready(self, db_path: str | None = None) -> bool:
        """True when widget already holds a populated tree for this DB."""
        want = self._db_path if db_path is None else str(db_path or "")
        return want == self._db_path and self.tree.topLevelItemCount() > 0

    @staticmethod
    def _source_mtime(db_path: str) -> float:
        """Max mtime of DBs that feed dynamic/taught categories."""
        if not db_path:
            return 0.0
        paths: list[str] = [db_path]
        try:
            from core.category_memory import _read_db_paths

            paths = list(_read_db_paths(db_path)) or paths
        except Exception:
            pass
        mt = 0.0
        for p in paths:
            try:
                mt = max(mt, float(Path(p).stat().st_mtime))
            except OSError:
                continue
        return mt

    def set_db_path(self, db_path: str) -> None:
        new = str(db_path or "")
        # StatusWorker ticks used to call this every ~1s and rebuild the whole
        # tree on the UI thread (freeze stacks → _populate_tree / set_db_path).
        if new == self._db_path and self.tree.topLevelItemCount() > 0:
            # Rebuild only when category memory / concepts DB changed (teach).
            if self._source_mtime(new) <= self._tree_source_mtime:
                return
        self._db_path = new
        self.refresh()

    def refresh(self) -> None:
        """Hard-code + öğretilmiş (category_memory) ağacı yeniden kur."""
        prev = self._active_path
        self._populate_tree()
        if prev:
            self._active_path = prev
            self.lbl_active.setText(f"Aktif: {prev}")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("<b>Kategori Filtresi</b>"))
        hint = QLabel("Kategori seçince arama yalnızca o grupta yapılır.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8;font-size:11px;")
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.btn_clear = QPushButton("Filtreyi Kaldır")
        self.btn_clear.clicked.connect(self.clear_filter)
        row.addWidget(self.btn_clear)
        row.addStretch()
        layout.addLayout(row)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree, stretch=1)
        self.lbl_active = QLabel("Aktif: Tümü")
        self.lbl_active.setStyleSheet("color:#8bc34a;font-weight:600;")
        layout.addWidget(self.lbl_active)

    def _populate_tree(self) -> None:
        self.tree.clear()
        db = self._db_path
        # One-pass hierarchy: avoids N+1 concepts()/category_memory opens.
        parents, children_map = build_category_tree_hierarchy(db)
        for parent in parents:
            parent_item = QTreeWidgetItem([parent])
            parent_item.setData(0, Qt.ItemDataRole.UserRole, category_path(parent))
            self.tree.addTopLevelItem(parent_item)
            children = list(children_map.get(parent) or [])
            for child in children:
                child_item = QTreeWidgetItem([child])
                child_item.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    category_path(parent, child),
                )
                parent_item.addChild(child_item)
            parent_item.setExpanded(len(children) <= 8)
        self._tree_source_mtime = self._source_mtime(db)

    def _on_item_clicked(self, item: QTreeWidgetItem) -> None:
        path = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        self._active_path = path
        self.lbl_active.setText(f"Aktif: {path or 'Tümü'}")
        self.category_filter_changed.emit(path)

    def clear_filter(self) -> None:
        self._active_path = ""
        self.tree.clearSelection()
        self.lbl_active.setText("Aktif: Tümü")
        self.category_filter_changed.emit("")

    def active_path(self) -> str:
        return self._active_path
