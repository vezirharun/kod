"""Sol panel — sekmeli kaynak / index / güncelleme / cache."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.progress_panel import ProgressPanel
from ui.sources_panel import SourcesPanel


class SourceSidebar(QWidget):
    """Kaynak yönetimi ve index işlemleri — sekmeli sol navigasyon."""

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
    index_start_clicked = Signal()
    index_stop_clicked = Signal()
    index_pause_clicked = Signal()
    index_resume_clicked = Signal()
    index_fast_archive_clicked = Signal()
    index_night_complete_clicked = Signal()
    index_backfill_clicked = Signal()
    index_purge_broken_clicked = Signal()
    index_requeue_thumbs_clicked = Signal()
    index_queue_ai_clicked = Signal()
    semantic_backfill_clicked = Signal()
    cache_refresh_clicked = Signal()
    cache_clean_clicked = Signal()
    cache_fill_missing_clicked = Signal()
    folder_watch_changed = Signal(bool)
    purge_missing_after_scan_changed = Signal(bool)
    purge_missing_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("<b>Kaynaklar ve İndeks</b>")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_sources(), "Kaynaklar")
        self.tabs.addTab(self._tab_index(), "İndeks")
        self.tabs.addTab(self._tab_update(), "Güncelleme")
        self.tabs.addTab(self._tab_cache(), "Önbellek")
        layout.addWidget(self.tabs)

    def _tab_sources(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.sources_panel = SourcesPanel()
        lay.addWidget(self.sources_panel)
        btn_row = QHBoxLayout()
        self.btn_select_all = QPushButton("Tümünü Seç")
        self.btn_select_none = QPushButton("Tümünü Kaldır")
        self.btn_select_all.clicked.connect(self.sources_panel.select_all_sources)
        self.btn_select_none.clicked.connect(self.sources_panel.clear_source_checks)
        btn_row.addWidget(self.btn_select_all)
        btn_row.addWidget(self.btn_select_none)
        lay.addLayout(btn_row)
        self.sources_panel.add_source_clicked.connect(self.add_source_clicked.emit)
        self.sources_panel.edit_source_clicked.connect(self.edit_source_clicked.emit)
        self.sources_panel.toggle_active_clicked.connect(
            self.toggle_active_clicked.emit
        )
        self.sources_panel.detach_source_clicked.connect(
            self.detach_source_clicked.emit
        )
        self.sources_panel.purge_source_clicked.connect(self.purge_source_clicked.emit)
        self.sources_panel.purge_orphans_clicked.connect(self.purge_orphans_clicked.emit)
        self.sources_panel.scan_source_clicked.connect(self.scan_source_clicked.emit)
        self.sources_panel.scan_all_clicked.connect(self.scan_all_clicked.emit)
        self.sources_panel.scan_due_clicked.connect(self.scan_due_clicked.emit)
        self.sources_panel.source_selection_changed.connect(
            self.source_selection_changed.emit
        )
        self.sources_panel.source_checks_changed.connect(
            self.source_checks_changed.emit
        )
        return w

    def _tab_index(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.progress_panel = ProgressPanel()
        self.progress_panel.index_start_clicked.connect(self.index_start_clicked.emit)
        self.progress_panel.index_stop_clicked.connect(self.index_stop_clicked.emit)
        self.progress_panel.index_pause_clicked.connect(self.index_pause_clicked.emit)
        self.progress_panel.index_resume_clicked.connect(self.index_resume_clicked.emit)
        self.progress_panel.index_fast_archive_clicked.connect(
            self.index_fast_archive_clicked.emit
        )
        self.progress_panel.index_night_complete_clicked.connect(
            self.index_night_complete_clicked.emit
        )
        self.progress_panel.index_backfill_clicked.connect(
            self.index_backfill_clicked.emit
        )
        self.progress_panel.index_purge_broken_clicked.connect(
            self.index_purge_broken_clicked.emit
        )
        self.progress_panel.index_requeue_thumbs_clicked.connect(
            self.index_requeue_thumbs_clicked.emit
        )
        self.progress_panel.index_queue_ai_clicked.connect(
            self.index_queue_ai_clicked.emit
        )
        self.progress_panel.semantic_backfill_clicked.connect(
            self.semantic_backfill_clicked.emit
        )
        lay.addWidget(self.progress_panel)
        return w

    def _tab_update(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.chk_auto_scan = QCheckBox("Açılışta otomatik tara")
        lay.addWidget(self.chk_auto_scan)
        self.chk_folder_watch = QCheckBox(
            "Klasör değişikliklerini izle (ekle/sil → hızlı tara)"
        )
        self.chk_folder_watch.setToolTip(
            "Kaynak kök klasörlerinde dosya eklenince veya silinince birkaç saniye içinde "
            "hızlı index tetiklenir. Ağ paylaşımlarında her zaman çalışmayabilir."
        )
        lay.addWidget(self.chk_folder_watch)
        self.chk_purge_missing = QCheckBox(
            "Taramadan sonra silinmiş kayıtları DB'den temizle"
        )
        self.chk_purge_missing.setToolTip(
            "Diskte artık olmayan dosyalar önce 'missing' işaretlenir, ardından index kaydı tamamen silinir."
        )
        lay.addWidget(self.chk_purge_missing)
        self.btn_purge_missing = QPushButton("Silinmiş Kayıtları Şimdi Temizle")
        self.btn_purge_missing.clicked.connect(self.purge_missing_clicked.emit)
        lay.addWidget(self.btn_purge_missing)
        self.lbl_missing_count = QLabel("Bekleyen silinmiş kayıt: —")
        self.lbl_missing_count.setWordWrap(True)
        self.lbl_missing_count.setStyleSheet("color: #94a3b8;")
        lay.addWidget(self.lbl_missing_count)
        self.lbl_skip_info = QLabel("Önceden indekslenenler hızlı geçilir.")
        self.lbl_skip_info.setWordWrap(True)
        self.lbl_skip_info.setStyleSheet("color: #94a3b8;")
        lay.addWidget(self.lbl_skip_info)
        self.lbl_last_stats = QLabel("Son tarama: —")
        self.lbl_last_stats.setWordWrap(True)
        lay.addWidget(self.lbl_last_stats)
        lay.addStretch()
        return w

    def _tab_cache(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_thumb_count = QLabel("Küçük görsel: —")
        self.lbl_missing_thumb = QLabel("Eksik küçük görsel: —")
        self.lbl_missing_texture = QLabel("Eksik doku haritası: —")
        self.lbl_missing_patch = QLabel("Eksik parça vektörü: —")
        self.lbl_missing_preview = QLabel("Eksik özellik önizlemesi: —")
        self.lbl_missing_ai = QLabel("Eksik yapay zekâ vektörü: —")
        self.lbl_cache_size = QLabel("Cache boyutu: —")
        for lbl in (
            self.lbl_thumb_count,
            self.lbl_missing_thumb,
            self.lbl_missing_texture,
            self.lbl_missing_patch,
            self.lbl_missing_preview,
            self.lbl_missing_ai,
            self.lbl_cache_size,
        ):
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        btn_row = QHBoxLayout()
        self.btn_cache_refresh = QPushButton("Yenile")
        self.btn_cache_fill = QPushButton("Eksikleri Tamamla")
        self.btn_cache_clean = QPushButton("Önbelleği Temizle")
        self.btn_cache_refresh.clicked.connect(self.cache_refresh_clicked.emit)
        self.btn_cache_fill.clicked.connect(self.cache_fill_missing_clicked.emit)
        self.btn_cache_clean.clicked.connect(self.cache_clean_clicked.emit)
        btn_row.addWidget(self.btn_cache_refresh)
        btn_row.addWidget(self.btn_cache_fill)
        btn_row.addWidget(self.btn_cache_clean)
        lay.addLayout(btn_row)
        lay.addStretch()
        return w

    def set_auto_scan(self, enabled: bool) -> None:
        self.chk_auto_scan.setChecked(enabled)
        self.sources_panel.set_auto_scan(enabled)

    def is_auto_scan_enabled(self) -> bool:
        return self.chk_auto_scan.isChecked()

    def set_folder_watch(self, enabled: bool) -> None:
        self.chk_folder_watch.setChecked(enabled)

    def is_folder_watch_enabled(self) -> bool:
        return self.chk_folder_watch.isChecked()

    def set_purge_missing_after_scan(self, enabled: bool) -> None:
        self.chk_purge_missing.setChecked(enabled)

    def is_purge_missing_after_scan_enabled(self) -> bool:
        return self.chk_purge_missing.isChecked()

    def update_missing_count(self, count: int) -> None:
        self.lbl_missing_count.setText(f"Bekleyen silinmiş kayıt: {count:,}")

    def update_cache_info(self, status: dict) -> None:
        indexed = status.get("indexed", 0)
        missing_tex = status.get("missing_texture_maps", 0)
        self.lbl_thumb_count.setText(f"Küçük görsel (indeksli): {indexed:,}")
        self.lbl_missing_thumb.setText("Eksik thumbnail: —")
        self.lbl_missing_thumb.setText(
            f"Eksik küçük görsel: {status.get('missing_thumbnails', 0):,}"
        )
        self.lbl_missing_texture.setText(f"Eksik texture map: {missing_tex:,}")
        self.lbl_missing_patch.setText(
            f"Eksik patch: {status.get('missing_patches', 0):,}"
        )
        self.lbl_missing_preview.setText(
            f"Eksik feature preview: {status.get('missing_feature_previews', 0):,}"
        )
        self.lbl_missing_ai.setText(
            f"Eksik AI embedding: {status.get('missing_ai_dino', 0) + status.get('missing_ai_clip', 0):,}"
        )
        cache_bytes = status.get("cache_size_bytes")
        if cache_bytes is None:
            self.lbl_cache_size.setText("Cache boyutu: yenile ile hesaplanır")
        else:
            size_mb = float(cache_bytes or 0) / (1024 * 1024)
            self.lbl_cache_size.setText(f"Cache boyutu: {size_mb:.1f} MB")

    def update_scan_stats(self, stats: dict) -> None:
        skipped = stats.get("skipped_fast", 0)
        new = stats.get("new", 0)
        changed = stats.get("changed", 0)
        tex = stats.get("texture_updated", 0)
        self.lbl_last_stats.setText(
            f"Hızlı geçilen: {skipped:,} | Yeni: {new:,} | Değişen: {changed:,} | "
            f"Texture güncellenen: {tex:,}"
        )
