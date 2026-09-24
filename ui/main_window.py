"""Ana pencere — profesyonel dashboard mimarisi."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.background_index import set_search_active
from core.background_index import status_label as index_status_label
from core.dynamic_groups import resolve_query_context_from_meta
from core.index_queue_manager import IndexQueueManager
from core.logger import setup_logger
from core.scan_scheduler import ScanScheduler
from core.search_models import SearchQuery, SearchResponse
from core.search_presets import PRESET_BY_KEY, high_threshold_warning
from core.search_report import save_search_report
from core.settings import APP_NAME, AppSettings
from core.ui_perf import UiPerfMonitor
from core.sources import SEARCH_SCOPE_LABELS, SearchScope, SourceManager
from ui.category_tree_panel import CategoryTreePanel
from ui.filter_panel import FilterPanel
from ui.folder_watch_controller import FolderWatchController
from ui.format_status_panel import FormatStatusPanel
from ui.health_panel import HealthPanel
from ui.fixed_hover_preview import InspectorDockContent
from ui.inspector_panel import InspectorPanel
from ui.layout_state import reset_layout, restore_layout, save_layout
from ui.query_image_panel import QueryImagePanel
from ui.results_panel import ResultsPanel
from ui.resume_index_dialog import ResumeIndexDialog
from ui.search_header import SearchHeader
from ui.simple_mode_helpers import simple_header_line, simple_status_line
from ui.source_dialog import SourceDialog
from ui.source_sidebar import SourceSidebar
from ui.teach_me_panel import TeachMePanel
from ui.status_bar import StatusBarWidget
from ui.thumbnail_scheduler import get_thumbnail_scheduler
from ui.theme import apply_theme
from ui.worker_threads import (
    BackgroundTask,
    CacheReconciliationWorker,
    IndexWorker,
    PurgeMissingWorker,
    PurgeSourceWorker,
    QuickIndexWorker,
    SearchWorker,
    SourceFileCountWorker,
    StatusWorker,
)

logger = setup_logger(__name__)

MIN_WIDTH = 1050

MIN_HEIGHT = 680

# JPG/PNG/TIFF → os.startfile. EPS/PDF/AI Explorer /select (mevcut güvenli davranış).
_DEFAULT_APP_OPEN_EXTS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
)


def _opens_with_default_app(path: str) -> bool:
    return Path(path).suffix.lower() in _DEFAULT_APP_OPEN_EXTS


class _ToolbarCompat:
    """Geriye uyumluluk — header + filter panel birleşik erişim."""

    def __init__(self, header: SearchHeader, filt: FilterPanel):
        self._header = header
        self._filt = filt

    def __getattr__(self, name):
        if hasattr(self._header, name):
            return getattr(self._header, name)
        return getattr(self._filt, name)

    def set_searching(self, active: bool) -> None:
        self._header.set_searching(active)


class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        self.settings = AppSettings.load()

        self._index_worker: IndexWorker | None = None
        self._index_workers: dict[str, IndexWorker] = {}
        self._source_fast_workers: dict[int, IndexWorker] = {}
        self._source_count_workers: dict[int, SourceFileCountWorker] = {}
        self._source_fast_progress_at: dict[int, float] = {}
        self._source_fast_shown: dict[int, int] = {}
        self._source_live_stats: dict[int, dict] = {}

        self._search_worker: SearchWorker | None = None
        self._pending_search: tuple | None = None  # (query, instant)
        self._search_finish_hooked = False

        self._quick_index_worker: QuickIndexWorker | None = None
        self._pending_quick_index: str | None = None
        self._quick_finish_hooked = False
        self._pending_source_counts: dict[int, str] = {}

        self._status_worker: StatusWorker | None = None
        self._lane_status_worker: StatusWorker | None = None
        self._status_generation = 0
        self._status_refresh_queued = False
        self._cache_reconciliation_worker: CacheReconciliationWorker | None = None
        self._cache_reconciliation_stats: dict = {}

        self._purge_worker: PurgeSourceWorker | None = None
        self._purge_missing_worker: PurgeMissingWorker | None = None
        self._background_tasks: set[BackgroundTask] = set()
        self._feedback_shortcuts: list[dict[str, str]] = []
        self._pending_watch_scans: set[int] = set()

        self._last_query_path: str | None = None
        self._query_image_paths: list[str] = []
        self._suppress_instant_search = False

        self._search_response = None

        self._display_count = 100

        self._use_crop = False
        self._pending_instant_use_crop = False

        self._shutting_down = False
        self._resume_index_suppressed = False
        self._rerun_after_index = False
        self._index_paused = False
        self._current_index_mode = "standard"
        self._pending_index_restart = None

        self._filter_timer = QTimer(self)

        self._filter_timer.setSingleShot(True)

        self._filter_timer.setInterval(300)

        self._filter_timer.timeout.connect(self._refresh_results_display)

        self._instant_search_timer = QTimer(self)
        self._instant_search_timer.setSingleShot(True)
        self._instant_search_timer.setInterval(60)
        self._instant_search_timer.timeout.connect(self._run_pending_instant_search)

        self._index_progress_timer = QTimer(self)
        self._index_progress_timer.setInterval(1000)
        self._index_progress_timer.timeout.connect(self._refresh_status_during_index)
        self._index_active = False
        self._searchable_count = 0
        self._index_percent = 0
        self._search_active = False
        self._search_result_count = 0
        self._search_refining = False
        self._search_phase = ""
        self._search_kind = ""
        self._empty_search_result = False
        self._status_loaded = False
        self._last_status: dict = {}
        self._startup_probe_running = False
        self._customer_load_running = False
        self._recent_customers: list[str] = []
        self._customer_registry = None

        # Keep construction cheap; DB open/migrate deferred until first source use.
        self._source_manager = SourceManager(
            self.settings, run_maintenance=False, defer_db=True
        )
        self._sources_ready = False

        self._query_detail_splitter: QSplitter | None = None
        self._content_splitter: QSplitter | None = None
        self._source_dock: QDockWidget | None = None
        self._filter_dock: QDockWidget | None = None
        self._inspector_dock: QDockWidget | None = None
        self._format_dock: QDockWidget | None = None
        self._health_dock: QDockWidget | None = None
        self._category_dock: QDockWidget | None = None
        self._teach_me_dock: QDockWidget | None = None
        self._panel_actions: dict[str, QAction] = {}
        self._panel_visibility: dict[str, bool] = {
            "query": True,
            "filters": False,
            "source": False,
            "inspector": False,
            "format": False,
            "health": False,
            "category": False,
            "teach_me": False,
        }
        self._category_filter_path = ""
        self._main_splitter: QSplitter | None = None
        self._center_splitter: QSplitter | None = None
        self._ui_perf: UiPerfMonitor | None = None
        self._thumb_scheduler = None

        self._build_ui()

        self._connect_signals()

        self._refresh_results_timer = QTimer(self)
        self._refresh_results_timer.setSingleShot(True)
        self._refresh_results_timer.setInterval(40)
        self._refresh_results_timer.timeout.connect(self._do_refresh_results_display)
        self._pending_refresh: dict = {}

        self._restore_ui_state()

        self._load_customers()

        QTimer.singleShot(0, self._warmup_sources_async)
        QTimer.singleShot(100, self._refresh_status)
        QTimer.singleShot(200, self._preload_feedback_shortcuts)
        QTimer.singleShot(300, self._check_resume_index_on_startup)

        if getattr(self.settings, "resource_monitor_enabled", True):
            QTimer.singleShot(400, self._start_health_monitor)
        QTimer.singleShot(1500, self._start_cache_reconciliation)

        if self.settings.auto_scan_on_startup:
            QTimer.singleShot(1000, self._start_scheduler_safe)

    # ── Geriye uyumluluk (testler / eski referanslar) ──

    @property
    def toolbar(self):
        return self._toolbar_compat

    @property
    def query_panel(self):
        return self.query_image_panel.query_panel

    @property
    def summary_panel(self):
        return self.filter_panel.summary_panel

    @property
    def slider_threshold(self):

        return self.toolbar.slider_threshold

    @property
    def lbl_threshold_count(self):

        return self.toolbar.lbl_threshold_count

    @property
    def txt_search(self):

        return self.toolbar.txt_search

    @property
    def cmb_limit(self):

        return self.toolbar.cmb_limit

    @property
    def cmb_scope(self):

        return self.toolbar.cmb_scope

    @property
    def cmb_customer(self):

        return self.toolbar.cmb_customer

    @property
    def cmb_mode(self):

        return self.toolbar.cmb_mode

    @property
    def cmb_color(self):

        return self.toolbar.cmb_color

    @property
    def cmb_search_method(self):

        return self.toolbar.cmb_search_method

    @property
    def chk_near_below(self):

        return self.toolbar.chk_near_below

    @property
    def sources_panel(self):

        return self.source_sidebar.sources_panel

    @property
    def progress_panel(self):

        return self.source_sidebar.progress_panel

    @property
    def detail_panel(self):

        return self.inspector_panel

    def _build_ui(self) -> None:

        self.setWindowTitle(f"{APP_NAME} — Tekstil Desen Görsel Arama")

        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

        central = QWidget()

        root = QVBoxLayout(central)

        root.setContentsMargins(0, 0, 0, 0)

        root.setSpacing(4)

        # Üst durum satırı: basit kullanıcı ekranını kirletmeden genel durumu gösterir.
        status_row = QHBoxLayout()

        status_row.setContentsMargins(12, 4, 12, 0)

        self.lbl_header_mode = QLabel("Basit Mod")

        self.lbl_header_mode.setStyleSheet("color:#94a3b8;")

        self.lbl_header_status = QLabel("İndeks: Hesaplanıyor…")

        self.lbl_header_status.setStyleSheet("color:#60a5fa;")

        status_row.addWidget(QLabel("<b>Vezir Pattern Search</b>"))

        status_row.addStretch()

        status_row.addWidget(self.lbl_header_mode)

        status_row.addWidget(QLabel("  |  "))

        status_row.addWidget(self.lbl_header_status)

        root.addLayout(status_row)

        # 1) Ana kullanıcı arama alanı.
        self.search_header = SearchHeader()

        preset_index = self.search_header.cmb_preset.findData(
            self.settings.search_preset
        )
        if preset_index >= 0:
            self.search_header.cmb_preset.setCurrentIndex(preset_index)

        root.addWidget(self.search_header)

        # 2) Filtre/istatistik alanı. Varsayılan Basit Mod'da gizli;
        # kullanıcı "Filtreler" ile açar. Nesne var kalır, eski bağlantılar bozulmaz.
        self.filter_panel = FilterPanel(self.settings)

        self.filter_panel.setVisible(True)

        self._toolbar_compat = _ToolbarCompat(self.search_header, self.filter_panel)

        # 3) Ana alan: sonuçlar merkezde; seçilen desen dock olarak taşınabilir.
        self.results_panel = ResultsPanel()
        self._thumb_scheduler = get_thumbnail_scheduler(self)
        from core.thumbnailer import Thumbnailer

        self._thumb_scheduler.configure(
            cache_dir=self.settings.cache_dir,
            thumbnailer=Thumbnailer(
                self.settings.cache_dir,
                self.settings.thumbnail_max_edge,
                self.settings.thumbnail_format,
            ),
        )
        # Startup: queue detail cache-miss work only after first interactive frame.
        self._thumb_scheduler._detail_ui_ready = False
        QTimer.singleShot(0, self._mark_preview_scheduler_interactive)
        QTimer.singleShot(250, self._mark_preview_scheduler_interactive)
        self.results_panel.set_thumbnail_scheduler(self._thumb_scheduler)
        import os as _os

        if _os.environ.get("VEZIR_QA_NONINTERACTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            self._ui_perf = None
        else:
            self._ui_perf = UiPerfMonitor(
                log_dir=Path(self.settings.db_path).parent / "logs",
                parent=self,
            )
            self._ui_perf.updated.connect(self._on_ui_perf_updated)
            self._thumb_scheduler.thumbnail_ready.connect(
                lambda *_a: self._ui_perf.record_thumbnail_loaded()
                if self._ui_perf
                else None
            )
        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._content_splitter.setObjectName("ContentSplitter")
        self._content_splitter.addWidget(self.results_panel)
        self._content_splitter_compat = QWidget()
        self._content_splitter.addWidget(self._content_splitter_compat)
        self._content_splitter_compat.hide()
        root.addWidget(self._content_splitter, stretch=1)

        self.status_bar = StatusBarWidget()
        root.addWidget(self.status_bar)

        self.setCentralWidget(central)

        self.query_image_panel = QueryImagePanel()
        self._query_dock = QDockWidget("Seçilen Desen", self)
        self._query_dock.setObjectName("QueryDock")
        self._query_dock.setWidget(self.query_image_panel)
        self._query_dock.setMinimumWidth(280)
        self._query_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._query_dock)

        self._filter_dock = QDockWidget("Arama & AI Ayarları", self)
        self._filter_dock.setObjectName("FilterDock")
        self._filter_dock.setWidget(self._scrollable_panel(self.filter_panel))
        self._configure_dock(self._filter_dock, minimum_width=340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._filter_dock)

        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )

        # Eski layout_state/test isimleri için uyumluluk.
        self._query_detail_splitter = None
        self._main_splitter = self._content_splitter
        self._center_splitter = None

        # Gelişmiş kaynak/index paneli artık varsayılan olarak kapalı.
        self.source_sidebar = SourceSidebar()

        self.source_sidebar.set_auto_scan(self.settings.auto_scan_on_startup)
        self.source_sidebar.set_folder_watch(self.settings.folder_watch_enabled)
        self.source_sidebar.set_purge_missing_after_scan(
            self.settings.purge_missing_after_scan
        )

        self._folder_watch = FolderWatchController(
            self.settings,
            self._source_manager,
            parent=self,
        )

        self._source_dock = QDockWidget("Kaynaklar ve İndeks", self)

        self._source_dock.setObjectName("SourceDock")

        self._source_dock.setWidget(self._scrollable_panel(self.source_sidebar))

        self._configure_dock(self._source_dock, minimum_width=320)

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._source_dock)

        # Sonuç detay paneli sağ dock olarak açılıp kapanır. İlk açılışta kapalıdır;
        # kullanıcı sonuç seçince otomatik açılır.
        self._inspector_content = InspectorDockContent()
        self.inspector_panel = self._inspector_content.inspector_panel
        self.inspector_panel.set_db_path(self.settings.db_path)
        self._hover_preview_panel = self._inspector_content.hover_preview

        self._inspector_dock = QDockWidget("Sonuç Detayı", self)

        self._inspector_dock.setObjectName("InspectorDock")

        self._inspector_dock.setWidget(self._scrollable_panel(self._inspector_content))

        self._configure_dock(self._inspector_dock, minimum_width=360)
        self._inspector_dock.setMaximumWidth(540)
        self._inspector_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._inspector_dock)
        self._place_inspector_in_detail_slot(force_resize=True)

        self.format_status_panel = FormatStatusPanel()
        self.format_status_panel.set_db_path(self.settings.db_path)
        self._format_dock = QDockWidget("Format Durumu", self)
        self._format_dock.setObjectName("FormatStatusDock")
        self._format_dock.setWidget(self._scrollable_panel(self.format_status_panel))
        self._configure_dock(self._format_dock, minimum_width=320)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._format_dock)

        self.health_panel = HealthPanel(self.settings)
        self._health_dock = QDockWidget("Sistem Sağlığı", self)
        self._health_dock.setObjectName("HealthDock")
        self._health_dock.setWidget(self._scrollable_panel(self.health_panel))
        self._configure_dock(self._health_dock, minimum_width=360)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._health_dock)

        self.category_tree_panel = CategoryTreePanel()
        self.category_tree_panel.set_db_path(self.settings.db_path)
        self._category_dock = QDockWidget("Kategori Filtresi", self)
        self._category_dock.setObjectName("CategoryTreeDock")
        self._category_dock.setWidget(self._scrollable_panel(self.category_tree_panel))
        self._configure_dock(self._category_dock, minimum_width=240)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._category_dock)

        self.teach_me_panel = TeachMePanel(self.settings)
        if self._thumb_scheduler is not None:
            self.teach_me_panel.set_thumbnail_scheduler(self._thumb_scheduler)
        self._teach_me_dock = QDockWidget("Bana Öğret", self)
        self._teach_me_dock.setObjectName("TeachMeDock")
        self._teach_me_dock.setWidget(self._scrollable_panel(self.teach_me_panel))
        self._configure_dock(self._teach_me_dock, minimum_width=360)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._teach_me_dock)

        self._source_dock.hide()

        self._filter_dock.hide()

        self._inspector_dock.hide()
        self._format_dock.hide()
        self._health_dock.hide()
        self._category_dock.hide()
        self._teach_me_dock.hide()

        self.query_image_panel.setAcceptDrops(True)

        self._init_panel_menu()

        self._source_dock.visibilityChanged.connect(self._on_source_dock_visibility)
        self._filter_dock.visibilityChanged.connect(self._on_filter_dock_visibility)
        self._inspector_dock.visibilityChanged.connect(
            self._on_inspector_dock_visibility
        )
        if self._format_dock:
            self._format_dock.visibilityChanged.connect(self._on_format_dock_visibility)
        if self._health_dock:
            self._health_dock.visibilityChanged.connect(self._on_health_dock_visibility)
        if self._category_dock:
            self._category_dock.visibilityChanged.connect(
                self._on_category_dock_visibility
            )
        if self._teach_me_dock:
            self._teach_me_dock.visibilityChanged.connect(
                self._on_teach_me_dock_visibility
            )
        self.teach_me_panel.taught.connect(self._on_teach_me_taught)
        if getattr(self, "_query_dock", None):
            self._query_dock.visibilityChanged.connect(self._on_query_dock_visibility)

        for dock in self.findChildren(QDockWidget):
            dock.dockLocationChanged.connect(lambda _area: self._save_ui_state())
            dock.topLevelChanged.connect(lambda _floating: self._save_ui_state())

    @staticmethod
    def _scrollable_panel(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    @staticmethod
    def _configure_dock(dock: QDockWidget, *, minimum_width: int) -> None:
        dock.setMinimumWidth(minimum_width)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

    def _place_inspector_in_detail_slot(self, *, force_resize: bool = False) -> None:
        """Keep result details in the dedicated right-side preview slot."""
        inspector = getattr(self, "_inspector_dock", None)
        if not inspector:
            return
        needs_move = (
            inspector.isFloating()
            or self.dockWidgetArea(inspector) != Qt.DockWidgetArea.RightDockWidgetArea
        )
        if inspector.isFloating():
            inspector.setFloating(False)
        if needs_move:
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, inspector)
        if force_resize or needs_move:
            target_width = min(500, max(380, self.width() // 3))
            self.resizeDocks(
                [inspector], [target_width], Qt.Orientation.Horizontal
            )

    def _on_query_dock_visibility(self, visible: bool) -> None:
        self._panel_visibility["query"] = visible
        self._sync_panel_menu()
        self._save_ui_state()

    def _init_panel_menu(self) -> None:
        menu = self.search_header.panels_menu
        menu.clear()
        self._panel_actions.clear()
        panel_defs = [
            ("query", "Seçilen Desen", "Sorgu görseli ve crop alanı"),
            ("filters", "Arama & AI Ayarları", "Eşik, mod, AI/OCR ve istatistikler"),
            ("category", "Kategori Filtresi", "Sol kategori ağacı ile sonuç daraltma"),
            ("format", "Format Durumu", "TIF/PSD/PDF desteği ve bağımlılıklar"),
            ("health", "Sistem Sağlığı", "CPU/RAM/GPU, kuyruk, doğrulama ve benchmark"),
            ("source", "Kaynaklar ve İndeks", "Kaynak yönetimi ve indeks işlemleri"),
            ("inspector", "Sonuç Detayı", "Seçili sonucun önizleme ve skor detayı"),
            ("teach_me", "Bana Öğret", "Şüpheli görselleri toplu öğret"),
        ]
        for key, label, tip in panel_defs:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setToolTip(tip)
            act.triggered.connect(
                lambda checked, k=key: self._set_panel_visible(k, checked)
            )
            menu.addAction(act)
            self._panel_actions[key] = act
        menu.addSeparator()
        show_all = menu.addAction("Tüm Panelleri Göster")
        show_all.triggered.connect(self._show_all_panels)
        menu.addSeparator()
        reset_simple = menu.addAction("Basit Düzeni Sıfırla")
        reset_simple.triggered.connect(lambda: self._reset_layout_for_mode("simple"))
        reset_advanced = menu.addAction("Gelişmiş Düzeni Sıfırla")
        reset_advanced.triggered.connect(
            lambda: self._reset_layout_for_mode("advanced")
        )

    def _show_all_panels(self) -> None:
        for key in ("query", "filters", "category", "format", "health", "source", "inspector", "teach_me"):
            self._set_panel_visible(key, True)

    def _reset_layout_for_mode(self, mode: str) -> None:
        if mode == self.settings.ui_mode:
            self._reset_panel_layout()
            return
        ui = dict(self.settings.ui_state or {})
        ui.pop(f"window_state_{mode}", None)
        ui.pop(f"panel_visibility_{mode}", None)
        self.settings.ui_state = ui
        self.settings.save()
        label = "Basit" if mode == "simple" else "Gelişmiş"
        self.status_bar.set_message(f"{label} Mod düzeni sıfırlandı")

    def _set_panel_visible(self, panel_key: str, visible: bool) -> None:
        self._panel_visibility[panel_key] = visible
        if panel_key == "query":
            if getattr(self, "_query_dock", None):
                self._query_dock.setVisible(visible)
                if visible:
                    self._query_dock.raise_()
            else:
                self.query_image_panel.setVisible(visible)
        elif panel_key == "filters":
            if self._filter_dock:
                self._filter_dock.setVisible(visible)
                if visible:
                    self._filter_dock.raise_()
            self.search_header.btn_filters.setText(
                "Filtreleri Gizle" if visible else "Filtreler"
            )
        elif panel_key == "source" and self._source_dock:
            self._source_dock.setVisible(visible)
            if visible:
                self._source_dock.raise_()
        elif panel_key == "inspector" and self._inspector_dock:
            if visible:
                self._place_inspector_in_detail_slot()
            self._inspector_dock.setVisible(visible)
            if visible:
                self._inspector_dock.raise_()
                QTimer.singleShot(
                    0,
                    lambda: self._place_inspector_in_detail_slot(
                        force_resize=True
                    ),
                )
        elif panel_key == "format" and self._format_dock:
            self._format_dock.setVisible(visible)
            if visible:
                self._format_dock.raise_()
                QTimer.singleShot(0, self.format_status_panel.refresh)
        elif panel_key == "health" and self._health_dock:
            self._health_dock.setVisible(visible)
            if visible:
                self._health_dock.raise_()
                QTimer.singleShot(0, self.health_panel._refresh_metrics)
        elif panel_key == "category" and self._category_dock:
            self._category_dock.setVisible(visible)
            if visible:
                self._category_dock.raise_()
                # Rebuild only when DB path changed or tree empty — never
                # force refresh on every open (UI freeze: N+1 concepts()).
                self.category_tree_panel.set_db_path(self.settings.db_path)
        elif panel_key == "teach_me" and self._teach_me_dock:
            self._teach_me_dock.setVisible(visible)
            if visible:
                self._teach_me_dock.raise_()
        self._sync_panel_menu()
        self._save_ui_state()

    def _sync_panel_menu(self) -> None:
        mapping = dict(self._panel_visibility)
        for key, visible in mapping.items():
            act = self._panel_actions.get(key)
            if act and act.isChecked() != visible:
                act.blockSignals(True)
                act.setChecked(visible)
                act.blockSignals(False)

    def _on_source_dock_visibility(self, visible: bool) -> None:
        self._panel_visibility["source"] = visible
        self._sync_panel_menu()
        self._save_ui_state()
        if visible:
            if self._last_status:
                self.progress_panel.update_status(self._last_status)
            self._refresh_status()

    def _on_filter_dock_visibility(self, visible: bool) -> None:
        self._panel_visibility["filters"] = visible
        self._sync_panel_menu()
        self._save_ui_state()

    def _on_inspector_dock_visibility(self, visible: bool) -> None:
        self._panel_visibility["inspector"] = visible
        self._sync_panel_menu()
        self._save_ui_state()

    def _on_format_dock_visibility(self, visible: bool) -> None:
        self._panel_visibility["format"] = visible
        self._sync_panel_menu()
        self._save_ui_state()

    def _on_health_dock_visibility(self, visible: bool) -> None:
        self._panel_visibility["health"] = visible
        self._sync_panel_menu()
        self._save_ui_state()

    def _on_category_dock_visibility(self, visible: bool) -> None:
        self._panel_visibility["category"] = visible
        self._sync_panel_menu()
        self._save_ui_state()

    def _on_teach_me_dock_visibility(self, visible: bool) -> None:
        # Modal Kaydet/önizleme diyaloğu Windows'ta dock'u gizlenmiş gibi bildirir.
        if not visible and QApplication.activeModalWidget() is not None:
            return
        self._panel_visibility["teach_me"] = visible
        self._sync_panel_menu()
        self._save_ui_state()
        if visible:
            QTimer.singleShot(0, self.teach_me_panel.reload)

    def _on_teach_me_taught(self, label: str, count: int) -> None:
        name = str(label or "").strip()
        n = int(count or 0)
        if n <= 0:
            return
        self.status_bar.set_message(f"{n} görsel «{name}» olarak öğretildi")

    def _apply_panel_setting_visibility(self) -> None:
        """Panel bayrağı gerçekten değiştiyse dock görünürlüğünü uygula.

        Böylece OCR/AI gibi alakasız ayarlar kaydedilince format/kategori
        paneli kendiliğinden açılıp kapanmaz.
        """
        fmt_enabled = bool(self.settings.format_panel_enabled)
        cat_enabled = bool(self.settings.category_tree_panel_enabled)
        prev_fmt = getattr(self, "_prev_format_panel_enabled", None)
        prev_cat = getattr(self, "_prev_category_panel_enabled", None)
        self._prev_format_panel_enabled = fmt_enabled
        self._prev_category_panel_enabled = cat_enabled
        if prev_fmt is not None and fmt_enabled != prev_fmt and self._format_dock:
            self._set_panel_visible("format", fmt_enabled)
        if prev_cat is not None and cat_enabled != prev_cat and self._category_dock:
            self._set_panel_visible("category", cat_enabled)

    def _restore_ui_state(self) -> None:

        restore_layout(self.settings, self)

        ui = getattr(self.settings, "ui_state", None) or {}
        advanced = self.settings.ui_mode == "advanced"

        mode_visibility = ui.get(f"panel_visibility_{self.settings.ui_mode}", {})
        if not isinstance(mode_visibility, dict):
            mode_visibility = {}

        self._panel_visibility["source"] = bool(
            mode_visibility.get("source", ui.get("source_dock_visible", False))
        )
        self._panel_visibility["inspector"] = bool(
            mode_visibility.get("inspector", ui.get("inspector_dock_visible", False))
        )
        self._panel_visibility["format"] = bool(
            mode_visibility.get("format", ui.get("format_dock_visible", False))
        )
        self._panel_visibility["health"] = bool(
            mode_visibility.get("health", ui.get("health_dock_visible", False))
        )
        self._panel_visibility["category"] = bool(
            mode_visibility.get("category", ui.get("category_dock_visible", False))
        )
        self._panel_visibility["teach_me"] = bool(
            mode_visibility.get("teach_me", ui.get("teach_me_dock_visible", False))
        )
        self._panel_visibility["filters"] = bool(
            mode_visibility.get("filters", ui.get("filters_visible", False))
        )
        query_visible = ui.get("query_panel_visible")
        self._panel_visibility["query"] = bool(
            mode_visibility.get(
                "query",
                True if query_visible is None else query_visible,
            )
        )

        if self._source_dock:
            self._source_dock.setVisible(self._panel_visibility["source"])

        if hasattr(self, "_inspector_dock") and self._inspector_dock:
            self._inspector_dock.setVisible(self._panel_visibility["inspector"])
            self._place_inspector_in_detail_slot(force_resize=True)

        if self._format_dock:
            show_format = self._panel_visibility["format"] or bool(
                self.settings.format_panel_enabled
                and ui.get("format_dock_visible") is None
            )
            self._format_dock.setVisible(show_format)
            self._panel_visibility["format"] = show_format

        if self._health_dock:
            self._health_dock.setVisible(self._panel_visibility.get("health", False))

        if self._category_dock:
            show_cat = self._panel_visibility["category"] or bool(
                self.settings.category_tree_panel_enabled
                and ui.get("category_dock_visible") is None
            )
            self._category_dock.setVisible(show_cat)
            self._panel_visibility["category"] = show_cat

        if self._teach_me_dock:
            self._teach_me_dock.setVisible(self._panel_visibility.get("teach_me", False))

        if self._filter_dock:
            self._filter_dock.setVisible(self._panel_visibility["filters"])
        if getattr(self, "_query_dock", None):
            self._query_dock.setVisible(self._panel_visibility["query"])
        else:
            self.query_image_panel.setVisible(self._panel_visibility["query"])
        self.filter_panel.set_advanced_mode(advanced)
        self._apply_ui_mode_panels(advanced)

        self.search_header.set_simple_mode(not advanced)
        self.search_header.btn_mode.setText("Basit Mod" if advanced else "Gelişmiş Mod")
        self.lbl_header_mode.setText("Gelişmiş Mod" if advanced else "Basit Mod")

        if hasattr(self.search_header, "btn_filters"):
            self.search_header.btn_filters.setText(
                "Filtreleri Gizle"
                if self._filter_dock and self._filter_dock.isVisible()
                else "Filtreler"
            )

        self._sync_panel_menu()

        scope_idx = self.search_header.cmb_scope.findData(self.settings.search_scope)
        if scope_idx >= 0:
            self.search_header.cmb_scope.setCurrentIndex(scope_idx)
        self._repair_stale_search_scope()
        self._update_scope_hint()
        self._update_user_status_bar()

    def _save_ui_state(self) -> None:

        save_layout(self.settings, self)

        ui = getattr(self.settings, "ui_state", None) or {}

        if isinstance(ui, dict):
            ui["filters_visible"] = self._panel_visibility["filters"]

            ui["query_panel_visible"] = self._panel_visibility["query"]

            ui["source_dock_visible"] = self._panel_visibility["source"]

            ui["inspector_dock_visible"] = self._panel_visibility["inspector"]

            ui["format_dock_visible"] = self._panel_visibility["format"]

            ui["health_dock_visible"] = self._panel_visibility.get("health", False)

            ui["category_dock_visible"] = self._panel_visibility["category"]

            ui["teach_me_dock_visible"] = self._panel_visibility.get("teach_me", False)

            ui[f"panel_visibility_{self.settings.ui_mode}"] = dict(
                self._panel_visibility
            )

            self.settings.ui_state = ui

        self.settings.similarity_threshold = self.slider_threshold.value() / 100

        self.settings.search_result_limit = self.cmb_limit.currentData()

        self.settings.search_mode = self.cmb_mode.currentData()

        self.settings.color_weight_mode = self.cmb_color.currentData()

        self.settings.show_near_below_threshold = self.chk_near_below.isChecked()

        self.settings.save()

    def _connect_signals(self) -> None:

        self.source_sidebar.add_source_clicked.connect(self._add_source)

        self.source_sidebar.edit_source_clicked.connect(self._edit_source)

        self.source_sidebar.toggle_active_clicked.connect(self._toggle_source_active)

        self.source_sidebar.detach_source_clicked.connect(self._detach_source)

        self.source_sidebar.purge_source_clicked.connect(self._purge_source_index)

        self.source_sidebar.purge_orphans_clicked.connect(self._purge_orphan_indexes)

        self.source_sidebar.scan_source_clicked.connect(self._scan_source)

        self.progress_panel.full_scan_clicked.connect(self._start_manual_full_scan)
        self.progress_panel.full_scan_auto_changed.connect(self._on_full_scan_auto_changed)
        self.progress_panel.set_full_scan_options(
            enabled=bool(getattr(self.settings, "background_full_scan_enabled", True)),
            interval_min=int(
                getattr(self.settings, "background_full_scan_interval_min", 30) or 30
            ),
        )
        # Index kontrol butonları progress_panel'de de var — sidebar ile aynı slotlar.
        self.progress_panel.index_start_clicked.connect(
            lambda: self._start_index("quick", index_mode="complete")
        )
        self.progress_panel.index_fast_archive_clicked.connect(
            lambda: self._start_index("quick", index_mode="fast_archive")
        )
        self.progress_panel.index_night_complete_clicked.connect(
            lambda: self._start_index("quick", index_mode="night_complete")
        )
        self.progress_panel.index_backfill_clicked.connect(
            lambda: self._start_index("quick", index_mode="backfill")
        )
        self.progress_panel.index_purge_broken_clicked.connect(self._index_purge_broken)
        self.progress_panel.index_requeue_thumbs_clicked.connect(
            self._index_requeue_thumbs
        )
        self.progress_panel.index_queue_ai_clicked.connect(self._index_queue_ai)
        self.progress_panel.patch_start_clicked.connect(
            lambda: self._start_post_ga(patch=True, ocr=False)
        )
        self.progress_panel.ocr_start_clicked.connect(
            lambda: self._start_post_ga(patch=False, ocr=True)
        )
        self.progress_panel.patch_ocr_start_clicked.connect(
            lambda: self._start_post_ga(patch=True, ocr=True)
        )
        self.progress_panel.patch_stop_clicked.connect(
            lambda: self._stop_post_ga_lane("patch")
        )
        self.progress_panel.ocr_stop_clicked.connect(
            lambda: self._stop_post_ga_lane("ocr")
        )
        self.progress_panel.post_ga_rerun_clicked.connect(
            lambda: self._start_post_ga(patch=True, ocr=True, rerun=True)
        )
        self.progress_panel.semantic_backfill_clicked.connect(
            self._index_semantic_backfill
        )
        self.progress_panel.index_stop_clicked.connect(self._stop_index)
        self.progress_panel.index_pause_clicked.connect(self._pause_index)
        self.progress_panel.index_resume_clicked.connect(self._resume_index)

        self.source_sidebar.scan_all_clicked.connect(self._start_index)

        self.source_sidebar.scan_due_clicked.connect(self._scan_due)

        self.source_sidebar.source_selection_changed.connect(self._on_source_selected)

        self.source_sidebar.source_checks_changed.connect(
            self._on_source_checks_changed
        )

        self.source_sidebar.index_start_clicked.connect(
            lambda: self._start_index("quick", index_mode="complete")
        )

        self.source_sidebar.index_fast_archive_clicked.connect(
            lambda: self._start_index("quick", index_mode="fast_archive")
        )
        self.source_sidebar.index_night_complete_clicked.connect(
            lambda: self._start_index("quick", index_mode="night_complete")
        )
        self.source_sidebar.index_backfill_clicked.connect(
            lambda: self._start_index("quick", index_mode="backfill")
        )
        self.source_sidebar.index_purge_broken_clicked.connect(self._index_purge_broken)
        self.source_sidebar.index_requeue_thumbs_clicked.connect(
            self._index_requeue_thumbs
        )
        self.source_sidebar.index_queue_ai_clicked.connect(self._index_queue_ai)
        self.source_sidebar.semantic_backfill_clicked.connect(
            self._index_semantic_backfill
        )

        self.source_sidebar.index_stop_clicked.connect(self._stop_index)

        self.source_sidebar.index_pause_clicked.connect(self._pause_index)

        self.source_sidebar.index_resume_clicked.connect(self._resume_index)

        self.source_sidebar.cache_refresh_clicked.connect(self._refresh_status)

        self.source_sidebar.cache_clean_clicked.connect(self._cache_clean)

        self.source_sidebar.cache_fill_missing_clicked.connect(self._cache_fill_missing)

        self.source_sidebar.chk_auto_scan.stateChanged.connect(
            self._on_auto_scan_changed
        )
        self.source_sidebar.chk_folder_watch.stateChanged.connect(
            self._on_folder_watch_changed
        )
        self.source_sidebar.chk_purge_missing.stateChanged.connect(
            self._on_purge_missing_setting_changed
        )
        self.source_sidebar.purge_missing_clicked.connect(self._purge_missing_records)
        self._folder_watch.scan_requested.connect(self._on_folder_watch_scan)

        self.search_header.search_clicked.connect(self._run_current_search)

        self.search_header.pick_image_clicked.connect(self._pick_query_image)

        self.search_header.pick_files_clicked.connect(self._pick_query_files)

        self.search_header.pick_folder_clicked.connect(self._pick_folder)

        self.search_header.quick_folder_clicked.connect(self._quick_folder_search)

        self.search_header.clear_clicked.connect(self._clear_search)

        self.search_header.cancel_search_clicked.connect(self._cancel_search)

        self.search_header.toggle_filters_clicked.connect(self._toggle_filter_panel)

        self.search_header.toggle_sources_clicked.connect(self._toggle_source_dock)

        self.search_header.toggle_mode_clicked.connect(self._toggle_ui_mode)
        self.search_header.reset_layout_clicked.connect(self._reset_panel_layout)

        self.search_header.scope_changed.connect(self._on_scope_changed)

        self.search_header.preset_changed.connect(self._apply_search_preset)

        self.search_header.txt_search.returnPressed.connect(self._run_current_search)
        self.search_header.customer_changed.connect(self._on_header_customer_changed)
        self.search_header.txt_customer.textEdited.connect(
            self._on_customer_query_edited
        )
        self.search_header.txt_customer.returnPressed.connect(self._run_current_search)

        self.filter_panel.slider_threshold.valueChanged.connect(
            self._on_threshold_changed
        )

        self.filter_panel.filter_changed.connect(self._on_filter_changed)

        self.filter_panel.filter_research.connect(self._on_filter_changed_research)

        self.query_panel.search_full.connect(
            lambda: self._run_image_search(use_crop=False)
        )

        self.query_panel.search_crop.connect(
            lambda: self._run_image_search(use_crop=True)
        )

        self.query_panel.change_image_clicked.connect(self._pick_query_image)
        self.query_panel.clear_image_clicked.connect(self._clear_query_image_only)

        self.query_panel.image_selected.connect(self._schedule_instant_visual_search)

        self.query_panel.crop_region_selected.connect(
            self._schedule_instant_crop_search
        )

        self.query_panel.crop_cleared.connect(self._schedule_instant_visual_search)

        self.results_panel.result_selected.connect(self._show_result_detail)

        self.results_panel.open_folder.connect(self._open_folder)
        self.results_panel.open_file.connect(self._open_file)

        self.results_panel.load_more_clicked.connect(self._load_more_results)

        self.results_panel.group_filter_changed.connect(self._refresh_results_display)

        self.results_panel.view_mode_changed.connect(self._on_view_mode_changed)
        self.results_panel.hover_preview.connect(self._on_hover_preview)
        self.results_panel.hover_preview_clear.connect(self._on_hover_preview_clear)
        self.results_panel.preview_selection_changed.connect(
            self.inspector_panel.set_edit_target_count
        )
        self.results_panel.edit_selected_requested.connect(self._on_edit_result_metadata)

        self.inspector_panel.open_folder.connect(self._open_folder)

        self.inspector_panel.open_file.connect(self._open_file)

        self.inspector_panel.search_similar.connect(self._search_similar)

        self.filter_panel.close_requested.connect(
            lambda: self._set_panel_visible("filters", False)
        )

        self.filter_panel.reindex_requested.connect(self._on_reindex_requested)
        self.filter_panel.ai_enable_warning.connect(self._on_ai_enable_warning)
        self.filter_panel.panel_settings_changed.connect(
            self._on_panel_settings_changed
        )

        self.category_tree_panel.category_filter_changed.connect(
            self._on_category_filter_changed
        )

        self.health_panel.status_message.connect(self.status_bar.set_message)

        self.query_image_panel.close_requested.connect(
            lambda: self._set_panel_visible("query", False)
        )

        self.inspector_panel.feedback_clicked.connect(self._on_user_feedback)
        self.inspector_panel.teach_tag_clicked.connect(self._on_teach_tag)
        self.inspector_panel.learn_saved.connect(self._on_learn_saved)
        self.inspector_panel.ai_prediction_action.connect(self._on_ai_prediction_action)

    def _start_health_monitor(self) -> None:
        import os

        if os.environ.get("VEZIR_QA_NONINTERACTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        if self._shutting_down:
            return
        try:
            from core.production.health_monitor import start_health_monitor

            start_health_monitor(
                self.settings,
                interval_sec=5.0,
                on_snapshot=self.health_panel.apply_snapshot,
            )
            self.health_panel.start_monitoring()

            def _cache_opt():
                from core.production.cache_optimizer import optimize_cache

                return optimize_cache(self.settings)

            self._run_background_task(
                _cache_opt,
                working_message="",
                success_message="",
            )
        except Exception as exc:
            logger.debug("Health monitor start: %s", exc)

    def _start_cache_reconciliation(self) -> None:
        """V3 arka plan full scan (gap + silinen) — index drain'den bağımsız."""
        import os

        if os.environ.get("VEZIR_QA_NONINTERACTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        if self._shutting_down:
            return
        if not bool(getattr(self.settings, "background_full_scan_enabled", True)):
            logger.info("Background full scan disabled")
            return
        if int(getattr(self.settings, "background_full_scan_interval_min", 30) or 0) <= 0:
            logger.info("Background full scan interval=0 (manual only)")
            return
        old = self._cache_reconciliation_worker
        if old is not None and old.isRunning():
            return
        if old is not None and not old.isRunning():
            try:
                if hasattr(old, "arm_delete_later_on_finished"):
                    old.arm_delete_later_on_finished()
                else:
                    old.deleteLater()
            except RuntimeError:
                pass
            self._cache_reconciliation_worker = None
        self._cache_reconciliation_worker = CacheReconciliationWorker(
            self.settings, parent=self, once=False
        )
        self._cache_reconciliation_worker.progress.connect(
            self._on_cache_reconciliation_progress
        )
        self._cache_reconciliation_worker.finished_ok.connect(
            self._on_full_scan_finished
        )
        self._cache_reconciliation_worker.error.connect(
            lambda m: self.status_bar.set_message(f"Arka plan tarama hata: {m}")
        )
        if hasattr(self._cache_reconciliation_worker, "arm_delete_later_on_finished"):
            self._cache_reconciliation_worker.arm_delete_later_on_finished()
        self._cache_reconciliation_worker.start()
        logger.info(
            "Background full scan started interval=%s min",
            getattr(self.settings, "background_full_scan_interval_min", 30),
        )

    def _on_full_scan_finished(self, result: dict) -> None:
        if result.get("skipped"):
            return
        self.progress_panel.set_full_scan_status(
            f"Tarama bitti · yeni +{int(result.get('new_found', 0) or 0)} · "
            f"değişen {int(result.get('changed_found', 0) or 0)} · "
            f"açık +{int(result.get('gaps_enqueued', 0) or 0)} · "
            f"silinen {int(result.get('deleted_files', result.get('missing_marked', 0)) or 0)}"
        )
        # Otomatik döngü worker bitince yeniden başlat
        if bool(getattr(self.settings, "background_full_scan_enabled", True)) and not (
            self._shutting_down
        ):
            from PySide6.QtCore import QTimer

            QTimer.singleShot(2000, self._start_cache_reconciliation)

    def _start_manual_full_scan(self) -> None:
        old = self._cache_reconciliation_worker
        if old is not None and old.isRunning():
            try:
                old.request_stop()
            except Exception:
                pass
            if hasattr(old, "arm_delete_later_on_finished"):
                old.arm_delete_later_on_finished()
            # Brief wait ≤100ms; if still running, defer start via finished
            old.wait(100)
            if old.isRunning():
                self.progress_panel.set_full_scan_status(
                    "Tarama: önceki tarama durduruluyor…"
                )
                self.status_bar.set_message("Önceki tarama bitince yeniden başlanacak…")

                def _start_once_after():
                    try:
                        old.deleteLater()
                    except RuntimeError:
                        pass
                    if self._cache_reconciliation_worker is old:
                        self._cache_reconciliation_worker = None
                    self._start_manual_full_scan()

                if not getattr(old, "_manual_scan_rehooked", False):
                    old._manual_scan_rehooked = True
                    old.finished.connect(_start_once_after)
                return
            try:
                old.deleteLater()
            except RuntimeError:
                pass
            self._cache_reconciliation_worker = None
        elif old is not None:
            try:
                if hasattr(old, "arm_delete_later_on_finished"):
                    old.arm_delete_later_on_finished()
                else:
                    old.deleteLater()
            except RuntimeError:
                pass
            self._cache_reconciliation_worker = None
        self.progress_panel.set_full_scan_status("Tarama: baştan tarama çalışıyor…")
        self.status_bar.set_message("Sistem baştan taranıyor (eksik/silinen)…")
        w = CacheReconciliationWorker(self.settings, parent=self, once=True)
        self._cache_reconciliation_worker = w
        w.progress.connect(self._on_cache_reconciliation_progress)
        w.finished_ok.connect(self._on_manual_full_scan_finished)
        w.error.connect(
            lambda m: (
                self.progress_panel.set_full_scan_status(f"Tarama hata: {m}"),
                self.status_bar.set_message(str(m)),
            )
        )
        if hasattr(w, "arm_delete_later_on_finished"):
            w.arm_delete_later_on_finished()
        w.start()

    def _on_manual_full_scan_finished(self, result: dict) -> None:
        self.progress_panel.set_full_scan_status(
            f"Tarama tamam · yeni +{int(result.get('new_found', 0) or 0)} · "
            f"değişen {int(result.get('changed_found', 0) or 0)} · "
            f"açık +{int(result.get('gaps_enqueued', 0) or 0)} · "
            f"silinen {int(result.get('deleted_files', result.get('missing_marked', 0)) or 0)}"
        )
        self.status_bar.set_message("Sistem tarama tamamlandı")
        self._refresh_status()
        if bool(getattr(self.settings, "background_full_scan_enabled", True)):
            from PySide6.QtCore import QTimer

            QTimer.singleShot(1500, self._start_cache_reconciliation)

    def _on_full_scan_auto_changed(self, enabled: bool, interval_min: int) -> None:
        self.settings.background_full_scan_enabled = bool(enabled)
        self.settings.background_full_scan_interval_min = int(interval_min or 0)
        self.settings.save()
        if self._cache_reconciliation_worker and self._cache_reconciliation_worker.isRunning():
            try:
                self._cache_reconciliation_worker.request_stop()
            except Exception:
                pass
        if enabled and int(interval_min or 0) > 0:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(500, self._start_cache_reconciliation)
        else:
            self.progress_panel.set_full_scan_status("Tarama: otomatik kapalı (elle)")

    def _on_cache_reconciliation_progress(self, stats: dict) -> None:
        self._cache_reconciliation_stats = dict(stats)
        phase = str(stats.get("phase") or "…")
        self.progress_panel.set_full_scan_status(
            f"Tarama: {phase} · yeni +{int(stats.get('new_found', 0) or 0)} · "
            f"değişen {int(stats.get('changed_found', 0) or 0)} · "
            f"açık +{int(stats.get('gaps_enqueued', 0) or 0)} · "
            f"silinen {int(stats.get('deleted_files', stats.get('missing_marked', 0)) or 0)}"
        )
        if hasattr(self.health_panel, "set_reconciliation_status"):
            self.health_panel.set_reconciliation_status(stats)

    def _toggle_filter_panel(self) -> None:

        self._set_panel_visible(
            "filters", not bool(self._filter_dock and self._filter_dock.isVisible())
        )

    def _toggle_source_dock(self) -> None:

        if self._source_dock:
            self._set_panel_visible("source", not self._source_dock.isVisible())

    def _toggle_ui_mode(self) -> None:
        self._save_ui_state()
        advanced = self.settings.ui_mode != "advanced"
        self.settings.ui_mode = "advanced" if advanced else "simple"
        self.settings.save()
        self._restore_ui_state()

    def _apply_ui_mode_panels(self, advanced: bool) -> None:
        simple = not advanced
        self.results_panel.set_simple_mode(simple)
        self.inspector_panel.set_simple_mode(simple)
        self.progress_panel.set_simple_mode(simple)
        self.progress_panel.set_semantic_enabled(
            bool(getattr(self.settings, "semantic_text_search_enabled", False))
        )
        self._update_user_status_bar()

    def _is_simple_mode(self) -> bool:
        return self.settings.ui_mode != "advanced"

    def _update_user_status_bar(self) -> None:
        indexing = self._index_is_running()
        if self._is_simple_mode():
            line = simple_status_line(
                searchable=self._searchable_count,
                index_percent=self._index_percent,
                indexing=indexing,
                searching=self._search_active,
                search_count=self._search_result_count,
                refining=self._search_refining,
                search_phase=self._search_phase,
                search_kind=self._search_kind,
                empty_result=self._empty_search_result and not self._search_active,
            )
            self.status_bar.set_simple_summary(line)
            if self._search_active:
                self.status_bar.set_busy(True)
            elif indexing and self._index_percent > 0:
                self.status_bar.set_indexing_progress(self._index_percent, "")
            elif indexing:
                self.status_bar.set_busy(True)
            else:
                self.status_bar.clear_indexing_progress()
            self.lbl_header_status.setText(
                simple_header_line(
                    searchable=self._searchable_count,
                    index_percent=self._index_percent,
                    indexing=indexing,
                    searching=self._search_active,
                    search_phase=self._search_phase,
                    search_count=self._search_result_count,
                )
            )
        elif indexing:
            self.status_bar.set_message("İndeks devam ediyor…")

    def _on_panel_settings_changed(self) -> None:
        """Panel görünürlüğü anında; arama debounce — menü geçişleri donmasın."""
        self._apply_panel_setting_visibility()
        self.progress_panel.set_semantic_enabled(
            bool(getattr(self.settings, "semantic_text_search_enabled", False))
        )
        # Ayar her değişince hemen arama başlatma → UI kilitleniyordu
        if not hasattr(self, "_panel_settings_search_timer"):
            self._panel_settings_search_timer = QTimer(self)
            self._panel_settings_search_timer.setSingleShot(True)
            self._panel_settings_search_timer.timeout.connect(
                self._flush_panel_settings_search
            )
        self._panel_settings_search_timer.start(450)

    def _flush_panel_settings_search(self) -> None:
        if self._build_search_query() is not None:
            self._run_current_search()
        elif self._search_response is not None:
            self._refresh_results_display()

    def _on_category_filter_changed(self, path: str) -> None:
        self._category_filter_path = path or ""
        if self._build_search_query() is not None:
            self._run_current_search()
        else:
            self.status_bar.set_message(f"Kategori filtresi: {path or 'Tümü'}")

    def _enable_index_ai_settings(self) -> None:
        """AI embedding indexlemesini aç ve paneli senkronla."""
        self.settings.index_skip_ai = False
        self.settings.save()
        self.filter_panel.ai_settings.load_from_settings()

    def _on_reindex_requested(self) -> None:
        if (
            QMessageBox.question(
                self,
                "Yeniden İndeksle",
                "Tüm aktif kaynaklar yeniden taranıp indexlenecek. Devam edilsin mi?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        if self.settings.ai_embedding_enabled:
            self._enable_index_ai_settings()
        self._set_panel_visible("source", True)
        self._start_index(scan_mode="deep")

    def _on_ai_enable_warning(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Yapay zekâ vektörü")
        box.setText(
            "Yapay zekâ vektörü açıldı. Eski indekste vektörü olmayan dosyalar için "
            "yeniden indeksleme gerekir. Şimdi yeniden indekslemek ister misiniz?"
        )
        btn_now = box.addButton(
            "Şimdi yeniden indexle", QMessageBox.ButtonRole.AcceptRole
        )
        btn_later = box.addButton("Sonra", QMessageBox.ButtonRole.RejectRole)
        btn_new = box.addButton(
            "Sadece yeni dosyalarda kullan",
            QMessageBox.ButtonRole.ActionRole,
        )
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_now:
            self._enable_index_ai_settings()
            self._start_index(scan_mode="deep")
        elif clicked is btn_new:
            self._enable_index_ai_settings()
            self.status_bar.set_message(
                "AI yalnızca yeni indexlenen dosyalarda kullanılacak"
            )

    def _reset_panel_layout(self) -> None:
        reset_layout(self)
        self._panel_visibility.update(
            {
                "query": True,
                "filters": False,
                "source": False,
                "inspector": False,
                "format": False,
                "category": False,
            }
        )
        self.query_image_panel.setVisible(True)
        if getattr(self, "_query_dock", None):
            self._query_dock.show()
            self._query_dock.raise_()
        if self._filter_dock:
            self._filter_dock.hide()
        if self._source_dock:
            self._source_dock.hide()
        if self._inspector_dock:
            self._inspector_dock.hide()
        if self._format_dock:
            self._format_dock.hide()
        if self._category_dock:
            self._category_dock.hide()
        if self._teach_me_dock:
            self._teach_me_dock.hide()
        self._sync_panel_menu()
        self._save_ui_state()
        self.status_bar.set_message("Panel düzeni sıfırlandı")

    def _apply_search_preset(self, key: str) -> None:
        preset = PRESET_BY_KEY.get(key)
        if not preset:
            return
        controls = (
            self.cmb_mode,
            self.cmb_color,
            self.slider_threshold,
            self.cmb_limit,
            self.chk_near_below,
        )
        for control in controls:
            control.blockSignals(True)
        try:
            self.cmb_mode.setCurrentIndex(self.cmb_mode.findData(preset.mode))
            self.cmb_color.setCurrentIndex(self.cmb_color.findData(preset.color_mode))
            self.slider_threshold.setValue(round(preset.threshold * 100))
            limit_index = self.cmb_limit.findData(preset.result_limit)
            if limit_index >= 0:
                self.cmb_limit.setCurrentIndex(limit_index)
            self.chk_near_below.setChecked(preset.show_near_below)
        finally:
            for control in controls:
                control.blockSignals(False)
        self.settings.search_preset = preset.key
        self._apply_search_settings()
        if self._search_response:
            self._refresh_results_display()

    def _confirm_search_threshold(self, query: SearchQuery) -> bool:
        warning = high_threshold_warning(
            self.settings.search_mode, float(query.threshold or 0)
        )
        if not warning:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Yüksek arama eşiği")
        box.setText(warning)
        recommended = box.addButton(
            "Önerilen ayara geç", QMessageBox.ButtonRole.AcceptRole
        )
        box.addButton("Yine de böyle ara", QMessageBox.ButtonRole.DestructiveRole)
        box.exec()
        if box.clickedButton() is recommended:
            key = "texture" if self.settings.search_mode == "style" else "comprehensive"
            index = self.search_header.cmb_preset.findData(key)
            self.search_header.cmb_preset.setCurrentIndex(index)
            query.threshold = self.settings.similarity_threshold
        return True

    def _on_hover_preview(
        self, file_id: int, thumbnail_path: str, filename: str, source_path: str = ""
    ) -> None:
        self._place_inspector_in_detail_slot()
        self._hover_preview_panel.cancel_hide()
        self._hover_preview_panel.show_thumbnail(
            int(file_id),
            thumbnail_path,
            filename,
            source_path=source_path,
        )
        if self._inspector_dock and not self._inspector_dock.isVisible():
            self._set_panel_visible("inspector", True)

    def _on_hover_preview_clear(self) -> None:
        self._hover_preview_panel.schedule_hide()

    def _show_result_detail(self, result) -> None:
        from core.db import Database
        from core.user_feedback import (
            UserFeedbackStore,
            apply_metadata_overlay_to_result,
        )

        try:
            overlay = UserFeedbackStore(
                Database(self.settings.db_path)
            ).metadata_overlay_for_file(result.file_id)
            apply_metadata_overlay_to_result(result, overlay)
        except Exception:
            pass
        self.inspector_panel.set_result(result)
        self._load_inspector_learned_tags(result.file_id)

        if self._inspector_dock and not self._inspector_dock.isVisible():
            self._set_panel_visible("inspector", True)

    def dragEnterEvent(self, event: QDragEnterEvent):

        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.toLocalFile()]
        if paths:
            self._apply_query_images(paths, auto_search=True)
        event.acceptProposedAction()

    def _schedule_instant_visual_search(self, _path: str = "") -> None:
        if self._suppress_instant_search:
            return
        if len(self._query_image_paths) > 1:
            return
        if not getattr(self.settings, "auto_search_on_image", True):
            return
        self._pending_instant_use_crop = False
        self._instant_search_timer.start()

    def _schedule_instant_crop_search(self) -> None:
        if not getattr(self.settings, "auto_search_on_image", True):
            return
        if not self.query_panel.crop_rect():
            return
        self._pending_instant_use_crop = True
        self._instant_search_timer.start()

    def _run_pending_instant_search(self) -> None:
        self._run_image_search(
            use_crop=self._pending_instant_use_crop,
            instant=True,
        )

    def _on_auto_scan_changed(self) -> None:

        self.settings.auto_scan_on_startup = self.source_sidebar.is_auto_scan_enabled()

        self.source_sidebar.set_auto_scan(self.settings.auto_scan_on_startup)

        self.settings.save()

    def _on_folder_watch_changed(self) -> None:
        enabled = self.source_sidebar.is_folder_watch_enabled()
        self.settings.folder_watch_enabled = enabled
        self._folder_watch.set_enabled(enabled)
        if enabled:
            self._refresh_folder_watch_async()
        self.settings.save()
        self.status_bar.set_message(
            "Klasör izleme açık — dosya ekleme/silme sonrası hızlı tarama"
            if enabled
            else "Klasör izleme kapalı"
        )

    def _on_purge_missing_setting_changed(self) -> None:
        self.settings.purge_missing_after_scan = (
            self.source_sidebar.is_purge_missing_after_scan_enabled()
        )
        self.settings.save()

    def _on_folder_watch_scan(self, source_id: int, reason: str) -> None:
        if source_id <= 0:
            return
        if self._index_is_running():
            self._pending_watch_scans.add(source_id)
            self.status_bar.set_message(
                f"Klasör değişikliği kuyruğa alındı (kaynak #{source_id})"
            )
            return
        self.status_bar.set_message(f"Klasör değişikliği — hızlı tarama başlıyor…")
        self._run_index(scan_mode="quick", source_id=source_id, quiet=True)

    def _flush_pending_watch_scans(self) -> None:
        if not self._pending_watch_scans or self._index_is_running():
            return
        sid = self._pending_watch_scans.pop()
        self._run_index(scan_mode="quick", source_id=sid, quiet=True)

    def _purge_missing_records(self) -> None:
        from core.db import Database

        if self._purge_missing_worker and self._purge_missing_worker.isRunning():
            QMessageBox.information(self, "Bilgi", "Temizlik zaten devam ediyor.")
            return
        db = Database(self.settings.db_path)
        pending = db.count_missing_files()
        if pending == 0:
            QMessageBox.information(self, "Bilgi", "Temizlenecek silinmiş kayıt yok.")
            return
        reply = QMessageBox.question(
            self,
            "Silinmiş Kayıtları Temizle",
            f"{pending:,} adet 'missing' index kaydı veritabanından tamamen silinecek.\n"
            "Orijinal dosyalar zaten diskte yok.\n\nDevam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.status_bar.set_busy(True)
        self.status_bar.set_message("Silinmiş kayıtlar temizleniyor…")
        self._purge_missing_worker = PurgeMissingWorker(self.settings, parent=self)
        self._purge_missing_worker.finished_ok.connect(self._on_purge_missing_finished)
        self._purge_missing_worker.error.connect(self._on_purge_missing_error)
        self._purge_missing_worker.start()

    def _on_purge_missing_finished(self, result: dict) -> None:
        self._purge_missing_worker = None
        self.status_bar.set_busy(False)
        removed = int(result.get("files_removed", 0))
        self._refresh_status()
        self.status_bar.set_message(f"Silinmiş kayıt temizlendi: {removed:,}")
        if removed:
            QMessageBox.information(
                self, "Tamamlandı", f"{removed:,} silinmiş index kaydı kaldırıldı."
            )

    def _on_purge_missing_error(self, message: str) -> None:
        self._purge_missing_worker = None
        self.status_bar.set_busy(False)
        QMessageBox.critical(self, "Temizlik Hatası", message)

    def _on_view_mode_changed(self, mode: str) -> None:

        self.settings.results_view_mode = mode

        self.settings.save()

        self._refresh_results_display()

    def _on_threshold_changed(self, value: int) -> None:

        self.toolbar.lbl_threshold.setText(f"%{value}")

        self.settings.similarity_threshold = value / 100

        self._filter_timer.start()

    def _on_filter_changed(self) -> None:

        self._apply_search_settings()

        if self._search_response:
            self._display_count = self.cmb_limit.currentData()

            self._refresh_results_display()

        else:
            self._filter_timer.start()

    def _on_filter_changed_research(self) -> None:

        self._apply_search_settings()

        if self._last_query_path or self.txt_search.text().strip():
            self._run_current_search()

        elif self._search_response:
            self._refresh_results_display()

    def _apply_search_settings(self) -> None:

        self.settings.similarity_threshold = self.slider_threshold.value() / 100

        self.settings.search_result_limit = self.cmb_limit.currentData()

        self.settings.search_scope = self.cmb_scope.currentData()

        self.settings.customer_filter = (
            self.search_header.selected_customer()
            or (self.cmb_customer.currentData() or "")
            or ""
        )

        if self.settings.search_scope == SearchScope.SELECTED_SOURCES.value:
            ids = self.sources_panel.selected_source_ids()

            if not ids:
                ids = [
                    s["id"] for s in self._source_manager.list_sources(active_only=True)
                ]

            self.settings.selected_source_ids = ids

        if self.settings.search_scope != SearchScope.QUICK_FOLDER.value:
            self.settings.quick_search_folder = ""

        if self.settings.search_scope == SearchScope.SELECTED_FOLDER.value:
            sid = self.sources_panel.selected_source_id()

            if sid:
                src = self._source_manager.get_source(sid)

                if src:
                    self.settings.selected_folder_path = src.get("root_path", "")

                    self.settings.selected_source_id = sid

        self.settings.show_near_below_threshold = self.chk_near_below.isChecked()

        self.settings.search_mode = self.cmb_mode.currentData()

        self.settings.color_weight_mode = self.cmb_color.currentData()

        from core.search_method_modes import apply_search_method_mode

        method_key = "hybrid"
        if hasattr(self, "cmb_search_method") and self.cmb_search_method is not None:
            method_key = self.cmb_search_method.currentData() or "hybrid"
        apply_search_method_mode(self.settings, str(method_key))

        mode_label = self.cmb_mode.currentText()

        if self._is_simple_mode():
            self.lbl_header_mode.setText("Basit Mod")
        else:
            self.lbl_header_mode.setText(f"Arama modu: {mode_label}")

        self.settings.save()

    def _refresh_results_display(
        self,
        *,
        live_stage: str = "",
        in_place: bool = False,
    ) -> None:
        self._pending_refresh = {
            "live_stage": live_stage,
            "in_place": in_place,
        }
        self._refresh_results_timer.start()

    def _do_refresh_results_display(self) -> None:
        pending = dict(self._pending_refresh or {})
        live_stage = str(pending.get("live_stage") or "")
        in_place = bool(pending.get("in_place"))

        if not self._search_response:
            self.summary_panel.clear()

            return

        from core.dynamic_groups import DEFAULT_HIDDEN_GROUPS, normalize_cluster_key
        from core.search_display import (
            filter_ranked_visual_results,
            is_ranked_visual_search,
            visual_search_floor,
        )

        meta = self._search_response.meta or {}
        ranked = is_ranked_visual_search(
            meta,
            has_text=bool(self.txt_search.text().strip()),
        )
        floor = (
            visual_search_floor(self.settings)
            if ranked
            else self.settings.similarity_threshold
        )
        threshold = floor
        q_family = str(meta.get("query_pattern_family") or "")
        q_animal = str(meta.get("query_animal_print_type") or "")

        if ranked:
            filtered = filter_ranked_visual_results(
                self._search_response.all_results,
                floor,
                hidden_clusters=(
                    DEFAULT_HIDDEN_GROUPS
                    if (
                        self.settings.search_mode == "comprehensive"
                        and not self.settings.show_unrelated_results
                    )
                    else None
                ),
                normalize_cluster_key=normalize_cluster_key,
                query_family=q_family,
                query_animal=q_animal,
            )
        else:
            threshold = self.settings.similarity_threshold
            filtered = self._search_response.filter_by_threshold(threshold)
            if (
                self.settings.search_mode == "comprehensive"
                and not self.settings.show_unrelated_results
            ):
                filtered = [
                    r
                    for r in filtered
                    if normalize_cluster_key(r.cluster_group or r.category or "")
                    not in DEFAULT_HIDDEN_GROUPS
                    and r.category != "weak"
                ]

        if not filtered:
            filtered = list(self._search_response.results or [])
        above = len(filtered)

        self.lbl_threshold_count.setText(
            f"| %{int(floor * 100)}+: {above:,}"
            if ranked
            else f"| Eşik üstü: {above:,}"
        )

        shown = filtered
        self._filtered_total = above

        near_below: list = []

        if (
            not ranked
            and self.settings.show_near_below_threshold
            and self._search_response.below_threshold_preview
        ):
            near_below = self._search_response.below_threshold_preview[:20]

        stats = self._search_response.stats

        stats.above_threshold = above

        stats.displayed = min(len(shown), self.results_panel.visible_count())

        stats.remaining = max(0, above - stats.displayed)

        page_size = int(self.cmb_limit.currentData() or 500)
        self.results_panel.set_scroll_page_size(page_size)

        self.summary_panel.update_stats(stats, threshold, len(near_below))

        if threshold >= 0.90 and self._search_response.all_results and not ranked:
            self.summary_panel.set_threshold_hint(
                f"%{int(threshold * 100)} eşik sadece çok yakın sonuçları gösterir. "
                "Benzer dokular için %50–70 önerilir."
            )
        elif ranked:
            crop_note = " (seçili bölge)" if meta.get("used_crop") else ""
            self.summary_panel.set_threshold_hint(
                f"Görsel{crop_note} — en yakın %{int(floor * 100)}–100 arası sonuçlar skor sırasıyla."
            )
        else:
            self.summary_panel.set_threshold_hint("")

        page_size = int(self.cmb_limit.currentData() or 500)
        progressive_live = bool(
            in_place
            or meta.get("streaming")
            or meta.get("progressive_stage")
            or meta.get("fast_hash_only")
            or self._is_search_worker_running()
        )
        updated = False
        # Progressive: sadece gorunen pencereyi ve kucuk tamponu guncelle.
        # Tum sayfayi UI thread'de tekrar islemek Windows "Yanıt Vermiyor" riskini artirir.
        visible_window = max(self.results_panel.rendered_card_count(), 12)
        live_buffer = 12 if ranked else 20
        live_slice = shown[: min(len(shown), visible_window + live_buffer)]
        panel_full = len(getattr(self.results_panel, "_full_results", []) or [])
        panel_above = int(getattr(self.results_panel, "_total_above", 0) or 0)
        # Yeni adaylar geldiyse in-place yetmez — aksi halde sadece ilk 1 sonuç kalır.
        need_rebuild = len(shown) > panel_full or above > panel_above or panel_full == 0
        if (
            progressive_live
            and self.results_panel.has_result_cards()
            and not need_rebuild
        ):
            updated = self.results_panel.update_scores_in_place(
                live_slice,
                stage=live_stage or "Tam Analiz",
            )
            if updated:
                if self._is_simple_mode():
                    self._search_result_count = above
                    self._search_refining = True
                    self._update_user_status_bar()
                return
        elif (
            in_place
            and getattr(self.settings, "live_score_updates", True)
            and not need_rebuild
        ):
            updated = self.results_panel.update_scores_in_place(
                live_slice,
                stage=live_stage or "Tam Analiz",
            )
        if not updated:
            self.results_panel.show_results(
                shown,
                stats,
                threshold,
                has_more=len(filtered) > page_size,
                below_threshold=near_below,
                display_limit=page_size,
                query_ctx=resolve_query_context_from_meta(self._search_response.meta),
                ranked=ranked,
                score_floor=floor if ranked else 0.0,
            )

        if self._is_simple_mode():
            self._search_result_count = above
            if self._is_search_worker_running():
                self._search_refining = bool(
                    self._search_response.meta.get("fast_hash_only")
                )
            self._update_user_status_bar()
        else:
            if self._is_search_worker_running():
                if self._search_response.meta.get("fast_hash_only"):
                    self.status_bar.set_message(
                        f"Hızlı sonuçlar ({above:,}) — derin analiz ve sıralama güncelleniyor…"
                    )
                else:
                    self.status_bar.set_message("Arama tamamlanıyor…")
            else:
                self.status_bar.set_message(
                    (
                        f"Kapsam: {self.cmb_scope.currentText()} | Index: {stats.total_indexed:,} | "
                        f"Aday: {stats.candidates_evaluated:,} | Mod: {self.cmb_mode.currentText()} | "
                        f"Eşik: %{int(threshold * 100)} | Patch: "
                        f"{'Kapalı' if self._search_response.meta.get('fast_hash_only') else 'Açık'} | "
                        f"AI: {'Açık' if self._search_response.meta.get('ai_used') else 'Kapalı'} | "
                        f"FAISS: {self._search_response.meta.get('faiss_dino_count', 0) + self._search_response.meta.get('faiss_clip_count', 0):,}"
                    )
                )

    def _load_more_results(self) -> None:
        self.results_panel.append_more_pages(1)
        if self._search_response:
            stats = self._search_response.stats
            stats.displayed = self.results_panel.visible_count()
            stats.remaining = max(0, int(getattr(self, "_filtered_total", 0)) - stats.displayed)
            self.results_panel.update_page_footer(stats)

    def _mark_preview_scheduler_interactive(self) -> None:
        sched = getattr(self, "_thumb_scheduler", None)
        if sched is not None and hasattr(sched, "mark_ui_interactive"):
            sched.mark_ui_interactive()

    def _on_ui_perf_updated(self, snap) -> None:
        search_q = 1 if self._is_search_worker_running() else 0
        if self._ui_perf:
            self._ui_perf.set_search_queue(search_q)
            if self._thumb_scheduler:
                self._ui_perf.set_preview_queue(self._thumb_scheduler.queue_depth())
        self.status_bar.set_perf_metrics(
            ui_fps=snap.ui_fps,
            search_queue=search_q,
            preview_queue=snap.preview_queue,
            thumbnails_per_sec=snap.thumbnails_per_sec,
        )

    def _set_searching(
        self,
        active: bool,
        *,
        kind: str = "",
        phase: str = "",
    ) -> None:

        self.search_header.set_searching(active)
        self._search_active = active
        if kind:
            self._search_kind = kind
        if active:
            self._search_phase = phase or "starting"
            self._empty_search_result = False
        else:
            self._search_result_count = 0
            self._search_refining = False
            self._search_phase = ""
            self.results_panel.set_search_activity(False)

        index_running = self._index_is_running()
        if active:
            if self._search_kind == "text":
                msg = "Metin aranıyor…"
            elif self._search_kind == "visual":
                msg = "Görsel aranıyor…"
            else:
                msg = "Benzer desenler aranıyor…"
            self.results_panel.set_search_activity(True, msg)
            self.results_panel.show_loading_skeleton(msg)
        if active and not index_running:
            self.status_bar.set_busy(True)
        elif not active and not index_running:
            self.status_bar.set_busy(False)

        if self.settings.index_throttle_on_search:
            set_search_active(active)

        self._update_user_status_bar()
        if self._is_simple_mode():
            return

        if active:
            self.status_bar.set_message("Hızlı sonuçlar aranıyor…")
        else:
            if not index_running:
                self.status_bar.set_busy(False)
            extra = index_status_label(self.settings)
            if extra:
                self.status_bar.set_message(extra)

    def _check_resume_index_on_startup(self) -> None:
        import os

        # Headless/autonomous QA: never open ResumeIndexDialog (blocks + freezes).
        if os.environ.get("VEZIR_QA_NONINTERACTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        if self._shutting_down or self._resume_index_suppressed:
            return
        if not getattr(self.settings, "resume_index_on_startup", True):
            return
        if self._startup_probe_running:
            return
        self._startup_probe_running = True

        def probe_startup():
            from core.db import Database
            from core.index_session import (
                load_session,
                mark_interrupted,
                snapshot_progress_totals,
            )
            from core.production.recovery import run_startup_recovery

            run_startup_recovery(self.settings)
            manager = SourceManager(self.settings, run_maintenance=True)
            manager.migrate_legacy_archive_root()
            db = Database(self.settings.db_path)
            queue = IndexQueueManager(db)
            queue.reset_stale_running()
            session = load_session(db)
            if session and session.status == "running":
                mark_interrupted(db)
                session = load_session(db)
            progress = snapshot_progress_totals(db)
            incomplete = queue.sources_with_incomplete_work()
            return {
                "incomplete": incomplete,
                "session": session,
                "progress": progress,
            }

        def probe_done(payload) -> None:
            self._startup_probe_running = False
            self._refresh_status()
            self._refresh_folder_watch_async()
            if self._shutting_down:
                return
            incomplete = (payload or {}).get("incomplete") or []
            session = (payload or {}).get("session")
            progress = (payload or {}).get("progress") or {}
            resumable = bool(session and getattr(session, "is_resumable", lambda: False)())
            if not incomplete and not resumable:
                return
            dlg = ResumeIndexDialog(
                incomplete, self, session=session, progress=progress
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                if dlg.choice == "stop_session":
                    self._resume_index_suppressed = True
                    try:
                        from core.db import Database
                        from core.index_session import clear_session

                        clear_session(Database(self.settings.db_path))
                    except Exception:
                        pass
                return
            if dlg.choice == "fresh_start":
                try:
                    from core.db import Database
                    from core.index_session import clear_session

                    clear_session(Database(self.settings.db_path))
                except Exception:
                    pass
                self._run_index(scan_mode="quick", index_mode="complete")
                return
            mode = "complete"
            if session and getattr(session, "index_mode", ""):
                mode = session.index_mode
            elif dlg.resume_index_mode not in ("", "standard"):
                mode = dlg.resume_index_mode
            self._run_index(scan_mode="quick", index_mode=mode)

        self._run_background_task(
            probe_startup,
            working_message="Index bilgisi arka planda hazırlanıyor…",
            success_message="",
            on_done=probe_done,
        )

    def _maybe_offer_on_demand_scan(self, response) -> None:
        import os

        if os.environ.get("VEZIR_QA_NONINTERACTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            return
        if not response:
            return
        from core.db import Database

        db = Database(self.settings.db_path)
        pending = db.count_pending_for_sources(
            list(self.settings.selected_source_ids or []) or None
        )
        if not self._on_demand_scan_needed(response, pending):
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("İndeks Eksik")
        box.setText(
            f"İndekste güçlü sonuç bulunamadı.\n"
            f"Seçili kapsamda yaklaşık {pending:,} bekleyen dosya var."
        )
        quick = box.addButton("Hızlı Tara ve Ara", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Mevcut İndeksle Devam Et", QMessageBox.ButtonRole.RejectRole)
        background = box.addButton(
            "Arka Planda Tara", QMessageBox.ButtonRole.ActionRole
        )
        box.exec()
        if box.clickedButton() is quick:
            self._rerun_after_index = True
            self._run_index(scan_mode="quick")
        elif box.clickedButton() is background:
            self._run_index(scan_mode="quick")

    @staticmethod
    def _on_demand_scan_needed(response, pending: int) -> bool:
        if not response or pending < 50:
            return False
        meta = response.meta or {}
        # Human/gender searches have an intentionally lower semantic acceptance
        # floor than textile similarity searches.  A low raw CLIP score or a
        # large pending index must not trigger the generic "İndeks Eksik" dialog
        # when the human route already produced valid candidates.
        if bool(meta.get("human_semantic_mode")):
            valid = [
                r for r in (getattr(response, "all_results", None) or [])
                if SearchResponse._passes_display_threshold(
                    r, float(meta.get("threshold", 0.60) or 0.60)
                )
            ]
            if valid:
                return False
        if int(meta.get("protected_exact_candidates_count", 0)) > 0:
            return False
        top = response.results[0].score if response.results else 0.0
        indexed = max(1, int(getattr(response.stats, "total_indexed", 0) or 0))
        coverage_low = pending > indexed * 2
        return top < 0.75 or coverage_low

    def _search_worker_alive(self) -> bool:
        from core.qthread_lifecycle import qobject_is_alive

        return qobject_is_alive(getattr(self, "_search_worker", None))

    def _is_search_worker_running(self) -> bool:
        from core.qthread_lifecycle import qthread_is_running

        return qthread_is_running(getattr(self, "_search_worker", None))

    def _clear_search_worker_ref_if_same(self, worker) -> None:
        """Clear Python ref when THAT worker's C++ object is destroyed.

        Must not call methods on the deleted QObject (zombie wrapper).
        """
        if self._search_worker is worker:
            self._search_worker = None

    def _hook_search_qthread_finished(self) -> None:
        if self._search_finish_hooked:
            return
        w = self._search_worker
        if w is None:
            return
        self._search_finish_hooked = True
        w.finished.connect(self._on_search_qthread_finished)

    def _cancel_search_worker(self) -> None:
        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        w = self._search_worker
        if w is None:
            return
        if qthread_is_running(w):
            w.request_stop()
            self._hook_search_qthread_finished()
            # Do not wait long; do not create a new worker; keep C++ object alive
            return
        if qobject_is_alive(w):
            try:
                if hasattr(w, "arm_delete_later_on_finished"):
                    w.arm_delete_later_on_finished()
                else:
                    w.deleteLater()
            except RuntimeError:
                pass
        self._search_worker = None

    def _on_search_qthread_finished(self) -> None:
        sender = self.sender()
        try:
            if sender is not None:
                sender.deleteLater()
        except RuntimeError:
            pass
        if sender is self._search_worker:
            self._search_worker = None
        self._search_finish_hooked = False
        pending = self._pending_search
        if pending is not None:
            self._pending_search = None
            query, instant = pending
            self._launch_search_worker(query, instant=bool(instant))

    def _cancel_search(self) -> None:
        self._pending_search = None
        self._cancel_search_worker()

        self._set_searching(False)

        self.status_bar.set_message("Arama iptal edildi")

    def _clear_search(self) -> None:
        self._pending_search = None
        self._cancel_search_worker()

        self._search_response = None

        self._last_query_path = None
        self._query_image_paths = []

        self.txt_search.clear()
        self.search_header.set_selected_customer("")
        self.settings.customer_filter = ""

        self.query_panel.set_image(None)

        self.search_header.set_has_image(False)

        self.query_panel.set_search_mode_label("")

        self.inspector_panel.set_result(None)

        self.summary_panel.clear()

    def _clear_query_image_only(self) -> None:
        self._last_query_path = None
        self._query_image_paths = []
        self.query_panel.set_image(None)
        self.search_header.set_has_image(False)
        self.query_panel.set_search_mode_label("")
        if self.txt_search.text().strip():
            self._run_current_search()
        else:
            self.status_bar.set_message("Sorgu görseli kaldırıldı")

        self.results_panel.show_results([], None)

        self.lbl_threshold_count.setText("")

        self._set_searching(False)

        self.status_bar.set_message("Hazır")

    def _build_search_query(
        self, use_crop: bool = False, fast_only: bool = False
    ) -> SearchQuery | None:

        self._apply_search_settings()

        text = self.txt_search.text().strip()

        path = self.query_panel.image_path() or self._last_query_path or ""

        if not path and not text:
            return None

        crop = self.query_panel.crop_rect() if use_crop else None

        if use_crop and not crop:
            QMessageBox.information(self, "Bilgi", "Önce görsel üzerinde alan seçin.")

            return None

        mode = "hybrid" if path and text else ("image" if path else "text")
        extra = list(self._query_image_paths or [])
        if path and extra and path not in extra:
            extra = [path] + [p for p in extra if p != path]
        if use_crop or text:
            extra = []

        return SearchQuery(
            mode=mode,
            image_path=path,
            text=text,
            crop_rect=crop,
            use_crop=use_crop and crop is not None,
            customer=self.search_header.selected_customer()
            or self.settings.customer_filter,
            threshold=self.settings.similarity_threshold,
            fast_only=fast_only
            or self.settings.search_scope == SearchScope.QUICK_FOLDER.value,
            category_path_filter=self._category_filter_path,
            image_paths=extra if len(extra) > 1 else [],
        )

    def _run_current_search(self) -> None:

        q = self._build_search_query(use_crop=False)

        if not q:
            QMessageBox.information(self, "Bilgi", "Görsel veya metin girin.")

            return

        if self._confirm_search_threshold(q):
            if not self._ensure_search_pool_ready():
                return
            self._start_search_worker(q)

    def _indexed_pool_size(self) -> int:
        # StatusWorker owns counts; never build SearchEngine on the UI thread.
        if self._last_status:
            indexed = int(self._last_status.get("indexed", 0) or 0)
            if indexed > 0:
                return indexed
        return max(0, int(self._searchable_count or 0))

    def _repair_stale_search_scope(self) -> None:
        """Pasif/boş klasör kapsamını düzelt — arama havuzu 0 olmasın."""
        from core.sources import SearchScope

        if not getattr(self, "_sources_ready", False):
            return
        scope = self.settings.search_scope
        if scope not in (
            SearchScope.SELECTED_FOLDER.value,
            SearchScope.QUICK_FOLDER.value,
        ):
            return
        folder = (
            self.settings.selected_folder_path
            if scope == SearchScope.SELECTED_FOLDER.value
            else self.settings.quick_search_folder
        )
        if not folder:
            self._set_search_scope(SearchScope.ALL.value)
            return
        sid = int(self.settings.selected_source_id or 0)
        if sid:
            src = self._source_manager.get_source(sid)
            if src and not src.get("is_active"):
                self._set_search_scope(SearchScope.ALL.value)
                self.status_bar.set_message(
                    f"Pasif kaynak kapsamı kaldırıldı — tüm aktif kaynaklarda aranacak ({src.get('name', '')})"
                )
                return
        # Empty-scope fallback is resolved inside SearchWorker.

    def _set_search_scope(self, scope: str) -> None:
        from core.sources import SearchScope

        self.settings.search_scope = scope
        idx = self.search_header.cmb_scope.findData(scope)
        if idx >= 0:
            self.search_header.cmb_scope.setCurrentIndex(idx)
        if scope == SearchScope.SELECTED_SOURCES.value:
            ids = self.sources_panel.selected_source_ids()
            if not ids:
                ids = [
                    s["id"] for s in self._source_manager.list_sources(active_only=True)
                ]
            self.settings.selected_source_ids = ids
        self.settings.save()

    def _ensure_search_pool_ready(self) -> bool:
        """Arama öncesi havuz kontrolü; boşsa kullanıcıya genişletme öner."""
        from core.sources import SearchScope

        self._apply_search_settings()
        self._repair_stale_search_scope()
        if not self._status_loaded:
            return True
        size = self._indexed_pool_size()
        if size > 0:
            return True
        scope = self.settings.search_scope
        folder = self.settings.selected_folder_path or self.settings.quick_search_folder
        reply = QMessageBox.question(
            self,
            "Arama Havuzu Boş",
            "Seçili arama kapsamında indexlenmiş dosya bulunamadı.\n\n"
            f"Kapsam: {scope}\n"
            f"Klasör: {folder or '—'}\n\n"
            "Tüm aktif kaynaklarda aramak ister misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            QMessageBox.information(
                self,
                "Arama Yapılamadı",
                "Index havuzunda sonuç yok. Kaynak & Index panelinden tarama yapın "
                "veya kapsamı 'Tüm aktif kaynaklar' olarak değiştirin.",
            )
            return False
        self._set_search_scope(SearchScope.ALL.value)
        self._update_scope_hint()
        return self._indexed_pool_size() > 0

    def _run_image_search(
        self, use_crop: bool = False, *, instant: bool = False
    ) -> None:

        from core.search_display import visual_search_floor

        self._use_crop = use_crop

        q = self._build_search_query(use_crop=use_crop)

        if not q:
            return

        if instant or (q.mode == "image" and not q.text):
            q.threshold = visual_search_floor(self.settings)

        if q and self._confirm_search_threshold(q):
            if instant:
                if self._status_loaded and self._indexed_pool_size() <= 0:
                    self._repair_stale_search_scope()
                    if self._indexed_pool_size() <= 0 and getattr(
                        self.settings, "search_scope_fallback_on_empty", False
                    ):
                        from core.sources import SearchScope

                        self._set_search_scope(SearchScope.ALL.value)
                if self._status_loaded and self._indexed_pool_size() <= 0:
                    self.status_bar.set_message("Index boş — önce kaynak tarayın.")
                    return
            elif not self._ensure_search_pool_ready():
                return
            self._start_search_worker(q, instant=instant)

    def _start_search_worker(
        self, query: SearchQuery, *, instant: bool = False
    ) -> None:
        from core.qthread_lifecycle import qthread_is_running, should_defer_new_worker

        self.results_panel.begin_new_search()

        page_size = int(self.cmb_limit.currentData() or 500)
        self.results_panel.set_scroll_page_size(page_size)

        kind = "text"
        if query.image_path and query.text:
            kind = "hybrid"
        elif query.image_path:
            kind = "visual"

        self._set_searching(True, kind=kind, phase="starting")

        if query.use_crop:
            if instant:
                from core.search_display import visual_search_floor

                floor = int(visual_search_floor(self.settings) * 100)
                self.query_panel.set_search_mode_label(
                    f"Anlık bölge araması — %{floor}–100 benzerler"
                )
            else:
                self.query_panel.set_search_mode_label("Seçili bölgeyle arandı")

        elif query.text and query.image_path:
            self.query_panel.set_search_mode_label(f"Görsel + metin: {query.text}")

        elif query.image_path:
            if instant:
                from core.search_display import visual_search_floor

                floor = int(visual_search_floor(self.settings) * 100)
                self.query_panel.set_search_mode_label(
                    f"Anlık arama — %{floor}–100 en yakın eşleşmeler"
                )
            else:
                self.query_panel.set_search_mode_label("Tüm görselle arandı")

        elif query.text:
            self.query_panel.set_search_mode_label(f"Metin araması: {query.text}")

        else:
            self.query_panel.set_search_mode_label("")

        if instant and not self._is_simple_mode():
            self.status_bar.set_message("En yakın eşleşmeler aranıyor…")

        cur = self._search_worker
        if cur is not None and should_defer_new_worker(qthread_is_running(cur)):
            # Single-flight: keep one SearchWorker; overwrite pending with latest
            self._pending_search = (query, instant)
            cur.request_stop()
            self._hook_search_qthread_finished()
            return

        self._launch_search_worker(query, instant=instant)

    def _launch_search_worker(
        self, query: SearchQuery, *, instant: bool = False
    ) -> None:
        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        # Ensure only one SearchWorker; never start while another is running
        cur = self._search_worker
        if qthread_is_running(cur):
            self._pending_search = (query, instant)
            cur.request_stop()
            self._hook_search_qthread_finished()
            return
        if cur is not None:
            if qobject_is_alive(cur):
                try:
                    if hasattr(cur, "arm_delete_later_on_finished"):
                        cur.arm_delete_later_on_finished()
                    else:
                        cur.deleteLater()
                except RuntimeError:
                    pass
            self._search_worker = None

        self._search_worker = SearchWorker(self.settings, query, parent=self)
        # When C++ object dies (e.g. finished→deleteLater without finish hook),
        # clear the Python ref so later isRunning() sites never see a zombie.
        _sw = self._search_worker
        _sw.destroyed.connect(
            lambda *_a, w=_sw: self._clear_search_worker_ref_if_same(w)
        )
        if self._thumb_scheduler:
            self._thumb_scheduler.cancel_all()

        self._search_worker.partial.connect(self._on_partial_search)

        self._search_worker.finished_ok.connect(self._on_search_finished)

        self._search_worker.error.connect(self._on_search_error)
        if hasattr(self._search_worker, "progress"):
            self._search_worker.progress.connect(self._on_multi_image_progress)

        if hasattr(self._search_worker, "arm_delete_later_on_finished"):
            self._search_worker.arm_delete_later_on_finished()
        self._search_worker.start()

    def _on_search_error(self, message: str) -> None:
        timer = getattr(self, "_partial_ui_timer", None)
        if timer is not None:
            timer.stop()
        pending = getattr(self, "_pending_partial_response", None)
        if pending is not None:
            self._pending_partial_response = None
            self._search_response = pending
        kept = self._search_response
        shown = 0
        if kept is not None:
            shown = len(kept.results or kept.all_results or [])
        self._set_searching(False)
        if kept is not None:
            self._search_response = kept
            self._search_result_count = shown
            self._search_refining = False
            self.results_panel.set_search_activity(False)
            self._refresh_results_display(in_place=True)
        QMessageBox.critical(self, "Arama Hatası", message)

    def _on_partial_search(self, response) -> None:
        # Arama sırasında menü tıklanabilsin — UI güncellemesini idle'a bırak
        self._pending_partial_response = response
        if not hasattr(self, "_partial_ui_timer"):
            self._partial_ui_timer = QTimer(self)
            self._partial_ui_timer.setSingleShot(True)
            self._partial_ui_timer.timeout.connect(self._flush_partial_search_ui)
        if not self._partial_ui_timer.isActive():
            self._partial_ui_timer.start(50)

    def _flush_partial_search_ui(self) -> None:
        response = getattr(self, "_pending_partial_response", None)
        if response is None:
            return
        self._pending_partial_response = None
        self._search_response = response
        self._search_refining = True
        meta = response.meta or {}
        stage = str(meta.get("progressive_stage") or "Tam Eşleşme")
        count = len(response.results or response.all_results or [])
        self._search_result_count = count
        self.results_panel.set_search_activity(
            True,
            f"{stage} — sonuçlar güncelleniyor",
            result_count=count,
            refining=True,
        )
        self._refresh_results_display(live_stage=stage, in_place=True)
        self._update_user_status_bar()

    def _on_search_finished(self, response) -> None:
        previous = self._search_response
        if (
            previous is not None
            and not (response.meta or {}).get("stable_order_applied")
            and not (response.meta or {}).get("multi_image")
        ):
            from core.live_scoring import stable_live_order

            lock = bool(getattr(self.settings, "lock_result_ranking", True)) and not bool(
                (response.meta or {}).get("adaptive_retrieve")
            )
            response.all_results = stable_live_order(
                list(previous.all_results or []),
                list(response.all_results or []),
                tolerance=0.03,
                lock_ranking=lock,
            )
            response.results = stable_live_order(
                list(previous.results or []),
                list(response.results or []),
                tolerance=0.03,
                lock_ranking=lock,
            )
        self._search_response = response

        self._set_searching(False)

        above = int(response.stats.above_threshold or 0)
        self._empty_search_result = above == 0
        self._search_result_count = above

        report_response = copy.deepcopy(response)
        self._run_background_task(
            lambda: save_search_report(report_response),
            working_message="",
            success_message="",
        )

        stage = (
            "AI ile Güncellendi"
            if (response.meta or {}).get("ai_used")
            else "Tam Analiz"
        )
        allow_live = not (
            stage == "AI ile Güncellendi"
            and not bool(getattr(self.settings, "ai_live_refresh", True))
        )
        self._refresh_results_display(live_stage=stage, in_place=allow_live)

        meta = response.meta or {}
        self.inspector_panel.set_family_reject_report(meta.get("family_reject_report"))
        self.results_panel.set_pipeline_health(
            meta.get("pipeline_audit"),
            warning=str(meta.get("ai_fallback_warning") or ""),
        )
        if meta.get("ai_fallback_warning"):
            QMessageBox.warning(
                self,
                "AI Pipeline",
                str(meta["ai_fallback_warning"]),
            )

        if meta.get("scope_fallback_used"):
            self.status_bar.set_message(
                "Kapsam otomatik genişletildi — pasif/boş klasör yerine tüm aktif kaynaklar tarandı"
            )

        self._maybe_offer_on_demand_scan(response)

        if self._is_simple_mode():
            self._update_user_status_bar()
            if above == 0:
                self.results_panel.show_empty_results(
                    "Sonuç bulunamadı. Semantik aramayı veya kapsamı genişletmeyi deneyin."
                )
        else:
            self.status_bar.set_message(
                f"Arama tamamlandı ({response.stats.search_ms:.0f} ms)"
            )

        meta = response.meta or {}
        qtext = (
            meta.get("text")
            or meta.get("text_query")
            or response.stats.text_query
            or ""
        )
        method = meta.get("search_method", "")
        if meta.get("mode") == "text":
            method = meta.get("search_method", "metin")
        elif qtext and meta.get("query_path"):
            method = "görsel + metin"
        elif meta.get("mode") == "image":
            method = "görsel"
        above = response.stats.above_threshold
        # Yazım düzeltme bildirimi
        spell_corrected = meta.get("spell_corrected", "")
        if spell_corrected and meta.get("spell_was_corrected"):
            self.search_header.set_search_summary(
                f"Arama: {qtext} → ✎ {spell_corrected} | Yöntem: {method} | Sonuç: {above:,}"
            )
        elif qtext:
            self.search_header.set_search_summary(
                f"Arama: {qtext} | Yöntem: {method} | Sonuç: {above:,}"
            )
        self._update_scope_hint()
        summary = str(meta.get("multi_summary") or "").strip()
        if summary:
            n = int(meta.get("multi_total") or 0)
            method = "çoklu görsel"
            self.search_header.set_search_summary(
                f"{summary} | Yöntem: {method} | Sonuç: {above:,}"
            )
            self.query_panel.set_search_mode_label(summary)
            self.status_bar.set_message(summary)

    def _on_multi_image_progress(self, current: int, total: int) -> None:
        self.status_bar.set_message(f"{int(current)} / {int(total)}")
        self.query_panel.set_load_status(f"{int(current)} / {int(total)}")

    def _apply_query_images(self, paths: list[str], *, auto_search: bool = True) -> None:
        from core.multi_image_search import collect_query_images

        collected = collect_query_images(paths)
        if not collected:
            self.status_bar.set_message("Desteklenen görsel bulunamadı")
            return
        self._query_image_paths = collected
        many = len(collected) > 1
        self._suppress_instant_search = many
        try:
            self._set_query_image(collected[0], auto_search=auto_search and not many)
        finally:
            self._suppress_instant_search = False
        if many:
            self.query_panel.set_load_status(f"{len(collected)} görsel seçildi")
            if auto_search:
                self._run_image_search(use_crop=False)

    def _set_query_image(self, path: str, *, auto_search: bool = False) -> None:

        self._last_query_path = path
        if path and path not in self._query_image_paths:
            self._query_image_paths = [path]

        self.query_panel.set_image(path)
        self.query_panel.set_query_thumbs(self._query_image_paths)

        self.search_header.set_has_image(True)
        self.query_panel.set_load_status("Görsel yüklendi")

        if auto_search:
            self.query_panel.set_load_status("Görsel aranıyor…")

    def _pick_query_image(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Sorgu Görseli",
            "",
            "Görseller (*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp *.psd *.ai)",
        )
        if paths:
            self._apply_query_images(list(paths), auto_search=True)

    def _pick_query_files(self) -> None:
        self._pick_query_image()

    def _pick_folder(self) -> None:

        folder = QFileDialog.getExistingDirectory(self, "Klasör Seç")

        if not folder:
            return

        self.settings.selected_folder_path = folder

        self.settings.search_scope = SearchScope.SELECTED_FOLDER.value

        idx = self.cmb_scope.findData(SearchScope.SELECTED_FOLDER.value)

        if idx >= 0:
            self.cmb_scope.setCurrentIndex(idx)

        self.settings.save()

        self.status_bar.set_message(f"Klasör seçildi: {folder}")

    def _maybe_index_before_search(self, path: str) -> None:

        folder = str(Path(path).parent)

        def check_and_prompt(count: int) -> None:
            if count > 0:
                return
            reply = QMessageBox.question(
                self,
                "Index Gerekli",
                f"Bu klasör indexlenmemiş:\n{folder}\n\nHızlı index oluşturulsun mu?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._quick_index_folder_async(folder)

        self._run_background_task(
            lambda: self.db_count_in_folder(folder),
            working_message="",
            success_message="",
            on_done=check_and_prompt,
        )

    def db_count_in_folder(self, folder: str) -> int:

        from core.db import Database

        db = Database(self.settings.db_path)

        return len(db.get_indexed_files(folder_prefix=folder))

    def _quick_folder_search(self) -> None:

        if not self._last_query_path:
            QMessageBox.information(self, "Bilgi", "Önce sorgu görseli seçin.")

            return

        folder = QFileDialog.getExistingDirectory(self, "Hızlı Arama Klasörü")

        if not folder:
            return

        self._quick_index_folder_async(folder)

        self.settings.quick_search_folder = folder

        self.settings.search_scope = SearchScope.QUICK_FOLDER.value

        QTimer.singleShot(800, lambda: self._run_image_search(use_crop=False))

    def _on_quick_index_qthread_finished(self) -> None:
        sender = self.sender()
        try:
            if sender is not None:
                sender.deleteLater()
        except RuntimeError:
            pass
        if sender is self._quick_index_worker:
            self._quick_index_worker = None
        self._quick_finish_hooked = False
        pending = self._pending_quick_index
        if pending is not None:
            self._pending_quick_index = None
            self._launch_quick_index_worker(pending)

    def _quick_index_folder_async(self, folder: str) -> None:
        old = self._quick_index_worker
        if old is not None and old.isRunning():
            self._pending_quick_index = folder
            old.request_stop()
            if hasattr(old, "arm_delete_later_on_finished"):
                old.arm_delete_later_on_finished()
            if not self._quick_finish_hooked:
                self._quick_finish_hooked = True
                old.finished.connect(self._on_quick_index_qthread_finished)
            self.status_bar.set_message(f"Hızlı index (bekleniyor): {folder}…")
            return
        self._launch_quick_index_worker(folder)

    def _launch_quick_index_worker(self, folder: str) -> None:
        old = self._quick_index_worker
        if old is not None and old.isRunning():
            self._pending_quick_index = folder
            old.request_stop()
            if hasattr(old, "arm_delete_later_on_finished"):
                old.arm_delete_later_on_finished()
            if not self._quick_finish_hooked:
                self._quick_finish_hooked = True
                old.finished.connect(self._on_quick_index_qthread_finished)
            return
        if old is not None and not old.isRunning():
            try:
                if hasattr(old, "arm_delete_later_on_finished"):
                    old.arm_delete_later_on_finished()
                else:
                    old.deleteLater()
            except RuntimeError:
                pass
            self._quick_index_worker = None

        self.status_bar.set_message(f"Hızlı index: {folder}…")

        self._quick_index_worker = QuickIndexWorker(self.settings, folder, parent=self)

        self._quick_index_worker.fast_done.connect(
            lambda s: self.status_bar.set_message(
                f"Hızlı sonuçlar hazır ({s.get('processed', 0)} dosya). Derin analiz…"
            )
        )

        self._quick_index_worker.finished_ok.connect(lambda _: self._refresh_status())

        if hasattr(self._quick_index_worker, "arm_delete_later_on_finished"):
            self._quick_index_worker.arm_delete_later_on_finished()
        self._quick_index_worker.start()

    def _add_source(self) -> None:

        dlg = SourceDialog(self)

        if dlg.exec():
            data = dlg.get_data()

            if not data.get("root_path"):
                QMessageBox.warning(self, "Uyarı", "Kök yol gerekli.")

                return

            try:
                sid = int(
                    self._source_manager.add_source(
                        name=data["name"],
                        root_path=data["root_path"],
                        source_type=data.get("source_type"),
                        is_active=bool(data.get("is_active", True)),
                        scan_interval_hours=data.get("scan_interval_hours"),
                        deep_scan_interval_days=data.get("deep_scan_interval_days"),
                    )
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Geçersiz kaynak", str(exc))
                return

            # Mevcut ana arşivi yeni klasörle ezme (kaynak 8/9 kapsamı korunur).
            if not str(self.settings.archive_root or "").strip():
                self.settings.archive_root = data["root_path"]

            ids = [
                int(x)
                for x in (self.settings.selected_source_ids or [])
                if int(x or 0) > 0
            ]
            if sid not in ids:
                ids.append(sid)
            self.settings.selected_source_ids = ids
            self.settings.save()
            try:
                from core.app_status import invalidate_status_cache

                invalidate_status_cache()
            except Exception:
                pass

            src_row = self._source_manager.get_source(sid) or {}
            try:
                self._source_manager.db.update_source_stats(
                    sid,
                    file_count=int(src_row.get("file_count") or 0),
                    error_count=int(src_row.get("error_count") or 0),
                    cache_status="queued",
                )
            except Exception:
                pass

            self._source_live_stats[sid] = {
                "found": int(src_row.get("file_count") or 0),
                "indexed": 0,
                "queued": 0,
                "label": "Kuyrukta",
            }
            self._status_generation += 1
            self._reload_source_table()
            self._refresh_folder_watch_async()
            self.status_bar.set_message("Yeni kaynak taranıyor...")
            self._begin_new_source_live_index(sid)

    def _begin_new_source_live_index(self, source_id: int) -> None:
        """Sayımı ayrı walk etmeden mevcut FAST discovery+drain'i başlat."""
        sid = int(source_id or 0)
        if sid <= 0:
            return
        self._include_source_in_running_index(sid)
        if self._index_is_running():
            self._kickoff_source_fast_index(sid)
            return
        self._run_index(
            scan_mode="quick",
            source_id=sid,
            quiet=True,
            index_mode="fast_archive",
        )

    def _on_source_count_qthread_finished(self, source_id: int) -> None:
        sid = int(source_id or 0)
        sender = self.sender()
        try:
            if sender is not None:
                sender.deleteLater()
        except RuntimeError:
            pass
        cur = self._source_count_workers.get(sid)
        if sender is cur:
            self._source_count_workers.pop(sid, None)
        pending_root = self._pending_source_counts.pop(sid, None)
        if pending_root is not None:
            self._start_source_file_count(sid, pending_root)

    def _start_source_file_count(self, source_id: int, root_path: str) -> None:
        sid = int(source_id or 0)
        if sid <= 0:
            return
        old = self._source_count_workers.get(sid)
        if old is not None and old.isRunning():
            # Same sid: pending replace; do not orphan-destroy running worker
            self._pending_source_counts[sid] = str(root_path or "")
            old.request_stop()
            if hasattr(old, "arm_delete_later_on_finished"):
                old.arm_delete_later_on_finished()
            # Re-hook once via unique attribute
            if not getattr(old, "_count_finish_hooked", False):
                old._count_finish_hooked = True
                old.finished.connect(
                    lambda *, s=sid: self._on_source_count_qthread_finished(s)
                )
            return
        if old is not None:
            self._source_count_workers.pop(sid, None)
            try:
                if hasattr(old, "arm_delete_later_on_finished"):
                    old.arm_delete_later_on_finished()
                elif not old.isRunning():
                    old.deleteLater()
            except RuntimeError:
                pass
        worker = SourceFileCountWorker(root_path, source_id=sid, parent=self)
        worker.progress.connect(
            lambda n, s=sid: self._on_source_file_count_progress(s, int(n))
        )
        worker.finished_ok.connect(
            lambda n, s=sid: self._on_source_file_count_finished(s, int(n))
        )
        worker.error.connect(
            lambda msg, s=sid: self._on_source_file_count_error(s, str(msg))
        )
        if hasattr(worker, "arm_delete_later_on_finished"):
            worker.arm_delete_later_on_finished()
        self._source_count_workers[sid] = worker
        worker.start()

    def _on_source_file_count_progress(self, source_id: int, found: int) -> None:
        sid = int(source_id or 0)
        n = int(found or 0)
        self.status_bar.set_message(
            f"Dosyalar taranıyor... {n:,}".replace(",", ".")
        )
        src = self._source_manager.get_source(sid) or {}
        try:
            self._source_manager.db.update_source_stats(
                sid,
                file_count=n,
                error_count=int(src.get("error_count") or 0),
                cache_status="scanning",
            )
        except Exception:
            pass
        live = dict(self._source_live_stats.get(sid) or {})
        live["found"] = n
        live["label"] = f"Dosyalar taranıyor... {n:,}".replace(",", ".")
        self._source_live_stats[sid] = live
        self._reload_source_table()

    def _on_source_file_count_finished(self, source_id: int, found: int) -> None:
        sid = int(source_id or 0)
        self._source_count_workers.pop(sid, None)
        n = int(found or 0)
        src = self._source_manager.get_source(sid) or {}
        try:
            self._source_manager.db.update_source_stats(
                sid,
                file_count=n,
                error_count=int(src.get("error_count") or 0),
                cache_status="queued",
            )
        except Exception:
            pass
        self._source_live_stats.pop(sid, None)
        self._reload_source_table()
        self.status_bar.set_message(
            f"{n:,} dosya bulundu — indexleme başlıyor...".replace(",", ".")
        )
        self._run_index(scan_mode="quick", source_id=sid, quiet=True)

    def _on_source_file_count_error(self, source_id: int, message: str) -> None:
        sid = int(source_id or 0)
        self._source_count_workers.pop(sid, None)
        self._source_live_stats.pop(sid, None)
        self.status_bar.set_message(f"Dosya sayımı başarısız: {message}")
        self._run_index(scan_mode="quick", source_id=sid, quiet=True)

    def _drop_source_from_running_index(self, source_id: int) -> None:
        sid = int(source_id or 0)
        if sid <= 0:
            return
        for w in list(self._alive_index_workers().values()):
            drop = getattr(w, "drop_source_id", None)
            if callable(drop):
                try:
                    drop(sid)
                except Exception:
                    pass
        try:
            from pathlib import Path as _P
            from core.index_v3.queues import JobStore

            job_db = _P(str(self.settings.db_path)).with_name(
                _P(str(self.settings.db_path)).stem + ".v3jobs.db"
            )
            if job_db.is_file():
                JobStore(job_db).delete_jobs_for_source(sid)
        except Exception:
            pass

    def _include_source_in_running_index(self, source_id: int) -> None:
        """Yeni kaynak genel indeks kapsamına girsin; sidecar ayrı toplam yaratmasın."""
        sid = int(source_id or 0)
        if sid <= 0:
            return
        for w in list(self._alive_index_workers().values()):
            if int(getattr(w, "source_id", 0) or 0) > 0:
                continue
            include = getattr(w, "include_source_id", None)
            if callable(include):
                try:
                    include(sid)
                except Exception:
                    pass

    def _source_fast_index_active(self, source_id: int) -> bool:
        w = self._source_fast_workers.get(int(source_id or 0))
        return bool(w is not None and w.isRunning())

    def _kickoff_source_fast_index(self, source_id: int) -> None:
        """Mevcut Hızlı İndeks (IndexWorker fast_archive) zincirini yeni kaynağa bağla."""
        sid = int(source_id or 0)
        if sid <= 0:
            return
        if self._source_fast_index_active(sid):
            return
        src = self._source_manager.get_source(sid) or {}
        if not str(src.get("root_path") or "").strip():
            return

        worker = IndexWorker(
            settings=self.settings,
            source_id=sid,
            scan_mode="quick",
            index_mode="fast_archive",
        )
        worker.progress.connect(
            lambda payload, s=sid: self._on_source_fast_index_progress(s, payload)
        )
        worker.finished_ok.connect(
            lambda result, s=sid: self._on_source_fast_index_finished(s, result)
        )
        worker.error.connect(
            lambda message, s=sid: self._on_source_fast_index_error(s, message)
        )
        self._include_source_in_running_index(sid)
        self._source_fast_workers[sid] = worker
        worker.start()

    def _merge_source_live(self, sources: list) -> list:
        out = []
        for src in sources or []:
            row = dict(src)
            live = self._source_live_stats.get(int(row.get("id") or 0))
            if live:
                row["file_count"] = int(live.get("found") or row.get("file_count") or 0)
                row["cache_status"] = "scanning"
                row["status_label"] = str(
                    live.get("label")
                    or f"Taranıyor · {int(live.get('indexed') or 0)}/{int(live.get('found') or 0)}"
                    f" · kuyruk {int(live.get('queued') or 0)}"
                )
            out.append(row)
        return out

    def _on_source_fast_index_progress(self, source_id: int, payload: dict) -> None:
        import time

        sid = int(source_id or 0)
        if sid <= 0:
            return
        stage = str((payload or {}).get("stage") or "")
        if stage in ("index_starting", "index_ready"):
            return
        src = self._source_manager.get_source(sid) or {}
        live = dict(self._source_live_stats.get(sid) or {})
        found = max(
            int(live.get("found") or 0),
            int(src.get("file_count") or 0),
            int((payload or {}).get("found") or 0),
        )
        indexed = max(
            int(live.get("indexed") or 0),
            int((payload or {}).get("indexed") or 0),
            int((payload or {}).get("processed") or 0),
        )
        queued = int(
            (payload or {}).get("queued")
            or (payload or {}).get("remaining_display")
            or live.get("queued")
            or 0
        )
        shown = max(found, indexed)
        live.update(
            {
                "found": found,
                "indexed": indexed,
                "queued": queued,
                "label": (
                    f"Taranıyor · {indexed:,}/{found:,} · kuyruk {queued:,}".replace(
                        ",", "."
                    )
                ),
            }
        )
        self._source_live_stats[sid] = live
        now = time.monotonic()
        last = float(self._source_fast_progress_at.get(sid) or 0)
        prev = int(self._source_fast_shown.get(sid, -1))
        if shown == prev and (now - last) < 0.25:
            return
        self._source_fast_progress_at[sid] = now
        self._source_fast_shown[sid] = shown
        try:
            self._source_manager.db.update_source_stats(
                sid,
                file_count=shown,
                error_count=int(src.get("error_count") or 0),
                cache_status="scanning",
            )
        except Exception:
            pass
        self._reload_source_table()
        if payload.get("total") is not None:
            self.progress_panel.update_progress(
                {
                    "engine": "index_v3",
                    "total": int(payload.get("total") or 0),
                    "light_done": int(payload.get("light_done") or 0),
                    "live_fast": dict(payload.get("live_fast") or {}),
                }
            )
        self.status_bar.set_message(
            f"Kaynak #{sid} taranıyor — bulunan {found:,} · "
            f"indeks {indexed:,} · kuyruk {queued:,}".replace(",", ".")
        )

    def _on_source_fast_index_finished(self, source_id: int, result: dict) -> None:
        sid = int(source_id or 0)
        self._source_fast_workers.pop(sid, None)
        self._source_fast_progress_at.pop(sid, None)
        self._source_fast_shown.pop(sid, None)
        self._source_live_stats.pop(sid, None)
        try:
            self._source_manager.update_source_stats(sid)
        except Exception:
            pass
        src = self._source_manager.get_source(sid) or {}
        fc = int(src.get("file_count") or 0)
        disc = (result or {}).get("discovery") or {}
        scanned = 0
        for row in disc.get("sources") or []:
            scanned = max(
                scanned,
                int(row.get("scanned") or 0),
                int(row.get("inserted") or 0),
            )
        shown = max(fc, scanned)
        stopped = bool((result or {}).get("stopped"))
        cache = str(src.get("cache_status") or "")
        if stopped:
            cache = "partial" if shown else "queued"
        elif cache in ("", "queued", "scanning", "empty"):
            cache = "ok" if shown else "empty"
        try:
            self._source_manager.db.update_source_stats(
                sid,
                file_count=shown,
                error_count=int(src.get("error_count") or 0),
                cache_status=cache,
            )
        except Exception:
            pass
        self._status_generation += 1
        self._reload_source_table()
        self._refresh_status()
        if stopped:
            msg = f"Kaynak #{sid} hızlı indeks durdu — {shown:,} dosya".replace(",", ".")
        elif shown:
            msg = f"Kaynak #{sid} hızlı indeks tamamlandı — {shown:,} dosya".replace(",", ".")
        else:
            msg = f"Kaynak #{sid} taranamadı veya klasör boş"
        self.status_bar.set_message(msg)

    def _on_source_fast_index_error(self, source_id: int, message: str) -> None:
        sid = int(source_id or 0)
        self._source_fast_workers.pop(sid, None)
        src = self._source_manager.get_source(sid) or {}
        try:
            self._source_manager.db.update_source_stats(
                sid,
                file_count=int(src.get("file_count") or 0),
                error_count=int(src.get("error_count") or 0) + 1,
                cache_status="error",
            )
        except Exception:
            pass
        self._reload_source_table()
        self.status_bar.set_message(f"Kaynak #{sid} hızlı indeks hatası: {message}")

    def _edit_source(self, source_id: int) -> None:

        src = self._source_manager.get_source(source_id)

        if not src:
            return

        dlg = SourceDialog(self, source=src)

        if dlg.exec():
            data = dlg.get_data()

            data["id"] = source_id

            self._source_manager.db.upsert_source(data)

            self._status_generation += 1
            self._reload_source_table()
            self._refresh_status()
            self._refresh_folder_watch_async()

    def _toggle_source_active(self, source_id: int) -> None:
        src = self._source_manager.get_source(source_id)
        if not src:
            return
        active = not bool(src.get("is_active"))
        self._source_manager.toggle_active(source_id, active)
        self._status_generation += 1
        self._reload_source_table()
        self._refresh_status()

    def _detach_source(self, source_id: int) -> None:
        src = self._source_manager.get_source(source_id)
        if not src:
            return
        name = str(src.get("name") or source_id)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Kaynağı Kaldır")
        box.setText(f'「{name}」 kaynağını nasıl kaldırmak istiyorsunuz?')
        box.setInformativeText(
            "Orijinal dosyalar diskten silinmez.\n\n"
            "• Index koru: kaynak satırı silinir; DB kayıtları yetim kalır "
            "(Tüm Arşiv'e girmez).\n"
            "• Index temizle: kaynak + tüm index/cache kayıtları silinir."
        )
        btn_keep = box.addButton(
            "Kaynağı kaldır, index koru", QMessageBox.ButtonRole.AcceptRole
        )
        btn_purge = box.addButton(
            "Kaynağı kaldır + index temizle", QMessageBox.ButtonRole.DestructiveRole
        )
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None or clicked == box.button(QMessageBox.StandardButton.Cancel):
            return
        if clicked is btn_purge:
            # Mevcut "Sil ve Index Temizle" ile aynı yol
            self._purge_source_index(source_id)
            return
        # Index koru: yalnız sources satırını sil → orphan diagnostic
        self._source_manager.remove_source(source_id)
        self._source_live_stats.pop(int(source_id), None)
        w = self._source_fast_workers.pop(int(source_id), None)
        if w is not None:
            try:
                w.request_stop()
            except Exception:
                pass
        self._drop_source_from_running_index(source_id)
        # seçim listesinden düş
        selected = [
            int(x)
            for x in (self.settings.selected_source_ids or [])
            if int(x) > 0 and int(x) != int(source_id)
        ]
        self.settings.selected_source_ids = selected
        self.settings.save()
        try:
            from core.app_status import invalidate_status_cache

            invalidate_status_cache()
        except Exception:
            pass
        self._status_generation += 1
        self._refresh_status()
        self.status_bar.set_message(
            f"Kaynak kaldırıldı (index korundu → yetim olabilir): {name}"
        )
    def _purge_source_index(self, source_id: int) -> None:
        src = self._source_manager.get_source(source_id)
        if not src:
            return
        if self._purge_worker and self._purge_worker.isRunning():
            QMessageBox.information(
                self, "Bilgi", "Index temizleme zaten devam ediyor."
            )
            return
        reply = QMessageBox.question(
            self,
            "Index Kayıtlarını Temizle",
            f"「{src.get('name', '')}」 kaynağının index kayıtları silinecek.\n\n"
            "Orijinal dosyalar silinmez. Büyük kaynaklarda işlem arka planda yapılır.\n"
            "Yeniden kullanmak için kaynağı tekrar taramak gerekir.\n\nDevam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if self._index_is_running():
            self._stop_index()
        self._set_source_actions_enabled(False)
        self.status_bar.set_busy(True)
        self.status_bar.set_message(f"Index temizleniyor: {src.get('name', '')}…")
        self._purge_worker = PurgeSourceWorker(
            self.settings,
            source_id,
            source_name=str(src.get("name", "")),
            parent=self,
        )
        self._purge_worker.progress.connect(self._on_purge_progress)
        self._purge_worker.finished_ok.connect(self._on_purge_finished)
        self._purge_worker.error.connect(self._on_purge_error)
        self._purge_worker.start()

    def _purge_orphan_indexes(self) -> None:
        """Yetim (hayalet) index kayıtlarını açık onayla temizle."""
        from core.db import Database
        from core.index_v3.scope import resolve_index_scope

        if self._purge_worker and self._purge_worker.isRunning():
            QMessageBox.information(
                self, "Bilgi", "Index temizleme zaten devam ediyor."
            )
            return
        db = Database(self.settings.db_path)
        scope = resolve_index_scope(db, [])
        n = int(scope.orphan_file_total or 0)
        if n <= 0:
            QMessageBox.information(self, "Bilgi", "Temizlenecek yetim kayıt yok.")
            return
        reply = QMessageBox.question(
            self,
            "Yetim Kayıtları Temizle",
            f"{n:,} yetim index kaydı silinecek.\n\n"
            "Bunlar sources tablosunda olmayan hayalet kayıtlardır.\n"
            "Orijinal dosyalar diskten silinmez.\n\nDevam edilsin mi?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if self._index_is_running():
            self._stop_index()
        self._set_source_actions_enabled(False)
        self.status_bar.set_busy(True)
        self.status_bar.set_message(f"Yetim kayıtlar temizleniyor ({n:,})…")
        try:
            result = self._source_manager.purge_orphan_indexes(
                self.settings.cache_dir,
                progress_callback=lambda data: self._on_purge_progress(data),
            )
            self._on_purge_finished(result)
        except Exception as exc:
            self._on_purge_error(str(exc))

    def _set_source_actions_enabled(self, enabled: bool) -> None:
        panel = self.source_sidebar.sources_panel
        for btn in (
            panel.btn_add,
            panel.btn_edit,
            panel.btn_toggle,
            panel.btn_detach,
            panel.btn_purge,
            panel.btn_scan_sel,
            panel.btn_scan_all,
            panel.btn_scan_due,
        ):
            btn.setEnabled(enabled)

    def _on_purge_progress(self, data: dict) -> None:
        msg = data.get("message") or "Index temizleniyor…"
        total = data.get("total")
        if total:
            msg = f"{msg} ({total:,} kayıt)"
        self.status_bar.set_message(msg)

    def _on_purge_finished(self, result: dict) -> None:
        self._purge_worker = None
        self._set_source_actions_enabled(True)
        self.status_bar.set_busy(False)
        removed = int(result.get("files_removed", 0))
        self._refresh_status()
        self.status_bar.set_message(f"Index temizlendi: {removed:,} kayıt silindi")
        QMessageBox.information(
            self,
            "Tamamlandı",
            f"{removed:,} index kaydı silindi.\n"
            "Orijinal dosyalar yerinde kaldı.\n"
            "Eski thumbnail/cache dosyaları için Cache sekmesinden temizlik yapabilirsiniz.",
        )

    def _on_purge_error(self, message: str) -> None:
        self._purge_worker = None
        self._set_source_actions_enabled(True)
        self.status_bar.set_busy(False)
        QMessageBox.critical(self, "Index Temizleme Hatası", message)
        self.status_bar.set_message("Index temizleme başarısız")

    def _on_scope_changed(self, scope: str) -> None:
        from core.sources import SearchScope

        self.settings.search_scope = scope
        if scope == SearchScope.SELECTED_SOURCES.value:
            self.settings.selected_source_ids = (
                self.source_sidebar.sources_panel.selected_source_ids()
            )
        self.settings.save()
        self._update_scope_hint()

    def _warmup_sources_async(self) -> None:
        """Open SourceManager DB off the UI thread after the window can paint."""
        if getattr(self, "_sources_warmup_running", False):
            return
        self._sources_warmup_running = True

        def warm():
            # Touch DB (migrate/open) + light source list only — no full rescan.
            return self._source_manager.list_sources(active_only=False)

        def apply(_sources) -> None:
            self._sources_warmup_running = False
            self._sources_ready = True
            self._repair_stale_search_scope()
            self._update_scope_hint()

        self.search_header.set_scope_hint("Kapsam: Hazırlanıyor…")
        self._run_background_task(
            warm,
            working_message="",
            success_message="",
            on_done=apply,
        )

    def _update_scope_hint(self) -> None:
        from core.sources import SearchScope

        if not getattr(self, "_sources_ready", False):
            self.search_header.set_scope_hint("Kapsam: Hazırlanıyor…")
            return

        sources = self._source_manager.list_sources(active_only=True)
        active = len(sources)
        server_on = any(s.get("source_type") == "server_share" for s in sources)
        local_on = any(s.get("source_type") == "local_pc" for s in sources)
        scope = self.settings.search_scope
        if scope == SearchScope.SELECTED_SOURCES.value:
            n = len(self.settings.selected_source_ids or [])
            text = f"Kapsam: {n} seçili kaynak"
        elif scope == SearchScope.SELECTED_FOLDER.value:
            pool = self._indexed_pool_size()
            text = f"Kapsam: klasör — {self.settings.selected_folder_path or '—'} | Havuz: {pool:,}"
            if pool == 0:
                text += " ⚠ boş"
        else:
            pool = self._indexed_pool_size()
            text = (
                f"Kapsam: {active} aktif kaynak | Havuz: {pool:,} | "
                f"Server: {'açık' if server_on else 'kapalı'} | "
                f"Local: {'açık' if local_on else 'kapalı'}"
            )
        self.search_header.set_scope_hint(text)

    def _on_source_selected(self, source_id: int) -> None:

        src = self._source_manager.get_source(source_id)

        if not src:
            return

        self.settings.selected_source_id = source_id

        self.settings.selected_folder_path = src.get("root_path", "")

        self.settings.save()

        self._update_scope_hint()

    def _on_source_checks_changed(self) -> None:

        self.settings.selected_source_ids = self.sources_panel.selected_source_ids()

        self.settings.save()

        if self.settings.search_scope == SearchScope.SELECTED_SOURCES.value:
            self._on_filter_changed_research()

        # Yüzde paydası seçili kaynağa göre — cache düşür, hemen yenile
        try:
            from core.app_status import invalidate_status_cache

            invalidate_status_cache()
        except Exception:
            pass
        self._refresh_status()

    def _scan_source(self, source_id: int, scan_mode: str) -> None:

        self._run_index(scan_mode=scan_mode, source_id=source_id)

    def _scan_due(self) -> None:

        self._run_index(run_due_only=True)

    def _start_scheduler_safe(self) -> None:
        """Auto-scan probe must not open DB / SourceManager on the UI thread."""
        if self._shutting_down:
            return

        self.settings.auto_scan_on_startup = self.source_sidebar.is_auto_scan_enabled()

        if not self.settings.auto_scan_on_startup:
            return

        def probe() -> bool:
            scheduler = ScanScheduler(self.settings)
            return bool(scheduler.should_auto_scan_on_startup())

        def apply(should_run) -> None:
            if self._shutting_down or not should_run:
                return
            self._run_index(run_due_only=True, quiet=True)

        self._run_background_task(
            probe,
            working_message="",
            success_message="",
            on_done=apply,
        )

    def _load_customers(self) -> None:
        import os

        if os.environ.get("VEZIR_QA_NONINTERACTIVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            self._customer_load_running = False
            self._apply_customers([])
            return
        if self._customer_load_running:
            return
        self._customer_load_running = True

        def load():
            from core.customer_discovery import discover_customers_from_db, set_customer_registry
            from core.db import Database

            db = Database(self.settings.db_path)
            reg = discover_customers_from_db(db)
            set_customer_registry(reg)
            return reg

        def apply(reg) -> None:
            self._customer_load_running = False
            self._customer_registry = reg
            self._apply_customers(reg.names() if reg else [])

        self._run_background_task(
            load,
            working_message="",
            success_message="",
            on_done=apply,
        )

    def _apply_customers(self, customers: list[str]) -> None:
        self.cmb_customer.clear()
        self.cmb_customer.addItem("Tümü", "")
        for c in customers:
            self.cmb_customer.addItem(c, c)
        if self.settings.customer_filter:
            idx = self.cmb_customer.findData(self.settings.customer_filter)
            if idx >= 0:
                self.cmb_customer.setCurrentIndex(idx)
            self.search_header.set_selected_customer(self.settings.customer_filter)
        # Boş alan / ilk yüklemede sık kullanılanlar
        labels = list(customers)
        reg = self._customer_registry
        if reg is not None:
            sug = reg.suggest("", limit=12, recent=self._recent_customers)
            labels = [s.label for s in sug] or labels
        self.search_header.set_customer_suggestions(labels)
        n = len(customers)
        from core.logger import setup_logger

        setup_logger("customer_ui").info(
            "Müşteri/klasör keşfi UI: %d aday yüklendi", n
        )

    def _on_header_customer_changed(self, name: str) -> None:
        name = (name or "").strip()
        self.settings.customer_filter = name
        if name:
            self._recent_customers = [name] + [
                x for x in self._recent_customers if x != name
            ]
            self._recent_customers = self._recent_customers[:12]
        idx = self.cmb_customer.findData(name)
        if name and idx >= 0:
            self.cmb_customer.blockSignals(True)
            self.cmb_customer.setCurrentIndex(idx)
            self.cmb_customer.blockSignals(False)
        elif not name:
            self.cmb_customer.blockSignals(True)
            self.cmb_customer.setCurrentIndex(0)
            self.cmb_customer.blockSignals(False)
        hint = f"Müşteri kapsamı: {name}" if name else ""
        self.search_header.set_scope_hint(hint)

    def _on_customer_query_edited(self, text: str) -> None:
        reg = self._customer_registry
        if reg is None:
            return
        sug = reg.suggest(text or "", limit=12, recent=self._recent_customers)
        self.search_header.set_customer_suggestions([s.label for s in sug])

    def _alive_index_workers(self) -> dict:
        from core.index_v3.ui_bridge import index_lane_for_mode

        out: dict[str, IndexWorker] = {}
        for lane, w in (self._index_workers or {}).items():
            if w is not None and w.isRunning():
                out[str(lane)] = w
        w0 = self._index_worker
        if w0 is not None and w0.isRunning():
            lane = index_lane_for_mode(getattr(w0, "index_mode", "") or "")
            out.setdefault(lane, w0)
        return out

    def _index_is_running(self) -> bool:
        return bool(self._alive_index_workers())

    def _start_index(
        self, scan_mode: str = "quick", index_mode: str = "complete"
    ) -> None:

        self._run_index(scan_mode=scan_mode, index_mode=index_mode)

    def _sync_index_controls_with_worker(self) -> bool:
        """Worker yok/ölüyse UI'yi kilitli 'Çalışıyor'dan kurtar. True = hâlâ çalışıyor."""
        if self._alive_index_workers():
            return True
        self._index_worker = None
        self._index_workers = {}
        self._index_active = False
        self._index_paused = False
        state = str(
            getattr(self.progress_panel, "_control_state", "") or ""
        ).lower()
        if state in ("running", "paused"):
            self.progress_panel.set_control_state("ready")
            self.progress_panel.freeze_progress(False)
            self.progress_panel.end_session()
        return False

    def _schedule_pending_index_watchdog(self) -> None:
        """Worker bitmezse pending restart takılı kalmasın."""
        from PySide6.QtCore import QTimer

        QTimer.singleShot(3000, self._pending_index_watchdog_tick)

    def _pending_index_watchdog_tick(self) -> None:
        pending = self._pending_index_restart
        if not pending:
            return
        if self._index_is_running():
            # Hâlâ kapanıyor — bir tur daha bekle (max ~30s toplam ayrı tick'lerle)
            tries = int(pending.get("_watch_tries", 0) or 0) + 1
            pending["_watch_tries"] = tries
            if tries < 10:
                self._schedule_pending_index_watchdog()
                return
            # Zombie: referansı bırak, yeni modu zorla başlat
            for w in list(self._alive_index_workers().values()):
                try:
                    w.stop()
                except Exception:
                    pass
            self._index_worker = None
            self._index_workers = {}
        self._pending_index_restart = None
        self.status_bar.set_message("Index yeniden başlatılıyor…")
        from PySide6.QtCore import QTimer

        QTimer.singleShot(
            0,
            lambda p=pending: self._run_index(
                scan_mode=p["scan_mode"],
                source_id=p.get("source_id") or 0,
                run_due_only=bool(p.get("run_due_only")),
                quiet=bool(p.get("quiet")),
                index_mode=p.get("index_mode") or "complete",
            ),
        )

    def _run_index(
        self,
        scan_mode="quick",
        source_id=0,
        run_due_only=False,
        quiet=False,
        index_mode="complete",
    ) -> None:
        from core.index_session import mode_display_label

        mode = str(index_mode or "complete")
        mode_label = mode_display_label(mode)

        sources = self._source_manager.list_sources(active_only=True)

        if not sources and not run_due_only:
            if not quiet:
                QMessageBox.warning(self, "Uyarı", "Aktif kaynak yok.")

            return

        # Butonlar kilitli / worker ölü senkronu
        still_running = self._sync_index_controls_with_worker()

        from core.index_v3.ui_bridge import index_lane_for_mode, lanes_are_parallel

        new_lane = index_lane_for_mode(mode)
        alive = self._alive_index_workers()
        still_running = bool(alive) or still_running

        if still_running:
            same_w = alive.get(new_lane)
            same_source = True
            if same_w is not None:
                same_source = int(source_id or 0) == int(
                    getattr(same_w, "source_id", 0) or 0
                )
            if (
                same_w is not None
                and same_source
                and not run_due_only
                and not self._pending_index_restart
            ):
                self.status_bar.set_message(
                    f"Index zaten çalışıyor — {mode_label}"
                )
                self.progress_panel.set_active_index_mode(
                    getattr(same_w, "index_mode", None) or mode, status="running"
                )
                return
            conflicts = [
                w
                for lane, w in alive.items()
                if lane != new_lane and not lanes_are_parallel(lane, new_lane)
            ]
            if conflicts:
                self._pending_index_restart = {
                    "scan_mode": scan_mode,
                    "source_id": source_id,
                    "run_due_only": run_due_only,
                    "quiet": quiet,
                    "index_mode": mode,
                    "_watch_tries": 0,
                }
                self._index_active = False
                self._index_paused = False
                for w in conflicts:
                    w.stop()
                if self._index_worker and self._index_worker in conflicts:
                    pass
                self.progress_panel.set_indexing(False, terminal="stopped")
                self.progress_panel.set_active_index_mode(mode, status="stopped")
                self.status_bar.set_message(
                    f"Önceki index durduruluyor — sonra: {mode_label}"
                )
                self._schedule_pending_index_watchdog()
                return

        # Gerçekten başlıyoruz — modu şimdi kilitle
        self._current_index_mode = mode
        self.progress_panel.set_active_index_mode(mode, status="running")

        self._pending_index_restart = None
        self._index_paused = False
        self.progress_panel.freeze_progress(False)
        if self._cache_reconciliation_worker:
            self._cache_reconciliation_worker.pause()
        # Index scope SSOT: checkbox → settings.selected_source_ids
        # Dolu = yalnız seçilen; boş = tüm arşiv. UI ile settings asla ayrışmasın.
        try:
            self.settings.selected_source_ids = list(
                self.sources_panel.selected_source_ids()
            )
            self.settings.save()
            from core.app_status import invalidate_status_cache

            invalidate_status_cache()
        except Exception:
            pass
        keep = list(self._alive_index_workers().values())
        old = self._index_workers.get(new_lane)
        if old is not None and old not in keep:
            try:
                old.progress.disconnect(self._on_index_progress)
            except Exception:
                pass
            try:
                old.finished_ok.disconnect(self._on_index_finished)
            except Exception:
                pass
            try:
                old.error.disconnect(self._on_index_error)
            except Exception:
                pass
        worker = IndexWorker(
            self.settings,
            customer_filter=self.search_header.selected_customer()
            or (self.cmb_customer.currentData() or "")
            or "",
            scan_mode=scan_mode,
            source_id=source_id,
            run_due_only=run_due_only,
            index_mode=mode,
            parent=self,
        )
        self._index_workers[new_lane] = worker
        self._index_worker = worker

        worker.progress.connect(self._on_index_progress)

        worker.finished_ok.connect(self._on_index_finished)

        worker.error.connect(self._on_index_error)

        self.progress_panel.begin_session(index_mode=mode)
        self.progress_panel.set_indexing(True, paused=False)
        self._index_active = True
        self._index_progress_timer.start()
        worker.start()
        self.status_bar.set_message(f"Index devam ediyor… ({mode_label})")
        self._sync_post_ga_lane_buttons()

    def _stop_index(self) -> None:
        self._index_paused = False
        self._index_active = False
        self._pending_index_restart = None
        self._index_progress_timer.stop()
        # UI hemen Durdu — worker bitene kadar tıklama kilitlenmesin
        self.progress_panel.set_indexing(False, terminal="stopped")
        self.progress_panel.freeze_progress(True)
        self.status_bar.set_message("Index durduruluyor…")
        for w in list(self._alive_index_workers().values()):
            w.stop()
        if self._index_worker:
            self._index_worker.stop()
        for w in list((self._source_fast_workers or {}).values()):
            if w is not None and w.isRunning():
                w.stop()

    def _sync_post_ga_lane_buttons(self) -> None:
        alive = self._alive_index_workers()
        self.progress_panel.set_post_ga_lane_running(
            patch="patch" in alive,
            ocr="ocr" in alive,
        )

    def _stop_post_ga_lane(self, lane: str) -> None:
        """Manuel PATCH veya OCR durdur — tamamlanmış artifact silinmez."""
        key = str(lane or "")
        w = (self._index_workers or {}).get(key)
        if w is None or not w.isRunning():
            self._sync_post_ga_lane_buttons()
            return
        try:
            w.stop()
        except Exception:
            pass
        self.status_bar.set_message(
            "PATCH durduruluyor…" if key == "patch" else "OCR durduruluyor…"
        )

    def _on_index_error(self, message: str) -> None:
        from core.index_v3.ui_bridge import index_lane_for_mode

        src = self.sender()
        if src is not None:
            lane = None
            for k, w in list((self._index_workers or {}).items()):
                if w is src:
                    lane = k
                    break
            if lane is None:
                lane = index_lane_for_mode(getattr(src, "index_mode", "") or "")
            self._index_workers.pop(lane, None)
        still = self._alive_index_workers()
        if still:
            self._index_worker = next(iter(still.values()))
            self._index_active = True
            QMessageBox.critical(self, "Hata", message)
            self._refresh_status()
            return
        pending = self._pending_index_restart
        self._pending_index_restart = None
        self._index_active = False
        self._index_paused = False
        self._index_progress_timer.stop()
        self.status_bar.clear_indexing_progress()
        self.progress_panel.set_indexing(False, terminal="stopped")
        self._index_worker = None
        self._index_workers = {}
        if pending:
            self.status_bar.set_message("Hata sonrası index yeniden deneniyor…")
            QTimer.singleShot(
                0,
                lambda p=pending: self._run_index(
                    scan_mode=p["scan_mode"],
                    source_id=p.get("source_id") or 0,
                    run_due_only=bool(p.get("run_due_only")),
                    quiet=bool(p.get("quiet")),
                    index_mode=p.get("index_mode") or "complete",
                ),
            )
            return
        if self._cache_reconciliation_worker:
            self._cache_reconciliation_worker.resume()
        QMessageBox.critical(self, "Hata", message)
        self._refresh_status()

    def _sync_scope_selection_from_checkboxes(self) -> None:
        """Checkbox = UI SSOT. Status/worker settings.selected_source_ids ile asla ayrışmasın.

        [] → tüm arşiv; [1] → yalnız Karşıdan. Status yenilemeden ÖNCE çağrılmalı.
        """
        try:
            ids = [
                int(x)
                for x in (self.sources_panel.selected_source_ids() or [])
                if int(x) > 0
            ]
            prev = [
                int(x)
                for x in (self.settings.selected_source_ids or [])
                if int(x) > 0
            ]
            if ids != prev:
                self.settings.selected_source_ids = ids
                self.settings.save()
                try:
                    from core.app_status import invalidate_status_cache

                    invalidate_status_cache()
                except Exception:
                    pass
        except Exception:
            pass

    def _clear_lane_status_worker_ref_if_same(self, worker) -> None:
        """Clear Python ref when THAT lane StatusWorker C++ object is destroyed."""
        if getattr(self, "_lane_status_worker", None) is worker:
            self._lane_status_worker = None

    def _clear_status_worker_ref_if_same(self, worker) -> None:
        """Clear Python ref when THAT full StatusWorker C++ object is destroyed."""
        if getattr(self, "_status_worker", None) is worker:
            self._status_worker = None

    def _refresh_status_during_index(self) -> None:
        """Index sırasında yalnız SSOT lane COUNT — ağır dashboard'dan bağımsız."""
        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        if not self._index_active:
            return
        self._sync_scope_selection_from_checkboxes()
        # Ayrı worker: full status JOIN lanes yenilemesini ASLA engellemez
        cur = getattr(self, "_lane_status_worker", None)
        if qthread_is_running(cur):
            return
        # Zombie wrapper after deleteLater — drop Python ref before spawning.
        if cur is not None and not qobject_is_alive(cur):
            self._lane_status_worker = None
        w = StatusWorker(self.settings, parent=self, lanes_only=True)
        self._lane_status_worker = w
        w._status_gen = self._status_generation
        w.finished_ok.connect(self._on_status_updated)
        if hasattr(w, "arm_delete_later_on_finished"):
            w.arm_delete_later_on_finished()
        try:
            w.destroyed.connect(
                lambda *_a, worker=w: self._clear_lane_status_worker_ref_if_same(worker)
            )
        except Exception:
            pass
        w.start()


    def _refresh_status(self) -> None:

        if self._shutting_down:
            return

        self._sync_scope_selection_from_checkboxes()

        if self._index_active:
            # Index açıkken yalnız hızlı lanes — JOIN yok
            self._refresh_status_during_index()
            return

        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        cur = getattr(self, "_status_worker", None)
        if qthread_is_running(cur):
            self._status_refresh_queued = True
            return
        if cur is not None and not qobject_is_alive(cur):
            self._status_worker = None

        w = StatusWorker(self.settings, parent=self, lanes_only=False)
        self._status_worker = w
        w._status_gen = self._status_generation
        w.finished_ok.connect(self._on_status_updated)
        if hasattr(w, "arm_delete_later_on_finished"):
            w.arm_delete_later_on_finished()
        try:
            w.destroyed.connect(
                lambda *_a, worker=w: self._clear_status_worker_ref_if_same(worker)
            )
        except Exception:
            pass
        w.start()

    def _on_index_progress(self, data: dict) -> None:
        if str(data.get("stage") or "") == "source_discover":
            found = int(data.get("found") or 0)
            self.status_bar.set_message(
                f"Yeni kaynak taranıyor... {found:,}".replace(",", ".")
            )
            sid = int(data.get("source_id") or 0)
            if sid > 0:
                live = dict(self._source_live_stats.get(sid) or {})
                live["found"] = max(int(live.get("found") or 0), found)
                live["queued"] = int(data.get("queued") or live.get("queued") or 0)
                live["indexed"] = int(data.get("indexed") or live.get("indexed") or 0)
                live["label"] = (
                    f"Yeni kaynak taranıyor... {found:,}".replace(",", ".")
                )
                self._source_live_stats[sid] = live
                self._reload_source_table()
            if data.get("total") is not None:
                self.progress_panel.update_progress(
                    {
                        "engine": "index_v3",
                        "total": int(data.get("total") or 0),
                        "light_done": int(data.get("light_done") or 0),
                        "live_fast": dict(data.get("live_fast") or {}),
                    }
                )
            return
        if not self._index_active:
            return
        if self._current_index_mode and "index_mode" not in data:
            data = {**data, "index_mode": self._current_index_mode}
        self.progress_panel.update_progress(data)
        jobs_total = int(data.get("jobs_total", 0) or 0)
        jobs_done = int(data.get("jobs_done", 0) or 0)
        processed = int(data.get("processed", 0) or 0)
        if jobs_total > 0:
            pct = int(jobs_done / jobs_total * 100)
            self._index_percent = pct
            if not self._is_simple_mode():
                self.status_bar.set_indexing_progress(
                    pct,
                    f"({jobs_done}/{jobs_total})",
                )
                self.lbl_header_status.setText(
                    f"Index: %{pct} — {jobs_done:,}/{jobs_total:,} dosya"
                )
        elif processed > 0 and not self._is_simple_mode():
            self.lbl_header_status.setText(f"Index: {processed:,} dosya işlendi")
        if self._is_simple_mode():
            self._update_user_status_bar()
            return
        stage = str(data.get("stage", "") or "")
        if stage == "ai_loading":
            self.status_bar.set_message(
                "AI modeli yükleniyor (ilk dosya yavaş olabilir)…"
            )
        elif data.get("current_file"):
            self.status_bar.set_message(f"Index: {data['current_file']}")

    def _pause_index(self) -> None:
        if not self._index_is_running() or not self._index_active:
            return
        self._index_paused = True
        for w in self._alive_index_workers().values():
            w.pause()
        self.progress_panel.set_indexing(True, paused=True)
        self.progress_panel.freeze_progress(True)
        self.status_bar.set_message("Index duraklatıldı")

    def _resume_index(self) -> None:
        # Oturum içi pause → resume
        if self._index_is_running() and self._index_paused:
            self._index_paused = False
            for w in self._alive_index_workers().values():
                w.resume()
            self.progress_panel.freeze_progress(False)
            self.progress_panel.set_indexing(True, paused=False)
            self.status_bar.set_message("Kaldığı yerden devam ediyor…")
            return
        # Worker yoksa kalıcı oturumdan devam
        if self._index_is_running():
            return
        try:
            from core.db import Database
            from core.index_session import load_session

            session = load_session(Database(self.settings.db_path))
            mode = (
                session.index_mode
                if session and session.index_mode
                else self._current_index_mode or "complete"
            )
        except Exception:
            mode = self._current_index_mode or "complete"
        self.status_bar.set_message("Kaldığı yerden devam ediyor…")
        self.progress_panel.freeze_progress(False)
        self._run_index(scan_mode="quick", index_mode=mode)

    def _index_purge_broken(self) -> None:
        from core.index_maintenance import purge_broken_index_records

        result = purge_broken_index_records(self.settings)
        QMessageBox.information(
            self,
            "Index bakım",
            f"Bozuk kayıtlar temizlendi:\n"
            f"• processing sıfır: {result.get('processing_reset', 0)}\n"
            f"• failed sıfır: {result.get('failed_reset', 0)}\n"
            f"• kuyruk sıfır: {result.get('queue_reset', 0)}",
        )
        self._refresh_status()

    def _index_requeue_thumbs(self) -> None:
        from core.index_maintenance import requeue_missing_thumbnails

        result = requeue_missing_thumbnails(self.settings)
        QMessageBox.information(
            self,
            "Index bakım",
            f"Eksik thumbnail kuyruğa alındı: {result.get('queued', 0)} dosya",
        )
        self._refresh_status()

    def _index_queue_ai(self) -> None:
        from core.index_maintenance import queue_missing_ai_embeddings

        result = queue_missing_ai_embeddings(self.settings)
        if result.get("reason") == "ai_disabled":
            QMessageBox.warning(
                self, "Index bakım", "AI embedding kapalı — önce AI'yı açın."
            )
            return
        QMessageBox.information(
            self,
            "Index bakım",
            f"AI embedding eksik dosyalar kuyruğa alındı: {result.get('queued', 0)}",
        )
        self._refresh_status()

    def _start_post_ga(
        self, *, patch: bool, ocr: bool, rerun: bool = False
    ) -> None:
        from pathlib import Path

        from core.index_freeze import process_search_active

        if process_search_active():
            self.status_bar.set_message("Arama oturumu: PATCH/OCR başlatılmaz.")
            return
        sources = self._source_manager.list_sources(active_only=True)
        sids = [int(s["id"]) for s in sources if int(s.get("id") or 0) > 0]
        if not sids:
            return
        # Manuel OCR basışı motoru açar; worker aksi halde ocr_processed=1 no-op yazar.
        if ocr:
            self.settings.ocr_enabled = True
        db_path = str(self.settings.db_path)

        def enqueue_jobs():
            from core.db import Database
            from core.index_v3.planner import enqueue_post_ga
            from core.index_v3.queues import JobStore

            db = Database(db_path)
            base = Path(db_path)
            store = JobStore(base.with_name(base.stem + ".v3jobs.db"))
            return enqueue_post_ga(
                db,
                store,
                sids,
                patch=patch,
                ocr=ocr,
                reopen_done=bool(rerun),
            )

        def after_enqueue(n) -> None:
            n = int(n or 0)
            if n <= 0:
                self.status_bar.set_message(
                    "PATCH/OCR: Genel AI tamam değil veya iş yok."
                )
                return
            self.status_bar.set_message(f"PATCH/OCR kuyruğa alındı: {n}")
            if patch:
                self._start_index(scan_mode="quick", index_mode="patch")
            if ocr:
                self._start_index(scan_mode="quick", index_mode="ocr")

        self._run_background_task(
            enqueue_jobs,
            working_message="PATCH/OCR kuyruğa alınıyor…",
            success_message="",
            on_done=after_enqueue,
        )

    def _index_semantic_backfill(self) -> None:
        if not getattr(self.settings, "semantic_text_search_enabled", False):
            self.status_bar.set_message(
                "Semantik arama kapalı. Arama & AI Ayarları panelinden açabilirsiniz."
            )
            return

        def run_backfill():
            from core.index_maintenance import backfill_missing_semantic_tags
            return backfill_missing_semantic_tags(self.settings)

        def backfill_done(result) -> None:
            from core.search_cache import SEARCH_RESPONSE_CACHE
            SEARCH_RESPONSE_CACHE.clear()
            self.status_bar.set_message(
                f"Semantik etiket: {int(result.get('processed', 0)):,} tamamlandı · "
                f"{int(result.get('remaining', 0)):,} bekliyor"
            )

        self._run_background_task(
            run_backfill,
            working_message="Semantik etiketler arka planda tamamlanıyor…",
            success_message="",
            on_done=backfill_done,
        )

    def _on_index_finished(self, stats: dict) -> None:
        from core.index_v3.ui_bridge import index_lane_for_mode

        lane = index_lane_for_mode(
            str(stats.get("index_mode") or self._current_index_mode or "")
        )
        self._index_workers.pop(lane, None)
        still = self._alive_index_workers()
        self._sync_post_ga_lane_buttons()
        pending = self._pending_index_restart
        if still:
            self._index_worker = next(iter(still.values()))
            self._index_active = True
            self.status_bar.set_message(
                f"Index lane bitti ({lane}); diğer lane devam ediyor"
            )
            if pending:
                self._schedule_pending_index_watchdog()
            return

        self._index_active = False
        self._index_paused = False
        self._index_percent = 0
        self._index_progress_timer.stop()
        self.status_bar.clear_indexing_progress()
        self._pending_index_restart = None
        if pending:
            # Kullanıcı Durdu iken tekrar Başlat'a basmıştı — şimdi başlat
            self._index_worker = None
            self.status_bar.set_message("Index yeniden başlatılıyor…")
            QTimer.singleShot(
                0,
                lambda p=pending: self._run_index(
                    scan_mode=p["scan_mode"],
                    source_id=p.get("source_id") or 0,
                    run_due_only=bool(p.get("run_due_only")),
                    quiet=bool(p.get("quiet")),
                    index_mode=p.get("index_mode") or "complete",
                ),
            )
            return
        if stats.get("stopped"):
            self.progress_panel.set_indexing(False, terminal="stopped")
            msg = (
                f"Index durdu — işlenen: {stats.get('processed', 0)} "
                f"(mod: {stats.get('index_mode', self._current_index_mode)})"
            )
        else:
            self.progress_panel.set_indexing(False, terminal="completed")
            msg = f"Index — işlenen: {stats.get('processed', 0)}"
        from core.search_cache import SEARCH_RESPONSE_CACHE

        SEARCH_RESPONSE_CACHE.clear()
        self.source_sidebar.update_scan_stats(stats)
        if stats.get("speed_report_path"):
            msg = f"{msg} | Hız raporu: {stats['speed_report_path']}"
        if stats.get("ai_fallback_warning"):
            QMessageBox.warning(self, "AI", str(stats["ai_fallback_warning"]))
            msg = f"{msg} | AI: hash+doku fallback"
        self.status_bar.set_message(msg)
        if self._cache_reconciliation_worker:
            self._cache_reconciliation_worker.resume()

        self._refresh_status()
        if self._rerun_after_index:
            self._rerun_after_index = False
            QTimer.singleShot(0, self._run_current_search)

        self._load_customers()
        self._refresh_folder_watch_async()
        QTimer.singleShot(200, self._flush_pending_watch_scans)

    def _reload_source_table(self) -> None:
        """Kaynak satırını status worker beklemeden tabloya yaz."""
        try:
            sources = self._source_manager.list_sources(active_only=False)
        except Exception:
            return
        self.sources_panel.set_sources(self._merge_source_live(sources))
        self.sources_panel.set_checked_source_ids(
            list(self.settings.selected_source_ids or [])
        )

    def _on_status_updated(self, status: dict) -> None:
        sender = self.sender()
        # finished_ok => drop Python refs by identity only (no isRunning on sender).
        # deleteLater may already have run; never touch C++ without qobject_is_alive.
        from core.qthread_lifecycle import qobject_is_alive

        if sender is self._status_worker:
            self._status_worker = None
        elif sender is self._lane_status_worker:
            self._lane_status_worker = None
        gen = None
        if qobject_is_alive(sender):
            try:
                gen = getattr(sender, "_status_gen", None)
            except RuntimeError:
                gen = None
        if gen is not None and int(gen) != int(self._status_generation):
            if self._status_refresh_queued and not self._index_active:
                self._status_refresh_queued = False
                QTimer.singleShot(0, self._refresh_status)
            return

        self._status_loaded = True
        self._last_status = dict(status)

        searchable = int(status.get("total", 0) or 0)
        self._searchable_count = searchable

        # Kaynak listesi indeks devam ederken de görünür kalmalı. Bu yalnızca
        # hafif status worker'dan gelen SQLite metadata'sını ekrana taşır.
        if status.get("sources"):
            self.sources_panel.set_sources(
                self._merge_source_live(status.get("sources", []))
            )
            # Boş liste de uygula: [] = tüm arşiv; checkbox ile settings senkron kalsın.
            # (Eski bug: first_build tümünü işaretli gösterirdi, settings [] kalırdı → orphan sızardı.)
            self.sources_panel.set_checked_source_ids(
                list(self.settings.selected_source_ids or [])
            )

        if self._index_active:
            self.progress_panel.update_status(status)
            if self._is_simple_mode():
                self._update_user_status_bar()
            return

        status["cache_dir"] = self.settings.cache_dir

        if hasattr(self, "format_status_panel"):
            fsp = self.format_status_panel
            # Avoid rebuilding format panel on every StatusWorker tick (UI freeze).
            if getattr(fsp, "_db_path", None) != self.settings.db_path:
                fsp.set_db_path(self.settings.db_path)

        if hasattr(self, "category_tree_panel"):
            # set_db_path is idempotent for same path; still avoid redundant calls.
            ctp = self.category_tree_panel
            if getattr(ctp, "_db_path", None) != self.settings.db_path or (
                hasattr(ctp, "tree") and ctp.tree.topLevelItemCount() == 0
            ):
                ctp.set_db_path(self.settings.db_path)

        self.progress_panel.update_status(status)

        self.source_sidebar.update_cache_info(status)
        self.source_sidebar.update_missing_count(int(status.get("missing_files", 0)))

        total = status.get("total", 0)

        indexed = status.get("indexed", 0)

        pending = status.get("pending", 0)

        if self._is_simple_mode():
            self._update_user_status_bar()
        else:
            self.lbl_header_status.setText(
                (
                    f"Index: {indexed:,}/{total:,} | Bekleyen: {pending:,} | "
                    + (
                        "Hash-only açık: kalite motoru devre dışı"
                        if self.settings.fast_hash_only
                        else "Patch+doku açık"
                    )
                )
            )
            if self.settings.ai_embedding_enabled:
                ai_text = (
                    "AI tam açık (DINO+CLIP index+arama)"
                    if self.settings.index_uses_ai()
                    else "AI arama açık, index hızlı mod"
                )
            else:
                ai_text = "AI kapalı: klasik motor"
            faiss_hint = (
                " | FAISS aramada yüklenir"
                if not status.get("faiss_status_checked", True)
                else ""
            )
            self.status_bar.set_advanced_index_hint(
                f"{ai_text} | DB DINO {status.get('db_dino_embeddings', 0):,} "
                f"| DB CLIP {status.get('db_clip_embeddings', 0):,}{faiss_hint}"
            )

    def _cache_clean(self) -> None:

        reply = QMessageBox.question(
            self,
            "Cache Temizle",
            "Thumbnail ve embedding önbelleği silinsin mi?\n(Index kayıtları korunur.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        cache = Path(self.settings.cache_dir)

        for sub in ("thumbnails", "embeddings"):
            p = cache / sub

            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

            p.mkdir(parents=True, exist_ok=True)

        self._refresh_status()

        self.status_bar.set_message("Cache temizlendi")

    def _cache_fill_missing(self) -> None:

        self._run_index(scan_mode="quick")

    def _cleanup_workers(self) -> None:

        self._cancel_search_worker()

        from core.qthread_lifecycle import qobject_is_alive, qthread_is_running

        for w in (
            self._index_worker,
            *list((self._index_workers or {}).values()),
            *list((self._source_fast_workers or {}).values()),
            *list((self._source_count_workers or {}).values()),
            self._quick_index_worker,
            self._status_worker,
            self._lane_status_worker,
            self._cache_reconciliation_worker,
        ):
            if not qthread_is_running(w):
                continue
            if not qobject_is_alive(w):
                continue
            try:
                w.request_stop()
                w.wait_until_finished()
            except RuntimeError:
                pass

        # BackgroundTask.function is not mid-run cooperative; never clear/destroy
        # while the QThread is still running (avoids native:
        # "QThread: Destroyed while thread 'BackgroundTask' is still running").
        for task in list(self._background_tasks):
            try:
                if not qobject_is_alive(task):
                    self._background_tasks.discard(task)
                    continue
                task.request_stop()
            except RuntimeError:
                self._background_tasks.discard(task)
        for task in list(self._background_tasks):
            try:
                if not qobject_is_alive(task):
                    self._background_tasks.discard(task)
                    continue
                if qthread_is_running(task):
                    # Wait for real finish — 500ms was too short for startup probe /
                    # DB mutations and left the thread running into QObject teardown.
                    if not task.wait_until_finished(30_000):
                        logger.warning(
                            "BackgroundTask still running after 30s; detaching parent"
                        )
                        try:
                            task.setParent(None)
                            task.finished.connect(task.deleteLater)
                        except RuntimeError:
                            pass
                        continue
                self._background_tasks.discard(task)
            except RuntimeError:
                self._background_tasks.discard(task)
        self._background_tasks.clear()

    def closeEvent(self, event: QCloseEvent) -> None:

        self._shutting_down = True

        self._save_ui_state()

        self._cleanup_workers()

        # FaceAutoIndexer is a daemon Python thread, but it may still be inside
        # OpenCV/SQLite when Qt tears down the window.  Stop it explicitly before
        # QApplication exits so the face subsystem cannot outlive the UI.
        try:
            face_indexer = getattr(self, "_face_auto_indexer", None)
            if face_indexer is not None:
                face_indexer.stop()
        except Exception:
            logger.debug("Yüz arka plan indeksi kapanışta durdurulamadı", exc_info=True)

        try:
            ai_scanner = getattr(self, "_archive_intelligence", None)
            if ai_scanner is not None:
                ai_scanner.stop()
        except Exception:
            logger.debug("Archive Intelligence kapanışta durdurulamadı", exc_info=True)

        try:
            from core.production.health_monitor import stop_health_monitor

            stop_health_monitor()
        except Exception:
            pass

        if self._ui_perf:
            self._ui_perf.stop()

        super().closeEvent(event)

    def _open_folder(self, file_id: int) -> None:

        from core.db import Database

        rec = Database(self.settings.db_path).get_file_by_id(file_id)

        if rec:
            # Klasör: dosyayı açma — yalnızca üst dizini Explorer'da aç.
            self._open_in_explorer(str(Path(rec["path"]).parent))

    def _open_file(self, file_id: int) -> None:

        from core.db import Database

        rec = Database(self.settings.db_path).get_file_by_id(file_id)

        if rec:
            path = rec["path"]
            if _opens_with_default_app(path):
                self._open_with_default_app(path)
            else:
                # EPS/PDF/AI vb. — mevcut güvenli davranış: Explorer'da seç.
                self._open_in_explorer(path)

    @staticmethod
    def _open_with_default_app(path: str) -> None:

        path = os.path.normpath(path)

        if not os.path.isfile(path):
            QMessageBox.warning(None, "Uyarı", f"Yol bulunamadı:\n{path}")

            return

        try:
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])
        except OSError as exc:
            QMessageBox.warning(None, "Uyarı", f"Dosya açılamadı:\n{path}\n{exc}")

    @staticmethod
    def _open_in_explorer(path: str) -> None:

        path = os.path.normpath(path)

        if not os.path.exists(path):
            QMessageBox.warning(None, "Uyarı", f"Yol bulunamadı:\n{path}")

            return

        if sys.platform == "win32":
            if os.path.isfile(path):
                subprocess.Popen(["explorer", "/select,", path])

            else:
                os.startfile(path)

        else:
            subprocess.Popen(["xdg-open", path])

    def _run_background_task(
        self,
        function,
        *,
        working_message: str = "Arka planda tamamlanıyor…",
        success_message: str = "İşlem tamamlandı.",
        on_done=None,
    ) -> BackgroundTask:
        """Queue a short DB/index mutation and keep its QThread alive."""
        task = BackgroundTask(function, parent=self)
        self._background_tasks.add(task)
        if working_message:
            self.status_bar.set_message(working_message)

        def finish(value) -> None:
            self._background_tasks.discard(task)
            if on_done:
                on_done(value)
            if success_message:
                self.status_bar.set_message(success_message)
            task.deleteLater()

        def fail(message: str) -> None:
            self._background_tasks.discard(task)
            self.status_bar.set_message(f"Arka plan işlemi tamamlanamadı: {message}")
            task.deleteLater()

        task.finished_ok.connect(finish)
        task.error.connect(fail)
        task.start()
        return task

    def _preload_feedback_shortcuts(self) -> None:
        def load_shortcuts_task():
            from core.db import Database
            from core.family_shortcuts import (
                bootstrap_shortcuts_from_feedback,
                load_shortcuts,
            )

            db = Database(self.settings.db_path)
            bootstrap_shortcuts_from_feedback(db)
            return load_shortcuts(db)

        self._run_background_task(
            load_shortcuts_task,
            working_message="",
            success_message="",
            on_done=lambda rows: setattr(self, "_feedback_shortcuts", list(rows or [])),
        )

    def _refresh_folder_watch_async(self) -> None:
        self._run_background_task(
            self._folder_watch.discover_watch_paths,
            working_message="",
            success_message="",
            on_done=self._folder_watch.apply_watch_paths,
        )

    def _on_user_feedback(self, action: str, label: str) -> None:
        from core.category_predictions import category_predictions_from_result
        from core.db import Database
        from core.feedback_learn import (
            apply_wrong_match_correction,
            clear_learned_family,
            persist_learned_family,
        )
        from core.pattern_groups import (
            RELATION_EXACT,
            RELATION_SIMILAR,
            PatternGroupManager,
        )
        from core.textile_terms import resolve_user_family_label
        from core.user_feedback import UserFeedbackStore
        from ui.feedback_family_dialog import FeedbackFamilyDialog

        result = self.inspector_panel._result
        if not result:
            return
        query_path = (
            self.txt_search.text().strip()
            or self._last_query_path
            or self.query_panel.image_path()
            or ""
        )
        db = Database(self.settings.db_path)
        store = UserFeedbackStore(db)
        if action == "clear_family_labels":

            def clear_labels():
                UserFeedbackStore(Database(self.settings.db_path)).record(
                    query_path,
                    result.file_id,
                    action,
                    label="",
                )
                return clear_learned_family(self.settings, result.file_id)

            self._run_background_task(
                clear_labels,
                working_message="Etiketler temizleniyor; arka planda tamamlanıyor…",
                success_message="Aile etiketi temizlendi.",
            )
            return

        if action in ("wrong", "demote"):
            from core.family_shortcuts import remember_typed_category

            query_family = ""
            if self._search_response and self._search_response.meta:
                query_family = str(
                    self._search_response.meta.get("query_pattern_family", "")
                    or self._search_response.meta.get("query_effective_family", "")
                )
            dlg = FeedbackFamilyDialog(
                self,
                filename=result.filename,
                query_family=query_family,
                learned_shortcuts=list(self._feedback_shortcuts),
                db_path=self.settings.db_path,
            )
            if dlg.exec() != dlg.DialogCode.Accepted:
                self._run_background_task(
                    lambda: UserFeedbackStore(Database(self.settings.db_path)).record(
                        query_path,
                        result.file_id,
                        action,
                        label=label,
                    ),
                    working_message="Geri bildirim kaydediliyor…",
                    success_message="Yanlış olarak işaretlendi (grup seçilmedi).",
                )
                return

            family = dlg.selected_family()
            tag = dlg.custom_tag()
            resolved = resolve_user_family_label(tag) if tag and not family else {}
            if not family and resolved.get("pattern_family"):
                family = resolved["pattern_family"]
            if tag and not resolved.get("tag"):
                resolved["tag"] = tag
            subtype = resolved.get("pattern_subtype", "") or dlg.pattern_subtype()
            correction_path = dlg.category_path_value()
            if not correction_path and tag:
                correction_path = str(
                    resolve_user_family_label(tag).get("category_path", "") or ""
                )
            scope = "exact" if dlg.propagate_exact() else "single"
            if (
                correction_path
                and str(getattr(self.settings, "user_role", "user")).lower() != "admin"
            ):
                from core.category_feedback import submit_category_correction
                from core.category_predictions import category_predictions_from_result

                old_predictions = category_predictions_from_result(result)
                self._run_background_task(
                    lambda: submit_category_correction(
                        self.settings,
                        file_id=result.file_id,
                        category_path=correction_path,
                        old_category=str(
                            result.debug.get("manual_category_path", "")
                            or result.pattern_family
                        ),
                        old_predictions=old_predictions,
                        propagation_scope=scope,
                    ),
                    working_message="Düzeltme kaydediliyor; arka planda tamamlanıyor…",
                    success_message="Düzeltme inceleme havuzuna gönderildi.",
                )
                return

            similarity_tier = dlg.similarity_tier()
            tier_label_text = dlg.tier_label()
            cluster_group = dlg.cluster_group()
            selected_category_path = dlg.category_path_value()
            parent_category = dlg.parent_category()
            child_category = dlg.child_category()
            propagate_exact = dlg.propagate_exact()
            resolved_label = dlg.resolved_label()
            shortcut = dlg.shortcut_entry() if dlg.should_remember_shortcut() else None
            old_predictions = (
                category_predictions_from_result(result) if correction_path else []
            )
            old_category = str(
                result.debug.get("manual_category_path", "") or result.pattern_family
            )

            def save_wrong_correction():
                stats = apply_wrong_match_correction(
                    self.settings,
                    query_path=query_path,
                    result_file_id=result.file_id,
                    pattern_family=family,
                    tag=resolved.get("tag", tag) if tag else "",
                    reject_query_family=(
                        query_family if query_family not in ("", "unknown") else ""
                    ),
                    animal_print_type=resolved.get("animal_print_type", ""),
                    pattern_subtype=subtype,
                    similarity_tier=similarity_tier,
                    tier_label_text=tier_label_text,
                    cluster_group=cluster_group,
                    category_path=selected_category_path,
                    parent_category=parent_category,
                    child_category=child_category,
                    propagate_exact=propagate_exact,
                )
                if correction_path:
                    from core.category_feedback import submit_category_correction

                    submit_category_correction(
                        self.settings,
                        file_id=result.file_id,
                        category_path=correction_path,
                        old_category=old_category,
                        old_predictions=old_predictions,
                        propagation_scope=scope,
                        apply_category=False,
                    )
                if shortcut:
                    remember_typed_category(
                        Database(self.settings.db_path),
                        tag=shortcut.get("tag", ""),
                        pattern_family=shortcut.get("pattern_family", family),
                        pattern_subtype=shortcut.get("pattern_subtype", subtype),
                        tier_id=shortcut.get("tier_id", ""),
                        tier_label=shortcut.get("label", ""),
                        cluster_group=shortcut.get("cluster_group", ""),
                    )
                # Kullanıcının marka seçimi/yazım düzeltmesi kalıcı marka hafızasına yazılır.
                selected_path = selected_category_path or correction_path
                if selected_path.startswith("Marka/"):
                    canonical = selected_path.split("/", 1)[1].strip()
                    typed_brand = dlg.brand_query_text()
                    if typed_brand and canonical:
                        from core.brand_aliases import register_brand_alias
                        register_brand_alias(self.settings.db_path, typed_brand, canonical)

                from core.search_cache import SEARCH_RESPONSE_CACHE

                SEARCH_RESPONSE_CACHE.clear()
                return stats

            def wrong_saved(stats):
                self._load_inspector_learned_tags(result.file_id)
                if shortcut:
                    self._preload_feedback_shortcuts()

                # "Bu çiçek" demek mevcut Leopard aramasını Çiçek aramasına
                # çevirmek değildir. Seçilen sonuç mevcut sorguda yanlıştır;
                # bu yüzden onu ve öğretilmiş exact kopyalarını mevcut listeden
                # anında çıkarıyoruz. Kullanıcı aynı Leopard sorgusunda kalır.
                response = self._search_response
                if response is not None:
                    wrong_now = UserFeedbackStore(
                        Database(self.settings.db_path)
                    ).wrong_result_ids(query_path)
                    if wrong_now:
                        response.all_results = [
                            r for r in (response.all_results or [])
                            if int(r.file_id) not in wrong_now
                        ]
                        response.results = [
                            r for r in (response.results or [])
                            if int(r.file_id) not in wrong_now
                        ]
                        response.below_threshold_preview = [
                            r for r in (response.below_threshold_preview or [])
                            if int(r.file_id) not in wrong_now
                        ]
                        if response.stats is not None:
                            response.stats.above_threshold = len(response.results)
                        if self.inspector_panel._result and int(
                            self.inspector_panel._result.file_id
                        ) in wrong_now:
                            self.inspector_panel.set_result(None)
                        self._refresh_results_display(
                            live_stage="Yanlış eşleşme çıkarıldı",
                            in_place=False,
                        )

                extra = int((stats or {}).get("propagated_extra", 0) or 0)
                spread = f" (+{extra} exact kopya)" if extra else ""
                self.status_bar.set_message(
                    f"Yanlış eşleşme çıkarıldı — mevcut liste korunuyor{spread}"
                )

            if correction_path:
                self._apply_category_to_visible_result(result, correction_path)
            self._run_background_task(
                save_wrong_correction,
                working_message="Kategori kaydedildi; metadata arka planda tamamlanıyor…",
                success_message="",
                on_done=wrong_saved,
            )
            return

        if action == "label_family" and label:
            result.pattern_family = label
            result.debug["result_family"] = label
            result.debug["result_confidence"] = 1.0
            self.inspector_panel.set_result(result)

        def save_quick_feedback():
            local_db = Database(self.settings.db_path)
            local_store = UserFeedbackStore(local_db)
            local_store.record(query_path, result.file_id, action, label=label)
            if action == "label_family" and label:
                persist_learned_family(
                    self.settings,
                    result.file_id,
                    pattern_family=label,
                )
            elif action == "same_pattern":
                PatternGroupManager(local_db).add_or_strengthen(
                    result.file_id,
                    result.file_id,
                    RELATION_EXACT,
                    result.score,
                    pattern_family=result.pattern_family,
                    animal_print_type=result.animal_print_type,
                    color_family=result.color_family,
                )
            elif action == "similar_texture":
                qrec = local_db.get_file_by_path(query_path) if query_path else None
                PatternGroupManager(local_db).add_or_strengthen(
                    int((qrec or {}).get("id") or result.file_id),
                    result.file_id,
                    RELATION_SIMILAR,
                    result.score,
                )
            from core.search_cache import SEARCH_RESPONSE_CACHE

            SEARCH_RESPONSE_CACHE.clear()
            return True

        if action == "not_similar":
            message = "Benzer değil olarak işaretlendi."
        elif action == "label_family" and label:
            message = f"Aile öğrenildi: {label}"
        elif action == "label_not_family" and label:
            message = f"Aile reddedildi: {label}"
        else:
            message = f"Geri bildirim kaydedildi: {action}"
        self._run_background_task(
            save_quick_feedback,
            working_message="Kaydedildi; arka planda tamamlanıyor…",
            success_message=message,
        )

    def _edit_target_ids(self, current_id: int) -> list[int]:
        selected: list[int] = []
        panel = getattr(self, "results_panel", None)
        if panel is not None and hasattr(panel, "selected_preview_ids"):
            selected = list(panel.selected_preview_ids())
        if len(selected) >= 2:
            return selected
        if len(selected) == 1:
            return selected
        cid = int(current_id or 0)
        return [cid] if cid > 0 else []

    def _on_edit_result_metadata(self) -> None:
        from core.db import Database
        from core.teach_me import (
            apply_metadata_edit_to_files,
            build_common_edit_overlay,
            nonempty_overlay,
        )
        from core.user_feedback import (
            UserFeedbackStore,
            apply_metadata_overlay_to_result,
        )
        from ui.result_metadata_dialog import ResultMetadataDialog

        result = self.inspector_panel._result
        current_id = int(getattr(result, "file_id", 0) or 0) if result else 0
        target_ids = self._edit_target_ids(current_id)
        if not target_ids:
            return
        bulk = len(target_ids) > 1
        # Open panel first — heavy overlay/options load must not block UI.
        seed_overlay: dict = {}
        dlg_result = None
        if not bulk and result and int(result.file_id) == target_ids[0]:
            dbg = getattr(result, "debug", None) or {}
            seed_overlay = {
                "parent": "",
                "child": "",
                "pattern_family": getattr(result, "pattern_family", "") or "",
                "color_family": getattr(result, "color_family", "") or "",
                "brand": "",
                "tags": list(dbg.get("user_tags") or []) if isinstance(dbg, dict) else [],
            }
            path = str(
                (dbg.get("manual_category_path") if isinstance(dbg, dict) else "") or ""
            )
            if path:
                parent, _, child = path.partition("/")
                seed_overlay["parent"] = parent.strip()
                seed_overlay["child"] = child.strip()
                seed_overlay["category_path"] = path
            dlg_result = result
        dlg = ResultMetadataDialog(
            self,
            result=dlg_result,
            db_path=self.settings.db_path,
            overlay=seed_overlay,
        )
        n = len(target_ids)
        if bulk:
            dlg.setWindowTitle(f"Düzenle — {n} görsel")
            dlg.lbl_intro.setText(
                f"{n} görsel seçildi. Dolu alanlar hepsine uygulanır; "
                "boş bırakılan alanlar her dosyanın mevcut bilgisini korur. "
                "Örnek: zebra kaydını leopard olarak düzeltmek."
            )

        def load_overlay():
            local_db = Database(self.settings.db_path)
            store = UserFeedbackStore(local_db)
            if bulk:
                return build_common_edit_overlay(local_db, target_ids)
            return store.metadata_overlay_for_file(target_ids[0])

        def apply_overlay(ov) -> None:
            if not isinstance(ov, dict) or not ov:
                return
            # Merge without wiping user edits already typed in the open dialog.
            from ui.result_metadata_dialog import form_values_from_record

            current = dlg.values()
            merged = dict(ov)
            for key in ("parent", "child", "pattern_family", "color_family", "brand"):
                if str(current.get(key) or "").strip():
                    merged[key] = current[key]
            if current.get("tags"):
                merged["tags"] = current["tags"]
            vals = form_values_from_record(dlg_result, merged)
            if vals.get("parent"):
                dlg.cmb_parent.set_text(vals["parent"])
            if vals.get("child"):
                dlg.cmb_child.set_text(vals["child"])
            if vals.get("pattern_family"):
                dlg.cmb_family.set_text(vals["pattern_family"])
            if vals.get("color_family"):
                dlg.cmb_color.set_text(str(vals["color_family"]))
            if vals.get("brand"):
                dlg.cmb_brand.set_text(vals["brand"])
            if vals.get("tags") and not dlg._tags:
                dlg._tags = list(vals["tags"])
                dlg._refresh_tags()

        self._run_background_task(
            load_overlay,
            working_message="",
            success_message="",
            on_done=apply_overlay,
        )

        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        values = dlg.values()
        query_path = self.txt_search.text().strip()
        teach = dlg.teach_apply()
        applied = nonempty_overlay(values)
        if result and int(getattr(result, "file_id", 0) or 0) in set(target_ids):
            apply_metadata_overlay_to_result(result, applied)
            self.inspector_panel.set_result(result)
        panel = getattr(self, "results_panel", None)
        if panel is not None:
            wanted = set(target_ids)
            for rec in getattr(panel, "_full_results", []) or []:
                try:
                    fid = int(getattr(rec, "file_id", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if fid in wanted:
                    apply_metadata_overlay_to_result(rec, applied)

        def save_overlay():
            local_db = Database(self.settings.db_path)
            stats = apply_metadata_edit_to_files(
                local_db,
                self.settings.db_path,
                target_ids,
                values,
                query_path=query_path,
            )
            from core.search_cache import SEARCH_RESPONSE_CACHE

            SEARCH_RESPONSE_CACHE.clear()
            return stats

        self._run_background_task(
            save_overlay,
            working_message="Kaydedildi; sorgu öğretisine yazılıyor…",
            success_message=(
                f"{n} görsel güncellendi."
                if bulk
                else (
                    "Bu sorgu için öğretildi."
                    if teach or query_path
                    else "Bu görsel için metadata kaydedildi."
                )
            ),
        )

    def _on_ai_prediction_action(
        self, action: str, category_path: str, scope: str
    ) -> None:
        result = self.inspector_panel._result
        if not result:
            return
        if action == "wrong":
            self._on_user_feedback("wrong", "")
            return
        if action == "edit":
            self._on_edit_result_metadata()
            return
        from core.db import Database

        if action == "undo":

            def undo_correction():
                from core.search_memory import undo_last

                undone = undo_last(self.settings.db_path)
                from core.search_cache import SEARCH_RESPONSE_CACHE

                SEARCH_RESPONSE_CACHE.clear()
                return undone

            def undo_done(value):
                value = value or {}
                self.status_bar.set_message(
                    "Son hafıza düzeltmesi geri alındı."
                    if value.get("undone")
                    else "Geri alınacak hafıza düzeltmesi yok."
                )

            self._run_background_task(
                undo_correction,
                working_message="Düzeltme geri alınıyor…",
                success_message="",
                on_done=undo_done,
            )
            return
        if action != "correct" or not category_path:
            self.status_bar.set_message(
                "AI bu sonuç için güvenilir kategori üretemedi."
            )
            return

        def confirm_prediction():
            from core.user_feedback import UserFeedbackStore

            return UserFeedbackStore(Database(self.settings.db_path)).record(
                self.txt_search.text().strip() or self._last_query_path or "",
                result.file_id,
                "ai_category_correct",
                label=category_path,
            )

        self._run_background_task(
            confirm_prediction,
            working_message="AI tahmini doğrulandı; kayıt arka planda tamamlanıyor…",
            success_message=f"AI tahmini doğrulandı: {category_path}",
        )

    def _apply_category_to_visible_result(self, result, category_path: str) -> None:
        from core.category_tree import pattern_fields_for_path

        fields = pattern_fields_for_path(category_path)
        result.pattern_family = (
            fields.get("pattern_family", "") or result.pattern_family
        )
        result.animal_print_type = fields.get("animal_print_type", "")
        result.debug["pattern_subtype"] = fields.get("pattern_subtype", "")
        result.debug["result_family"] = result.pattern_family
        result.debug["result_confidence"] = 1.0
        result.debug["manual_category_path"] = category_path
        result.debug["category_source"] = "manual_user"
        self.inspector_panel.set_result(result)
        self.results_panel.update_scores_in_place(
            self._search_response.results if self._search_response else [result],
            stage="Tam Analiz",
        )

    def _on_learn_saved(self, parent: str, child: str, tag: str) -> None:
        if tag.strip():
            self._on_teach_tag(tag.strip())
        if parent or child:
            path = "/".join(p for p in (parent, child) if p)
            if path:
                self._on_teach_tag(path)
        self.status_bar.set_message("Desen etiketi kaydedildi")

    def _on_teach_tag(self, tag: str) -> None:
        from core.db import Database
        from core.family_shortcuts import remember_typed_category
        from core.indexer import Indexer
        from core.textile_terms import resolve_user_family_label
        from core.user_feedback import UserFeedbackStore

        result = self.inspector_panel._result
        if not result or not tag.strip():
            return
        query_path = (
            self.txt_search.text().strip()
            or self._last_query_path
            or self.query_panel.image_path()
            or ""
        )
        clean = tag.strip()
        self.inspector_panel.txt_teach_tag.clear()
        self.inspector_panel.set_learned_tags([clean])

        def save_tag():
            db = Database(self.settings.db_path)
            store = UserFeedbackStore(db)
            store.record_custom_tag(query_path, result.file_id, clean)
            resolved = resolve_user_family_label(clean)
            remember_typed_category(
                db,
                tag=clean,
                pattern_family=resolved.get("pattern_family", ""),
                pattern_subtype=resolved.get("pattern_subtype", ""),
            )
            rec = db.get_file_by_id(result.file_id)
            if rec:
                detailed = db.get_indexed_files_by_ids([result.file_id])
                tm = (detailed[0].get("texture_map") if detailed else {}) or {}
                Indexer(self.settings)._update_text_index(
                    result.file_id,
                    rec.get("filename", ""),
                    rec.get("path", ""),
                    rec.get("ocr_text", "") or "",
                    tm,
                )
            from core.search_cache import SEARCH_RESPONSE_CACHE

            SEARCH_RESPONSE_CACHE.clear()
            return store.custom_tags_for_file(result.file_id)

        self._run_background_task(
            save_tag,
            working_message="Etiket kaydedildi; arka planda tamamlanıyor…",
            success_message=f"Etiket öğrenildi: {clean}",
            on_done=self.inspector_panel.set_learned_tags,
        )

    def _load_inspector_learned_tags(self, file_id: int) -> None:
        def load_tags():
            from core.db import Database
            from core.user_feedback import UserFeedbackStore

            return UserFeedbackStore(
                Database(self.settings.db_path),
            ).custom_tags_for_file(file_id)

        selected_id = int(file_id)
        self._run_background_task(
            load_tags,
            working_message="",
            success_message="",
            on_done=lambda tags: (
                self.inspector_panel.set_learned_tags(tags)
                if self.inspector_panel._result
                and int(self.inspector_panel._result.file_id) == selected_id
                else None
            ),
        )

    def _search_similar(self, file_id: int) -> None:

        from core.db import Database

        rec = Database(self.settings.db_path).get_file_by_id(file_id)

        if rec and rec.get("path"):
            self._set_query_image(rec["path"], auto_search=True)


def run_app() -> None:

    app = QApplication.instance() or QApplication(sys.argv)

    app.setApplicationName(APP_NAME)

    apply_theme(app)

    window = MainWindow()

    # Face index is opt-in. Do not auto-start FaceAutoIndexer / FaceIndexScanner
    # on app launch; an explicit command must call start() after the user enables it.
    # Object index is also opt-in. Do not auto-start an object archive scan.
    try:
        from core.face_auto_indexer import FaceAutoIndexer

        window._face_auto_indexer = FaceAutoIndexer(window.settings)
    except Exception:
        logger.exception("Yüz zekâsı arka plan indeksi kurulamadı")

    # Archive Intelligence: delayed low-priority self-audit (sidecar only).
    try:
        from core.archive_intelligence import ArchiveIntelligenceScanner

        window._archive_intelligence = ArchiveIntelligenceScanner(window.settings)
        QTimer.singleShot(5000, window._archive_intelligence.start)
    except Exception:
        logger.exception("Archive Intelligence kurulamadı")

    window.show()

    sys.exit(app.exec())
