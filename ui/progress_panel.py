"""Index Dashboard v2 — operatör özeti + pipeline + canlı işlem."""

from __future__ import annotations

import time
from collections import deque

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

_PHASE_LABELS = {
    "Hızlı Index": "Hızlı Index",
    "Ağır Analiz": "Genel Index",
    "Genel Index": "Genel Index",
    "Genel AI": "Genel Index",
    "Eksik Tamamlama": "Eksikleri Tamamla",
}

_STATUS_TR = {
    "ready": "Hazır",
    "running": "Çalışıyor",
    "paused": "Duraklatıldı",
    "stopped": "Durdu",
    "completed": "Tamamlandı",
    "idle": "Hazır",
}

_PIPELINE_ROWS = (
    ("Önizleme", "preview_ready", "pct_preview"),
    ("Vektör", "embedding_ready", "pct_embedding"),
    ("Desen DNA", "pattern_dna_count", "pct_pattern_dna"),
    ("Anlamsal", "semantic_tag_count", "pct_semantic_tag"),
    ("OCR", "ocr_done", "pct_ocr"),
    ("Doku", "texture_done", "pct_texture"),
    ("Parça vektörü", "patch_embedding_ready", "pct_patch"),
)

# Üretim sağlığı ağırlıkları (toplam 100)
_HEALTH_WEIGHTS = {
    "preview_ready": 15,
    "embedding_ready": 25,
    "patch_embedding_ready": 15,
    "pattern_dna_count": 15,
    "texture_done": 10,
    "semantic_tag_count": 10,
    "ocr_done": 5,
    # Hash 5% — light_done proxy (temel hazır)
    "light_done": 5,
}

# AI Hazır = Genel AI hattı (PATCH/OCR hariç)
_AI_READY_KEYS = (
    "db_dino_embeddings",
    "db_clip_embeddings",
    "pattern_dna_count",
    "semantic_tag_count",
    "texture_done",
)


OWL_BAR_CLASSIC = "#22c55e"
OWL_BAR_DEFERRED = "#86efac"


def _health_color(pct: int) -> str:
    if pct >= 90:
        return "#22c55e"
    if pct >= 50:
        return "#eab308"
    return "#ef4444"


def _owl_bar_color(status: dict, pct: int) -> str:
    ov = status.get("owlv2") or {}
    tone = str(ov.get("bar_tone") or "")
    if tone == "classic":
        return OWL_BAR_CLASSIC
    if tone == "deferred":
        return OWL_BAR_DEFERRED
    return _health_color(pct)


def _health_emoji(pct: int) -> str:
    if pct >= 90:
        return "✓"
    if pct >= 50:
        return "⚠"
    return "✗"


class _PipelineRow(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        head = QHBoxLayout()
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight:600;")
        self.lbl_count = QLabel("0 / 0")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignRight)
        head.addWidget(self.lbl_title)
        head.addStretch()
        head.addWidget(self.lbl_count)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(12)
        self.lbl_meta = QLabel("")
        self.lbl_meta.setStyleSheet("color:#9ca3af;font-size:11px;")
        lay.addLayout(head)
        lay.addWidget(self.bar)
        lay.addWidget(self.lbl_meta)

    def set_values(
        self,
        ready: int,
        total: int,
        pct: int | None = None,
        *,
        session_delta: int | None = None,
        remaining: int | None = None,
        legacy: int = 0,
        extra: str = "",
        bar_color: str | None = None,
    ) -> None:
        total = max(0, int(total))
        ready = max(0, int(ready))
        if pct is None:
            pct = int(100 * ready / total) if total > 0 else 0
        pct = max(0, min(100, int(pct)))
        self.lbl_count.setText(f"{ready:,} / {total:,}")
        self.bar.setValue(pct)
        color = str(bar_color or "") or _health_color(pct)
        self.bar.setStyleSheet(
            f"QProgressBar {{ background:#1f2937; border:none; border-radius:4px; }}"
            f"QProgressBar::chunk {{ background:{color}; border-radius:4px; }}"
        )
        self.lbl_title.setStyleSheet(f"font-weight:600; color:{color};")
        rem = (
            max(0, total - ready)
            if remaining is None
            else max(0, int(remaining))
        )
        bits = []
        if session_delta is not None:
            bits.append(f"Bu oturum: {int(session_delta):+,}")
        bits.append(f"Kalan: {rem:,}")
        bits.append(f"%{pct}")
        if int(legacy or 0) > 0:
            bits.append(f"Gate bekleyen: {int(legacy):,}")
        if extra:
            bits.append(str(extra))
        self.lbl_meta.setText(" · ".join(bits))
        tip = (
            "Gate bekleyen = feature/path DB'de var ama physical preview hazır değil "
            "(çoğunlukla silinmiş cache). Bu kayıtlar Tamamlanan'a dahildir; "
            "yeniden DINO/CLIP üretmeyi gerektirmez."
            if int(legacy or 0) > 0
            else ""
        )
        self.lbl_meta.setToolTip(tip)


class ProgressPanel(QWidget):
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
    patch_start_clicked = Signal()
    ocr_start_clicked = Signal()
    patch_ocr_start_clicked = Signal()
    patch_stop_clicked = Signal()
    ocr_stop_clicked = Signal()
    post_ga_rerun_clicked = Signal()
    full_scan_clicked = Signal()
    full_scan_auto_changed = Signal(bool, int)  # enabled, interval_min
    semantic_backfill_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session_active = False
        self._jobs_total = 0
        self._jobs_done = 0
        self._session_started = 0.0
        self._task_started = 0.0
        self._speed_window_sec = 60.0
        self._speed_samples: deque[tuple[float, int]] = deque()
        self._db_status_snapshot: dict | None = None
        self._semantic_enabled = False
        self._session_delta = {"light": 0, "heavy": 0}
        self._session_light_base = 0
        self._session_heavy_base = 0
        self._last_speed = 0.0
        self._pipeline_rows: dict[str, _PipelineRow] = {}
        self._active_index_mode = ""
        self._lane_t0 = {"light": 0.0, "heavy": 0.0}
        self._lane_base = {"light": 0, "heavy": 0}
        self._lane_speed_ema: dict[str, float] = {"light": 0.0, "heavy": 0.0}
        self._lane_speed_samples: dict[str, deque[tuple[float, int]]] = {
            "light": deque(),
            "heavy": deque(),
        }
        self._lane_processing = {"light": 0, "heavy": 0}
        self._control_state = "ready"
        self._progress_frozen = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Kontroller ---
        action_group = QGroupBox("Index")
        action_layout = QVBoxLayout(action_group)
        self.btn_start = QPushButton("Hızlı + Genel AI")
        self.btn_start.setMinimumHeight(38)
        self.btn_start.setToolTip(
            "Hızlı index ve genel AI birlikte. Kaldığı yerden devam eder."
        )
        self.btn_start.clicked.connect(self.index_start_clicked.emit)
        action_layout.addWidget(self.btn_start)

        row1 = QGridLayout()
        self.btn_fast_only = QPushButton("Hızlı İndeks")
        self.btn_heavy_only = QPushButton("Genel Yapay Zekâ")
        self.btn_backfill = QPushButton("Eksikleri Tamamla")
        for btn in (self.btn_fast_only, self.btn_heavy_only, self.btn_backfill):
            btn.setMinimumHeight(32)
        self.btn_heavy_only.setToolTip(
            "Önizlemesi hazır dosyalarda genel yapay zekâ analizi. NAS taramaz."
        )
        self.btn_fast_only.setToolTip(
            "Küçük görsel, önizleme ve özet. Genel yapay zekâ işleyicisi başlatmaz."
        )
        self.btn_backfill.setToolTip(
            "Eksik kalan hızlı/genel işleri tamamlar. Kaldığı yerden devam eder."
        )
        self.btn_fast_only.clicked.connect(self.index_fast_archive_clicked.emit)
        self.btn_heavy_only.clicked.connect(self.index_night_complete_clicked.emit)
        self.btn_backfill.clicked.connect(self.index_backfill_clicked.emit)
        row1.addWidget(self.btn_fast_only, 0, 0)
        row1.addWidget(self.btn_heavy_only, 0, 1)
        row1.addWidget(self.btn_backfill, 1, 0, 1, 2)
        action_layout.addLayout(row1)

        ctrl = QHBoxLayout()
        self.btn_stop = QPushButton("Durdur")
        self.btn_pause = QPushButton("Duraklat")
        self.btn_resume = QPushButton("Devam Et")
        for btn in (self.btn_stop, self.btn_pause, self.btn_resume):
            btn.setMinimumHeight(30)
            ctrl.addWidget(btn)
        self.btn_stop.clicked.connect(self.index_stop_clicked.emit)
        self.btn_pause.clicked.connect(self.index_pause_clicked.emit)
        self.btn_resume.clicked.connect(self.index_resume_clicked.emit)
        action_layout.addLayout(ctrl)

        # Full scan — index drain'den bağımsız
        scan_box = QGroupBox("Sistem tarama (eksik / silinen)")
        scan_lay = QVBoxLayout(scan_box)
        self.btn_full_scan = QPushButton("Sistemi baştan tara")
        self.btn_full_scan.setMinimumHeight(34)
        self.btn_full_scan.setToolTip(
            "Eksik index gap'lerini bulur, silinen dosyaları missing yapar.\n"
            "Normal Hızlı/Genel AI kuyruğunu kilitlemez (arka plan)."
        )
        self.btn_full_scan.clicked.connect(self.full_scan_clicked.emit)
        scan_lay.addWidget(self.btn_full_scan)
        auto_row = QHBoxLayout()
        self.chk_full_scan_auto = QCheckBox("Belirli aralıkta otomatik tara")
        self.chk_full_scan_auto.setChecked(True)
        self.cmb_full_scan_interval = QComboBox()
        for label, mins in (
            ("15 dk", 15),
            ("30 dk", 30),
            ("1 saat", 60),
            ("3 saat", 180),
            ("6 saat", 360),
            ("12 saat", 720),
            ("24 saat", 1440),
        ):
            self.cmb_full_scan_interval.addItem(label, mins)
        self.cmb_full_scan_interval.setCurrentIndex(1)  # 30 dk
        auto_row.addWidget(self.chk_full_scan_auto, 1)
        auto_row.addWidget(self.cmb_full_scan_interval, 0)
        scan_lay.addLayout(auto_row)
        self.lbl_full_scan_status = QLabel("Tarama: —")
        self.lbl_full_scan_status.setStyleSheet("color:#9ca3af;font-size:11px;")
        self.lbl_full_scan_status.setWordWrap(True)
        scan_lay.addWidget(self.lbl_full_scan_status)

        def _emit_auto(_=None) -> None:
            self.full_scan_auto_changed.emit(
                bool(self.chk_full_scan_auto.isChecked()),
                int(self.cmb_full_scan_interval.currentData() or 30),
            )

        self.chk_full_scan_auto.stateChanged.connect(_emit_auto)
        self.cmb_full_scan_interval.currentIndexChanged.connect(_emit_auto)
        action_layout.addWidget(scan_box)
        layout.addWidget(action_group)

        # --- Durum özeti ---
        health_group = QGroupBox("Durum")
        health_layout = QVBoxLayout(health_group)
        self.lbl_active_mode = QLabel("Aktif Mod: —")
        self.lbl_active_mode.setStyleSheet(
            "color:#86efac;font-weight:700;font-size:14px;"
        )
        self.lbl_worker_status = QLabel("Durum: Hazır")
        self.lbl_worker_status.setStyleSheet(
            "color:#cbd5e1;font-size:13px;font-weight:600;"
        )
        self.lbl_total_files = QLabel("Toplam: —")
        self.lbl_total_files.setStyleSheet("color:#e5e7eb;font-size:13px;")
        self.lbl_current_file = QLabel("İşleniyor: —")
        self.lbl_current_file.setWordWrap(True)
        self.lbl_current_file.setStyleSheet("color:#93c5fd;font-size:12px;")
        # Gizli teknik alanlar
        self.lbl_health = QLabel("")
        self.lbl_health.hide()
        self.lbl_health_checks = QLabel("")
        self.lbl_health_checks.hide()
        self.lbl_basic_ready = QLabel("")
        self.lbl_basic_ready.hide()
        self.lbl_search_ready = QLabel("")
        self.lbl_search_ready.hide()
        self.lbl_ai_ready = QLabel("")
        self.lbl_ai_ready.hide()
        self.lbl_bottleneck = QLabel("")
        self.lbl_bottleneck.hide()
        self.lbl_processing = QLabel("")
        self.lbl_processing.hide()
        self.lbl_queue = QLabel("")
        self.lbl_queue.hide()
        self.lbl_speed = QLabel("")
        self.lbl_speed.hide()
        self.lbl_eta = QLabel("")
        self.lbl_eta.hide()
        self.bar_full_ready = QProgressBar()
        self.bar_full_ready.hide()
        self.lbl_active_phase = QLabel("")
        self.lbl_active_phase.hide()
        self.lbl_flow = QLabel("")
        self.lbl_flow.hide()
        self.lbl_current_task = QLabel("")
        self.lbl_current_task.hide()
        self.lbl_next_stage = QLabel("")
        self.lbl_next_stage.hide()
        self.lbl_worker = QLabel("")
        self.lbl_worker.hide()
        self.lbl_task_ms = QLabel("")
        self.lbl_task_ms.hide()
        self.progress = QProgressBar()
        self.progress.hide()
        self.lbl_session_delta = QLabel("")
        self.lbl_session_delta.hide()
        for w in (
            self.lbl_active_mode,
            self.lbl_worker_status,
            self.lbl_total_files,
            self.lbl_current_file,
        ):
            health_layout.addWidget(w)
        layout.addWidget(health_group)

        # --- Hızlı Index / Genel AI (yalnızca DB toplamı) ---
        light_group = QGroupBox("HIZLI INDEX")
        light_layout = QVBoxLayout(light_group)
        self.lbl_fast_summary = QLabel(
            "Toplam: —\nTamamlanan: —\nKuyruk: —\nİşleniyor: —"
        )
        self.lbl_fast_summary.setStyleSheet("font-weight:600;font-size:13px;")
        self.lbl_fast_perf = QLabel("Hız: —\nTahmini: —")
        self.lbl_fast_perf.setStyleSheet("color:#93c5fd;font-size:12px;")
        self.bar_fast = QProgressBar()
        self.bar_fast.setRange(0, 100)
        self.bar_fast.setValue(0)
        self.bar_fast.setMinimumHeight(18)
        light_layout.addWidget(self.lbl_fast_summary)
        light_layout.addWidget(self.lbl_fast_perf)
        light_layout.addWidget(self.bar_fast)
        layout.addWidget(light_group)

        heavy_group = QGroupBox("GENEL AI")
        heavy_layout = QVBoxLayout(heavy_group)
        self.lbl_general_summary = QLabel(
            "Toplam: —\nTamamlanan: —\nKuyruk: —\nİşleniyor: —"
        )
        self.lbl_general_summary.setStyleSheet("font-weight:600;font-size:13px;")
        self.lbl_general_perf = QLabel("Hız: —\nTahmini: —")
        self.lbl_general_perf.setStyleSheet("color:#86efac;font-size:12px;")
        self.bar_general = QProgressBar()
        self.bar_general.setRange(0, 100)
        self.bar_general.setValue(0)
        self.bar_general.setMinimumHeight(18)
        heavy_layout.addWidget(self.lbl_general_summary)
        heavy_layout.addWidget(self.lbl_general_perf)
        heavy_layout.addWidget(self.bar_general)
        layout.addWidget(heavy_group)

        extra_group = QGroupBox("PATCH / OCR (Genel AI sonrası)")
        extra_layout = QVBoxLayout(extra_group)
        self.lbl_general_ai_status = QLabel("Genel AI: Bekliyor")
        self.lbl_patch_lane_status = QLabel("PATCH: Bekliyor")
        self.lbl_ocr_lane_status = QLabel("OCR: Bekliyor")
        extra_layout.addWidget(self.lbl_general_ai_status)
        extra_layout.addWidget(self.lbl_patch_lane_status)
        extra_layout.addWidget(self.lbl_ocr_lane_status)
        extra_btns = QGridLayout()
        self.btn_start_patch = QPushButton("PATCH'i Başlat")
        self.btn_start_ocr = QPushButton("OCR'ı Başlat")
        self.btn_stop_patch = QPushButton("PATCH'i Durdur")
        self.btn_stop_ocr = QPushButton("OCR'ı Durdur")
        self.btn_start_patch_ocr = QPushButton("PATCH + OCR'ı Başlat")
        self.btn_rerun_post_ga = QPushButton("Yeniden Çalıştır")
        self._patch_lane_running = False
        self._ocr_lane_running = False
        self._post_ga_base_en = False
        for b in (
            self.btn_start_patch,
            self.btn_start_ocr,
            self.btn_stop_patch,
            self.btn_stop_ocr,
            self.btn_start_patch_ocr,
            self.btn_rerun_post_ga,
        ):
            b.setEnabled(False)
            b.setMinimumHeight(28)
        self.btn_start_patch.clicked.connect(self.patch_start_clicked.emit)
        self.btn_start_ocr.clicked.connect(self.ocr_start_clicked.emit)
        self.btn_stop_patch.clicked.connect(self.patch_stop_clicked.emit)
        self.btn_stop_ocr.clicked.connect(self.ocr_stop_clicked.emit)
        self.btn_start_patch_ocr.clicked.connect(self.patch_ocr_start_clicked.emit)
        self.btn_rerun_post_ga.clicked.connect(self.post_ga_rerun_clicked.emit)
        extra_btns.addWidget(self.btn_start_patch, 0, 0)
        extra_btns.addWidget(self.btn_start_ocr, 0, 1)
        extra_btns.addWidget(self.btn_stop_patch, 1, 0)
        extra_btns.addWidget(self.btn_stop_ocr, 1, 1)
        extra_btns.addWidget(self.btn_start_patch_ocr, 2, 0, 1, 2)
        extra_btns.addWidget(self.btn_rerun_post_ga, 3, 0, 1, 2)
        extra_layout.addLayout(extra_btns)
        layout.addWidget(extra_group)

        session_group = QGroupBox("BU OTURUM — YENİ OLUŞTURULANLAR (havuz Δ)")
        session_layout = QVBoxLayout(session_group)
        self.lbl_session_block = QLabel("")
        self.lbl_session_block.setWordWrap(True)
        self.lbl_session_block.setStyleSheet("color:#a78bfa;font-size:11px;")
        session_layout.addWidget(self.lbl_session_block)
        self.lbl_retry_files = QLabel("")
        self.lbl_retry_files.setWordWrap(True)
        self.lbl_retry_files.setStyleSheet("color:#fbbf24;font-size:11px;")
        session_layout.addWidget(self.lbl_retry_files)
        self.lbl_failed_files = QLabel("")
        self.lbl_failed_files.setWordWrap(True)
        self.lbl_failed_files.setStyleSheet("color:#f87171;font-size:11px;")
        session_layout.addWidget(self.lbl_failed_files)
        layout.addWidget(session_group)
        # Havuz session_plus (current − oturum başı); event sayacı değil.

        # --- Kaynak (tek satır) ---
        src_group = QGroupBox("Kaynak")
        src_layout = QVBoxLayout(src_group)
        self.lbl_source_line = QLabel("0 kaynak · 0 görsel")
        self.lbl_source_scan = QLabel("")
        self.lbl_source_scan.hide()
        src_layout.addWidget(self.lbl_source_line)
        layout.addWidget(src_group)

        # --- AI İşlemleri (katlanabilir) ---
        self._ai_toggle = QToolButton()
        self._ai_toggle.setText("Toplam Veritabanı — Genel AI Tamamlananlar")
        self._ai_toggle.setCheckable(True)
        self._ai_toggle.setChecked(False)
        self._ai_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._ai_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._ai_toggle.toggled.connect(self._on_ai_toggled)
        layout.addWidget(self._ai_toggle)

        self._ai_group = QFrame()
        self._ai_group.setVisible(False)
        ai_layout = QVBoxLayout(self._ai_group)
        ai_layout.setContentsMargins(0, 0, 0, 0)
        # Dosya bazlı SSOT satırları — artifact havuz (event değil)
        self._ai_display_rows: list[tuple[str, str]] = [
            ("Thumbnail", "thumbnail_ready"),
            ("Preview", "preview_ready"),
            ("Hash", "hash_ready"),
            ("Metadata", "metadata_ready"),
            ("DINO", "db_dino_embeddings"),
            ("OpenCLIP", "db_clip_embeddings"),
            ("Texture", "texture_done"),
            ("Semantic", "semantic_tag_count"),
            ("Pattern DNA", "pattern_dna_count"),
            ("Object Index", "object_concept_ready"),
            ("OWLv2", "owlv2_ready"),
            ("Patch", "patch_embedding_ready"),
            ("OCR", "ocr_done"),
            ("AI Final", "ai_final_ready"),
        ]
        self._pipeline_rows = {}
        for title, key in self._ai_display_rows:
            row = _PipelineRow(title)
            self._pipeline_rows[key] = row
            ai_layout.addWidget(row)
        # Gizli eski anahtarlar (geriye uyum)
        for title, key, _ in _PIPELINE_ROWS:
            if key not in self._pipeline_rows:
                row = _PipelineRow(title)
                row.hide()
                self._pipeline_rows[key] = row
        layout.addWidget(self._ai_group)
        self._pipe_group = self._ai_group

        # --- Geliştirici Modu ---
        self._dev_toggle = QToolButton()
        self._dev_toggle.setText("Geliştirici Modu")
        self._dev_toggle.setCheckable(True)
        self._dev_toggle.setChecked(False)
        self._dev_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._dev_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._dev_toggle.toggled.connect(self._on_dev_toggled)
        layout.addWidget(self._dev_toggle)

        self.maint_group = QFrame()
        self.maint_group.setVisible(False)
        maint_layout = QVBoxLayout(self.maint_group)
        maint_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_dev_block = QLabel("—")
        self.lbl_dev_block.setWordWrap(True)
        self.lbl_dev_block.setStyleSheet("color:#9ca3af;font-size:11px;")
        maint_layout.addWidget(self.lbl_dev_block)
        self.lbl_session_debug = QLabel("")
        self.lbl_session_debug.setWordWrap(True)
        self.lbl_session_debug.setStyleSheet("color:#a78bfa;font-size:11px;")
        maint_layout.addWidget(self.lbl_session_debug)

        self.btn_purge_broken = QPushButton("Bozuk kayıtları temizle")
        self.btn_requeue_thumbs = QPushButton("Eksik küçük görselleri kuyruğa al")
        self.btn_queue_ai = QPushButton("Yapay zekâ vektörlerini kuyruğa al")
        self.btn_semantic_backfill = QPushButton("Anlamsal etiketleri tamamla")
        self.btn_semantic_backfill.setEnabled(False)
        for btn in (
            self.btn_purge_broken,
            self.btn_requeue_thumbs,
            self.btn_queue_ai,
            self.btn_semantic_backfill,
        ):
            btn.setMinimumHeight(28)
            maint_layout.addWidget(btn)
        self.btn_purge_broken.clicked.connect(self.index_purge_broken_clicked.emit)
        self.btn_requeue_thumbs.clicked.connect(self.index_requeue_thumbs_clicked.emit)
        self.btn_queue_ai.clicked.connect(self.index_queue_ai_clicked.emit)
        self.btn_semantic_backfill.clicked.connect(self.semantic_backfill_clicked.emit)
        layout.addWidget(self.maint_group)

        # Gizli alias'lar
        self.lbl_source_count = QLabel()
        self.lbl_searchable = QLabel()
        self.lbl_embedding_status = QLabel()
        self.lbl_ocr_status = QLabel()
        self.lbl_full_done_status = QLabel()
        self.lbl_light_block = QLabel()
        self.lbl_heavy_block = QLabel()
        self.lbl_db_block = QLabel()
        self.lbl_deep_violation = QLabel()
        self.lbl_faiss = QLabel()
        for w in (
            self.lbl_source_count,
            self.lbl_searchable,
            self.lbl_embedding_status,
            self.lbl_ocr_status,
            self.lbl_full_done_status,
            self.lbl_light_block,
            self.lbl_heavy_block,
            self.lbl_db_block,
            self.lbl_deep_violation,
            self.lbl_faiss,
        ):
            w.hide()

        self.lbl_stage = self.lbl_active_phase
        self.lbl_current = self.lbl_current_task
        self.lbl_total = self.lbl_searchable
        self.lbl_ready_block = self.lbl_db_block
        self.lbl_pending_block = QLabel()
        self.lbl_quarantine_block = QLabel()
        self.lbl_pct = QLabel()
        self.lbl_pct_ai = QLabel()
        self.lbl_fast_archive = self.btn_fast_only
        self.lbl_night_complete = self.btn_heavy_only
        self.mode_group = action_group
        layout.addStretch()
        self.set_control_state("ready")

    def _on_ai_toggled(self, checked: bool) -> None:
        self._ai_group.setVisible(checked)
        self._ai_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _on_dev_toggled(self, checked: bool) -> None:
        self.maint_group.setVisible(checked)
        self._dev_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def set_simple_mode(self, simple: bool) -> None:
        # Basit modda geliştirici kapalı kalsın; açılabilir
        if simple:
            self._dev_toggle.setChecked(False)
            self.maint_group.setVisible(False)

    def set_semantic_enabled(self, enabled: bool) -> None:
        self._semantic_enabled = bool(enabled)
        self.btn_semantic_backfill.setEnabled(self._semantic_enabled)

    def set_root_path(self, path: str) -> None:
        pass

    def begin_session(self, *, index_mode: str = "") -> None:
        self._session_active = True
        self._jobs_total = 0
        self._jobs_done = 0
        self._session_started = time.monotonic()
        self._task_started = time.monotonic()
        self._speed_samples.clear()
        self._session_delta = {"light": 0, "heavy": 0}
        snap = self._db_status_snapshot or {}
        # session_plus baseline = oturum başı artifact havuz sayıları
        pools = dict(snap.get("artifact_pools") or {})
        self._session_pool_base = {
            "thumbnail": int(pools.get("thumbnail", snap.get("thumbnail_ready", 0)) or 0),
            "preview": int(pools.get("preview", snap.get("preview_ready", 0)) or 0),
            "dino": int(pools.get("dino", snap.get("db_dino_embeddings", 0)) or 0),
            "clip": int(pools.get("clip", snap.get("db_clip_embeddings", 0)) or 0),
            "texture": int(pools.get("texture", snap.get("texture_done", 0)) or 0),
            "semantic": int(pools.get("semantic", snap.get("semantic_tag_count", 0)) or 0),
            "dna": int(pools.get("dna", snap.get("pattern_dna_count", 0)) or 0),
            "patch": int(pools.get("patch", snap.get("patch_embedding_ready", 0)) or 0),
            "ai_final": int(pools.get("ai_final", snap.get("ai_final_ready", 0)) or 0),
        }
        self._session_light_base = int(self._session_pool_base["preview"])
        self._session_heavy_base = int(self._session_pool_base["ai_final"])
        self._session_pipeline_base = {
            "thumbnail_ready": self._session_pool_base["thumbnail"],
            "preview_ready": self._session_pool_base["preview"],
            "db_dino_embeddings": self._session_pool_base["dino"],
            "db_clip_embeddings": self._session_pool_base["clip"],
            "patch_embedding_ready": self._session_pool_base["patch"],
            "semantic_tag_count": self._session_pool_base["semantic"],
            "pattern_dna_count": self._session_pool_base["dna"],
            "texture_done": self._session_pool_base["texture"],
            "ai_final_ready": self._session_pool_base["ai_final"],
        }
        self._session_counter_last = {}
        self._session_counter_offset = {}
        self._session_counter_values = {}
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_session_block.setText(self._format_session_summary({}))
        self.lbl_session_debug.setText(self._format_session_block({}))
        self._lane_t0 = {"light": time.monotonic(), "heavy": time.monotonic()}
        self._lane_base = {"light": 0, "heavy": 0}
        self._lane_speed_ema = {"light": 0.0, "heavy": 0.0}
        self._lane_speed_samples = {"light": deque(), "heavy": deque()}
        self._lane_processing = {"light": 0, "heavy": 0}
        self.lbl_fast_perf.setText("Hız: —\nTahmini: —")
        self.lbl_general_perf.setText("Hız: —\nTahmini: —")
        self.lbl_current_file.setText("İşleniyor: Kaldığı yerden devam ediyor…")
        if index_mode:
            self.set_active_index_mode(index_mode, status="running")
        else:
            self.set_control_state("running")
        if snap:
            self._paint_session_lanes(snap)

    def end_session(self) -> None:
        self._session_active = False
        if self._control_state not in ("running", "paused"):
            self.lbl_current_file.setText("İşleniyor: —")
        if self._db_status_snapshot:
            self._apply_db_status(self._db_status_snapshot)

    def set_active_index_mode(self, index_mode: str, *, status: str = "running") -> None:
        from core.index_session import mode_display_label

        self._active_index_mode = str(index_mode or "")
        label = mode_display_label(self._active_index_mode)
        self.lbl_active_mode.setText(f"Aktif Mod: {label}")
        self.set_control_state(status)

    def set_worker_status(self, status: str) -> None:
        # Geriye uyum: Running/Paused/Idle → control state
        key = {
            "Running": "running",
            "Paused": "paused",
            "Stopping": "running",
            "Idle": "ready",
            "Stopped": "stopped",
            "Completed": "completed",
        }.get(status, str(status or "ready").lower())
        self.set_control_state(key)

    def set_full_scan_options(self, *, enabled: bool, interval_min: int) -> None:
        self.chk_full_scan_auto.blockSignals(True)
        self.cmb_full_scan_interval.blockSignals(True)
        self.chk_full_scan_auto.setChecked(bool(enabled))
        idx = self.cmb_full_scan_interval.findData(int(interval_min or 30))
        if idx < 0:
            idx = 1
        self.cmb_full_scan_interval.setCurrentIndex(idx)
        self.chk_full_scan_auto.blockSignals(False)
        self.cmb_full_scan_interval.blockSignals(False)

    def set_full_scan_status(self, text: str) -> None:
        self.lbl_full_scan_status.setText(str(text or 'Tarama: —'))

    def set_control_state(self, state: str) -> None:
        """Hazır | Çalışıyor | Duraklatıldı | Durdu | Tamamlandı."""
        state = str(state or "ready").lower()
        if state == "idle":
            state = "ready"
        self._control_state = state
        label = _STATUS_TR.get(state, state)
        color = {
            "ready": "#94a3b8",
            "running": "#86efac",
            "paused": "#fbbf24",
            "stopped": "#f87171",
            "completed": "#60a5fa",
        }.get(state, "#cbd5e1")
        self.lbl_worker_status.setText(f"Durum: {label}")
        self.lbl_worker_status.setStyleSheet(
            f"color:{color};font-size:13px;font-weight:600;"
        )
        self._apply_button_state(state)

    def set_post_ga_lane_running(
        self, *, patch: bool | None = None, ocr: bool | None = None
    ) -> None:
        if patch is not None:
            self._patch_lane_running = bool(patch)
        if ocr is not None:
            self._ocr_lane_running = bool(ocr)
        self._apply_post_ga_button_state()

    def _apply_post_ga_button_state(self, base_en: bool | None = None) -> None:
        if not hasattr(self, "btn_start_patch"):
            return
        if base_en is not None:
            self._post_ga_base_en = bool(base_en)
        en = bool(getattr(self, "_post_ga_base_en", False))
        patch_run = bool(getattr(self, "_patch_lane_running", False))
        ocr_run = bool(getattr(self, "_ocr_lane_running", False))
        self.btn_start_patch.setEnabled(en and not patch_run)
        self.btn_start_ocr.setEnabled(en and not ocr_run)
        self.btn_stop_patch.setEnabled(patch_run)
        self.btn_stop_ocr.setEnabled(ocr_run)
        self.btn_start_patch_ocr.setEnabled(en and not (patch_run and ocr_run))
        self.btn_rerun_post_ga.setEnabled(en and not patch_run and not ocr_run)

    def _apply_button_state(self, state: str) -> None:
        """Butonlar görünür kalır; geçersiz olanlar pasif."""
        start_ok = state in ("ready", "stopped", "completed")
        stop_ok = state in ("running", "paused")
        pause_ok = state == "running"
        resume_ok = state == "paused"
        for btn in (
            self.btn_start,
            self.btn_fast_only,
            self.btn_heavy_only,
            self.btn_backfill,
        ):
            btn.setEnabled(start_ok)
        self.btn_stop.setEnabled(stop_ok)
        self.btn_pause.setEnabled(pause_ok)
        self.btn_resume.setEnabled(resume_ok)
        maint_ok = start_ok
        for btn in (
            self.btn_purge_broken,
            self.btn_requeue_thumbs,
            self.btn_queue_ai,
        ):
            btn.setEnabled(maint_ok)
        self.btn_semantic_backfill.setEnabled(maint_ok and self._semantic_enabled)
        # Disabled görünürlüğü
        for btn in (
            self.btn_start,
            self.btn_fast_only,
            self.btn_heavy_only,
            self.btn_backfill,
            self.btn_stop,
            self.btn_pause,
            self.btn_resume,
        ):
            if btn.isEnabled():
                btn.setStyleSheet("")
            else:
                btn.setStyleSheet("color:#6b7280;")

    @staticmethod
    def _balanced_lane(done: int, total: int) -> tuple[int, int, int]:
        """İşlenen + Kuyruk = Toplam (asla aşmaz)."""
        total = max(0, int(total))
        done = max(0, min(int(done), total if total > 0 else int(done)))
        if total <= 0:
            return done, 0, 0
        queue = max(0, total - done)
        return done, queue, total

    @staticmethod
    def _fmt_eta(seconds: int) -> str:
        if seconds <= 0:
            return "—"
        # Aşırı uzun ETA'yı gösterme (yanıltıcı 470s vb.)
        if seconds > 72 * 3600:
            days = seconds / 86400.0
            return f"~{days:.0f} gün+"
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}s {m}dk"
        if m:
            return f"{m}dk {s}sn"
        return f"{s}sn"

    @staticmethod
    def _fmt_speed(speed_per_min: float) -> str:
        """Düşük hızda 0 dosya/sn yazıp ETA şişirmeyi engelle."""
        spm = float(speed_per_min or 0.0)
        if spm <= 0:
            return "—"
        sps = spm / 60.0
        if sps >= 1.0:
            return f"{sps:.0f} dosya/sn"
        if spm >= 1.0:
            return f"{spm:.0f} dosya/dk"
        return f"{spm:.1f} dosya/dk"

    def _update_lane_card(
        self,
        *,
        summary: QLabel,
        perf: QLabel,
        bar: QProgressBar,
        done: int,
        total: int,
        speed_per_min: float,
        processing: int = 0,
        queue: int | None = None,
        failed: int = 0,
        title: str = "",
        eta_queue: int | None = None,
        executable_pending: int | None = None,
        claimed_jobs: int = 0,
        retry_pending: int = 0,
        failed_permanent: int = 0,
        pending_files: int | None = None,
    ) -> None:
        # SSOT: Tamamlanan=havuz READY, Kuyruk=kalan, İşleniyor=aktif claim
        total = max(0, int(total))
        done = max(0, min(int(done), total if total else int(done)))
        if queue is None:
            done, queue, total = self._balanced_lane(done, total)
        else:
            queue = max(0, int(queue))
        # ETA için hazır iş (Genel AI: yalnız light=done havuzu)
        eta_q = max(0, int(eta_queue if eta_queue is not None else queue))
        pct = int(100 * done / total) if total else 0
        fail_n = max(0, int(failed))
        fail_line = f"\nBaşarısız   : {fail_n:,}" if fail_n else ""
        active = max(0, int(processing or 0))
        if claimed_jobs and not processing:
            active = max(active, int(claimed_jobs))
        exec_n = max(0, int(executable_pending or 0))
        # GENEL AI: Tamamlanan=AI Final; ara DINO/CLIP bar'ı oynatmaz — aktif job ayrı satır.
        if "GENEL" in str(title or "").upper() or "AI" in str(title or "").upper():
            note = ""
            if done == 0 and (active > 0 or exec_n > 0):
                note = "\n(AI Final henüz yok — heavy çalışıyor/bekliyor)"
            # exec_n = aşama job (hash/dino/clip…); dosya sayısı ayrı verilirse göster.
            file_n = max(0, int(pending_files or 0)) if pending_files is not None else 0
            files_line = (
                f"Bekleyen dosya: {file_n:,}\n" if pending_files is not None else ""
            )
            jobs_label = "Bekleyen aşama" if pending_files is not None else "Bekleyen job"
            summary.setText(
                f"Toplam      : {total:,}\n"
                f"AI Final    : {done:,}\n"
                f"Kalan dosya : {queue:,}\n"
                f"İşleniyor   : {active:,}\n"
                f"{files_line}"
                f"{jobs_label}: {exec_n:,}\n"
                f"Yeniden işleme alınacak: {max(0, int(retry_pending)):,}\n"
                f"Hatalı dosya : {max(0, int(failed_permanent)):,}"
                f"{fail_line}{note}"
            )
        else:
            summary.setText(
                f"Toplam      : {total:,}\n"
                f"Tamamlanan  : {done:,}\n"
                f"Kalan dosya : {queue:,}\n"
                f"İşleniyor   : {active:,}\n"
                f"Yeniden işleme alınacak: {max(0, int(retry_pending)):,}\n"
                f"Hatalı dosya : {max(0, int(failed_permanent)):,}"
                f"{fail_line}"
            )
        spm = float(speed_per_min or 0.0)
        speed_txt = self._fmt_speed(spm)
        eta = "—"
        # Aktif iş veya kuyruk varken sonsuza — gösterme
        measuring = active > 0 or exec_n > 0 or eta_q > 0
        if spm >= 2.0 and eta_q > 0:
            secs = int(eta_q / spm * 60.0)
            eta = self._fmt_eta(secs)
            if speed_txt == "—":
                speed_txt = self._fmt_speed(spm)
        elif spm > 0 and eta_q > 0:
            if speed_txt == "—":
                speed_txt = self._fmt_speed(max(spm, 0.1))
            eta = "Ölçülüyor…"
        elif measuring and spm <= 0:
            speed_txt = "Ölçülüyor…"
            eta = "Bekleniyor…"
        elif measuring:
            eta = "Ölçülüyor…"
        perf.setText(f"Hız         : {speed_txt}\nTahmini     : {eta}")
        self._set_mode_bar(bar, done, total_files=total, label="")
        bar.setFormat(f"{pct}%")
        _ = title

    def _apply_simple_summaries(
        self,
        status: dict,
        *,
        live_light: int | None = None,
        live_heavy: int | None = None,
        session_note: str = "",
        light_total_override: int | None = None,
        heavy_total_override: int | None = None,
        light_speed_pm: float = 0.0,
        heavy_speed_pm: float = 0.0,
        light_processing: int | None = None,
        heavy_processing: int | None = None,
    ) -> None:
        """Ana kartlar YALNIZ SSOT DB — oturum/RAM sayaçları yok."""
        _ = (
            session_note,
            light_total_override,
            heavy_total_override,
            live_light,
            live_heavy,
        )
        total = int(
            status.get("ssot_total")
            or status.get("total")
            or status.get("total_searchable")
            or status.get("archive_total")
            or 0
        )
        light_done = int(status.get("light_done", 0) or 0)
        # General AI İşlenen = AI Final (zorunlu artifact tamam); Patch residual DEĞİL
        if "ai_final_ready" in status:
            ai_final = int(status.get("ai_final_ready") or 0)
        else:
            ai_final = int(status.get("heavy_done", 0) or 0)
        ai_final = max(0, min(ai_final, total))
        light_failed = int(status.get("light_failed", 0) or 0)
        heavy_failed = int(status.get("heavy_failed", 0) or 0)
        light_queue = int(
            status.get("light_queue_display")
            if status.get("light_queue_display") is not None
            else (
                int(status.get("light_pending", 0) or 0)
                + int(status.get("light_processing", 0) or 0)
            )
        )
        # Executable heavy kuyruk (WAITING_PREVIEW hariç)
        heavy_ready = int(status.get("heavy_ready_pending", 0) or 0)
        db_heavy_proc = int(status.get("heavy_processing", 0) or 0)
        if status.get("heavy_queue_display") is not None:
            heavy_queue = int(status.get("heavy_queue_display") or 0)
        else:
            heavy_queue = max(0, heavy_ready + db_heavy_proc)
        # İşleniyor = status light/heavy claimed (event kwargs yedek).
        light_proc = max(
            0,
            int(
                status.get("light_processing", 0)
                or status.get("claimed_light_jobs", 0)
                or (light_processing or 0)
                or 0
            ),
        )
        heavy_proc = max(
            0,
            int(
                status.get("heavy_processing", 0)
                or status.get("claimed_heavy_jobs", 0)
                or db_heavy_proc
                or (heavy_processing or 0)
                or 0
            ),
        )

        # Genel AI ETA: yalnız hazır havuz (WAITING_PREVIEW hariç)
        heavy_ready_eta = heavy_ready + db_heavy_proc
        waiting_preview = int(
            status.get("waiting_preview")
            or status.get("needs_medium_preview_count")
            or 0
        )
        self._update_lane_card(
            summary=self.lbl_fast_summary,
            perf=self.lbl_fast_perf,
            bar=self.bar_fast,
            done=light_done,
            total=total,
            speed_per_min=light_speed_pm,
            processing=light_proc,
            queue=light_queue,
            failed=light_failed,
            title="HIZLI INDEX",
            executable_pending=int(status.get("pending_light_jobs", 0) or 0),
            claimed_jobs=light_proc,
            retry_pending=int(status.get("retry_light_jobs", 0) or 0),
            failed_permanent=int(status.get("failed_permanent_jobs", 0) or 0),
        )
        self._update_lane_card(
            summary=self.lbl_general_summary,
            perf=self.lbl_general_perf,
            bar=self.bar_general,
            done=ai_final,
            total=total,
            speed_per_min=heavy_speed_pm,
            processing=heavy_proc,
            queue=heavy_queue,
            failed=heavy_failed,
            title="GENEL AI",
            eta_queue=heavy_ready_eta,
            executable_pending=int(status.get("pending_heavy_jobs", 0) or 0),
            claimed_jobs=heavy_proc,
            retry_pending=int(status.get("retry_heavy_jobs", 0) or 0),
            failed_permanent=int(status.get("failed_permanent_jobs", 0) or 0),
            pending_files=(
                int(status["pending_heavy_files"])
                if status.get("pending_heavy_files") is not None
                else None
            ),
        )
        # PATCH/OCR grubundaki Genel AI satırı: AI Final %0 iken heavy aktifse yanıltma.
        if hasattr(self, "lbl_general_ai_status"):
            pend_h = int(status.get("pending_heavy_jobs", 0) or 0)
            if heavy_proc > 0:
                self.lbl_general_ai_status.setText(
                    f"Genel AI: Çalışıyor ({heavy_proc} job)"
                )
            elif pend_h > 0 and ai_final < total:
                self.lbl_general_ai_status.setText(
                    f"Genel AI: Kuyrukta {pend_h:,} job · AI Final {ai_final:,}/{total:,}"
                )
            elif total > 0 and ai_final >= total:
                self.lbl_general_ai_status.setText("Genel AI: Tamamlandı")
            elif not str(status.get("general_ai_status_label") or "").strip():
                self.lbl_general_ai_status.setText("Genel AI: Bekliyor")

        sources = int(status.get("source_count", 0) or 0)
        missing = int(status.get("missing_files", 0) or 0)
        scope_ids = status.get("status_scope_source_ids") or []
        orphan_files = int(status.get("orphan_file_total", 0) or 0)
        scope_mode = str(status.get("scope_mode") or "")
        # Orphan arşiv/index toplamına GİRMEZ — ayrı diagnostic
        if total == 0 and sources == 0 and scope_mode != "selected":
            src_bits = [
                "Tüm arşiv · 0 kayıtlı",
                "Index: 0",
                f"Yetim kayıtlar: {orphan_files:,}",
            ]
            self.lbl_source_line.setText(" · ".join(src_bits))
            self.lbl_total_files.setText("Toplam: 0")
            return
        if scope_mode == "selected" and scope_ids:
            src_bits = [
                f"{len(scope_ids):,} kaynak seçili",
                f"Index: {total:,}",
                f"Hızlı: {light_done:,}",
                f"Yapay zekâ son durumu: {ai_final:,}",
            ]
        else:
            label = f"{sources:,} kaynak" if sources else "0 kayıtlı kaynak"
            if scope_mode == "all_archive":
                label = f"Tüm arşiv · {sources:,} kayıtlı"
            src_bits = [
                label,
                f"Index: {total:,}",
                f"Hızlı: {light_done:,}",
                f"Yapay zekâ son durumu: {ai_final:,}",
            ]
        src_bits.append(f"Yetim kayıtlar: {orphan_files:,}")
        if waiting_preview:
            src_bits.append(f"Preview bekleyen: {waiting_preview:,}")
        if missing:
            src_bits.append(f"Eksik: {missing:,}")
        rem = status.get("ui_remaining_light")
        if rem is not None:
            src_bits.append(f"Kuyruk: {int(rem):,}")
        reason_summary = str(status.get("remaining_reason_summary") or "").strip()
        if reason_summary:
            src_bits.append(reason_summary)
        legacy = status.get("legacy_pools") or {}
        leg_tex = int(legacy.get("texture") or 0)
        if leg_tex > 0:
            src_bits.append(f"Eski doku kaydı: {leg_tex:,}")
        self.lbl_source_line.setText(" · ".join(src_bits))
        self.lbl_total_files.setText(f"Toplam: {total:,}")

    @staticmethod
    def _display_stage_name(stage: str) -> str:
        key = str(stage or "").strip().lower()
        names = {
            "thumbnail": "Küçük görsel",
            "preview": "Preview",
            "hash": "Hash",
            "metadata": "Metadata",
            "dino": "Embedding — DINO",
            "clip": "Embedding — OpenCLIP",
            "texture": "Doku analizi",
            "semantic": "Anlamsal etiket",
            "dna": "Desen DNA",
            "patch": "Parça Embedding",
            "ocr": "OCR",
        }
        return names.get(key, stage or "—")

    def _paint_retry_and_failed_files(self, status: dict) -> None:
        retry = [str(x).strip() for x in (status.get("retry_file_names") or []) if str(x).strip()]
        failed = [str(x).strip() for x in (status.get("failed_file_names") or []) if str(x).strip()]
        if retry:
            shown = "\n".join(f"• {x}" for x in retry[:25])
            more = "" if len(retry) <= 25 else f"\n… ve {len(retry)-25:,} dosya daha"
            self.lbl_retry_files.setText("Yeniden işleme alınacak dosyalar:\n" + shown + more)
        else:
            self.lbl_retry_files.setText("")
        if failed:
            shown = "\n".join(f"• {x}" for x in failed[:25])
            more = "" if len(failed) <= 25 else f"\n… ve {len(failed)-25:,} dosya daha"
            self.lbl_failed_files.setText("Hatalı dosyalar:\n" + shown + more)
        else:
            self.lbl_failed_files.setText("")

    def _paint_session_lanes(self, status: dict) -> None:
        """Kartlar: Hızlı=path coverage, Genel=AI FINAL coverage. Event/jobs_done yok."""
        _ = self._session_delta
        pools = status.get("artifact_pools") or {}
        total = int(status.get("total") or pools.get("total") or 0)
        # Sistemde mevcut Hızlı tamamlanan (geçmiş + bu oturum path coverage)
        preview = int(
            status.get("fast_completed")
            if status.get("fast_completed") is not None
            else status.get("light_done", 0)
            or 0
        )
        ai_final = int(
            status.get("general_completed")
            or pools.get("ai_final")
            or status.get("ai_final_ready")
            or 0
        )
        terminal_error_files = max(
            0, int(status.get("terminal_error_files") or 0)
        )
        rem_light = max(0, total - preview)
        rem_heavy = max(0, total - ai_final)
        status = {
            **status,
            "total": total,
            "light_done": preview,
            "ai_final_ready": ai_final,
            "heavy_done": ai_final,
            "light_queue_display": rem_light,
            "ui_remaining_light": rem_light,  # kaynak satırı ile kart aynı küme
            "heavy_queue_display": rem_heavy,
            "ui_remaining_heavy": rem_heavy,
        }
        light_job_spm = float(
            status.get("light_job_speed_pm")
            or status.get("jobs_completed_light_1m")
            or 0
        )
        heavy_job_spm = float(
            status.get("heavy_job_speed_pm")
            or status.get("jobs_completed_heavy_1m")
            or 0
        )
        self._apply_simple_summaries(
            status,
            light_speed_pm=self._lane_speed_per_min(
                preview, "light", job_speed_pm=light_job_spm
            ),
            heavy_speed_pm=self._lane_speed_per_min(
                ai_final, "heavy", job_speed_pm=heavy_job_spm
            ),
            light_processing=int(status.get("light_processing", 0) or 0),
            heavy_processing=int(status.get("heavy_processing", 0) or 0),
        )

    @staticmethod
    def _format_light_db_block(status: dict) -> str:
        lines = [
            f"Bekleyen: {int(status.get('light_pending', 0) or 0):,}",
            f"İşlenen: {int(status.get('light_processing', 0) or 0):,}",
            f"Tamam: {int(status.get('light_done', 0) or 0):,}",
            f"Başarısız: {int(status.get('light_failed', 0) or 0):,}",
            f"Preview üretimi tamamlanan: {int(status.get('preview_ready', 0) or 0):,}",
        ]
        report = status.get("failed_report") or {}
        top = list(report.get("top") or [])[:10]
        if top:
            lines.append("--- Başarısız Top 10 ---")
            for item in top:
                label = item.get("label") or item.get("reason") or "?"
                lines.append(f"{label}: {int(item.get('count', 0) or 0):,}")
        buckets = list(report.get("buckets") or [])
        if buckets:
            lines.append("--- Kovalar ---")
            for b in buckets:
                n = int(b.get("count", 0) or 0)
                if n:
                    lines.append(f"{b.get('bucket', '?')}: {n:,}")
        return "\n".join(lines)

    @staticmethod
    def _format_heavy_db_block(status: dict) -> str:
        return (
            f"Bekleyen: {int(status.get('heavy_pending', 0) or 0):,}\n"
            f"Vektör (DINO/CLIP): {int(status.get('embedding_ready', 0) or 0):,}\n"
            f"Parça vektörü: {int(status.get('patch_embedding_ready', 0) or 0):,}\n"
            f"Derin içerik (DNA): {int(status.get('deep_content_ready', status.get('ai_analyzed', 0)) or 0):,}\n"
            f"Semantic: {int(status.get('semantic_tag_count', 0) or 0):,}\n"
            f"Pattern DNA: {int(status.get('pattern_dna_count', 0) or 0):,}\n"
            f"OCR metni: {int(status.get('ocr_done', 0) or 0):,}\n"
            f"Doku: {int(status.get('texture_done', 0) or 0):,}\n"
            f"Yapay zekâ son durumu: {int(status.get('ai_final_ready', 0) or 0):,}\n"
            f"Tamamlanan aşama: {int(status.get('full_done_stage', 0) or 0):,}\n"
            f"Ağır analiz tamam: {int(status.get('heavy_done', 0) or 0):,}\n"
            f"Eksik önizleme: {int(status.get('needs_medium_preview_count', 0) or 0):,}"
        )

    @staticmethod
    def _format_db_block(status: dict) -> str:
        return (
            f"Preview üretimi: {int(status.get('preview_ready', 0) or 0):,}\n"
            f"Embedding (dino/clip): {int(status.get('embedding_ready', 0) or 0):,}\n"
            f"Derin içerik (DNA): {int(status.get('deep_content_ready', status.get('ai_analyzed', 0)) or 0):,}\n"
            f"Semantic: {int(status.get('semantic_tag_count', 0) or 0):,}\n"
            f"Pattern DNA: {int(status.get('pattern_dna_count', 0) or 0):,}\n"
            f"OCR metni: {int(status.get('ocr_done', 0) or 0):,}\n"
            f"Texture: {int(status.get('texture_done', 0) or 0):,}\n"
            f"Yapay zekâ son durumu: {int(status.get('ai_final_ready', 0) or 0):,}"
        )

    def _session_db_deltas(self, status: dict | None = None) -> dict[str, int]:
        """session_plus = current_pool_count - session_start_pool_count."""
        snap = status if status is not None else (self._db_status_snapshot or {})
        base = getattr(self, "_session_pipeline_base", {}) or {}
        keys = (
            "thumbnail_ready",
            "preview_ready",
            "hash_ready",
            "metadata_ready",
            "db_dino_embeddings",
            "db_clip_embeddings",
            "texture_done",
            "semantic_tag_count",
            "pattern_dna_count",
            "object_concept_ready",
            "owlv2_ready",
            "patch_embedding_ready",
            "ai_final_ready",
        )
        out: dict[str, int] = {}
        for key in keys:
            current = self._pipeline_ready(snap, key)
            start = int(base.get(key, 0) or 0)
            out[key] = int(current) - start
        return out

    def _format_session_block(self, data: dict) -> str:
        """Oturum: yalnız havuz session_plus. Event telemetry ayrı (sayaca eklenmez)."""
        _ = data
        deltas = self._session_db_deltas()
        labels = (
            ("Küçük görsel", "thumbnail_ready"),
            ("Preview", "preview_ready"),
            ("Hash", "hash_ready"),
            ("Metadata", "metadata_ready"),
            ("DINO", "db_dino_embeddings"),
            ("OpenCLIP", "db_clip_embeddings"),
            ("Texture", "texture_done"),
            ("Semantic", "semantic_tag_count"),
            ("Pattern DNA", "pattern_dna_count"),
            ("Object Index", "object_concept_ready"),
            ("OWLv2", "owlv2_ready"),
            ("Patch", "patch_embedding_ready"),
            ("Yapay zekâ son durumu", "ai_final_ready"),
        )
        lines = ["Bu Oturum — session_plus (havuz Δ)"]
        for title, key in labels:
            lines.append(f"{title}: {int(deltas.get(key, 0) or 0):+,}")
        return "\n".join(lines)

    def _format_session_summary(self, data: dict) -> str:
        _ = data
        deltas = self._session_db_deltas()
        return (
            f"Thumb: {int(deltas.get('thumbnail_ready', 0) or 0):+,}\n"
            f"Preview: {int(deltas.get('preview_ready', 0) or 0):+,}\n"
            f"Hash: {int(deltas.get('hash_ready', 0) or 0):+,}\n"
            f"Metadata: {int(deltas.get('metadata_ready', 0) or 0):+,}\n"
            f"DINO: {int(deltas.get('db_dino_embeddings', 0) or 0):+,}\n"
            f"OpenCLIP: {int(deltas.get('db_clip_embeddings', 0) or 0):+,}\n"
            f"Texture: {int(deltas.get('texture_done', 0) or 0):+,}\n"
            f"Semantic: {int(deltas.get('semantic_tag_count', 0) or 0):+,}\n"
            f"DNA: {int(deltas.get('pattern_dna_count', 0) or 0):+,}\n"
            f"Patch: {int(deltas.get('patch_embedding_ready', 0) or 0):+,}\n"
            f"Yapay zekâ son durumu: {int(deltas.get('ai_final_ready', 0) or 0):+,}"
        )

    def _apply_live_pipeline_totals(self, data: dict) -> None:
        """Event sayaçları DB toplamına ASLA eklenmez — no-op (geriye uyum)."""
        _ = data
        return

    def _accumulate_session_counters(self, data: dict) -> dict:
        """Kaynak değişiminde sıfırlanan worker sayaçlarını oturumda biriktir."""
        result = dict(data)
        for key in (
            "light_processed",
            "fast_index_done",
            "skipped_unchanged",
            "preview_generated",
            "hash_computed",
            "light_open_once",
            "light_open_reused",
            "light_preview_created",
            "light_metadata_saved",
            "heavy_processed",
            "dino_embedding_saved",
            "clip_embedding_saved",
            "patch_embedding_saved",
            "embedding_complete_saved",
            "ai_final_saved",
            "semantic_saved",
            "dna_saved",
            "ocr_saved",
            "texture_saved",
            "local_cache_read_count",
            "db_metadata_read_count",
            "light_network_original_open_count",
            "full_network_original_open_count",
        ):
            if key not in data:
                continue
            raw = int(data.get(key, 0) or 0)
            previous = self._session_counter_last.get(key)
            offset = int(self._session_counter_offset.get(key, 0) or 0)
            if previous is not None and raw < previous:
                offset += previous
                self._session_counter_offset[key] = offset
            self._session_counter_last[key] = raw
            self._session_counter_values[key] = offset + raw
        result.update(self._session_counter_values)
        return result

    @staticmethod
    def _set_mode_bar(
        bar: QProgressBar,
        done: int,
        *,
        total_files: int,
        pending: int = 0,
        processing: int = 0,
        label: str = "",
    ) -> None:
        total_files = max(0, int(total_files))
        done = max(0, int(done))
        if total_files <= 0:
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("—")
            return
        pct = int(100 * done / total_files)
        bar.setRange(0, total_files)
        bar.setValue(min(done, total_files))
        extra = ""
        if pending or processing:
            extra = f"  · kuyruk {pending:,} · devam {processing:,}"
        prefix = f"{label} " if label else ""
        bar.setFormat(f"{prefix}{pct}%  ({done:,}/{total_files:,}){extra}")
        color = _health_color(pct)
        bar.setStyleSheet(
            f"QProgressBar {{ background:#1f2937; border:none; border-radius:4px; text-align:center; }}"
            f"QProgressBar::chunk {{ background:{color}; border-radius:4px; }}"
        )

    def _pipeline_ready(self, status: dict, key: str) -> int:
        """Stage completed = artifact_pools DISTINCT READY (event yok)."""
        if key == "owlv2_ready":
            ov = status.get("owlv2") or {}
            return int(ov.get("completed") or status.get("owlv2_ready") or 0)
        pools = status.get("artifact_pools") or {}
        pool_map = {
            "thumbnail_ready": "thumbnail",
            "preview_ready": "preview",
            "hash_ready": "hash",
            "metadata_ready": "metadata",
            "verified_preview_ready": "preview",
            "db_dino_embeddings": "dino",
            "db_clip_embeddings": "clip",
            "texture_done": "texture",
            "semantic_tag_count": "semantic",
            "pattern_dna_count": "dna",
            "object_concept_ready": "object_concept",
            "owlv2_ready": "owlv2",
            "patch_embedding_ready": "patch",
            "ai_final_ready": "ai_final",
            "light_done": "light_coverage",
            "heavy_done": "ai_final",
        }
        pool_key = pool_map.get(key)
        if pool_key and pool_key in pools:
            return int(pools.get(pool_key) or 0)
        if key == "db_dino_embeddings":
            return int(status.get("db_dino_embeddings") or 0)
        if key == "db_clip_embeddings":
            return int(status.get("db_clip_embeddings") or 0)
        if key == "light_done":
            return int(
                status.get("fast_completed")
                or status.get("light_done")
                or pools.get("light_coverage")
                or status.get("preview_ready")
                or 0
            )
        return int(status.get(key, 0) or 0)

    def _ai_ready_count(self, status: dict) -> tuple[int, str]:
        """AI Hazır = AI Final (tüm zorunlu artifact'lar) veya darboğaz min."""
        if "ai_final_ready" in status:
            return int(status.get("ai_final_ready", 0) or 0), "Yapay zekâ son durumu"
        scores: list[tuple[str, int]] = []
        labels = {
            "db_dino_embeddings": "DINO",
            "db_clip_embeddings": "OpenCLIP",
            "pattern_dna_count": "DNA",
            "semantic_tag_count": "Semantic",
            "texture_done": "Texture",
            "patch_embedding_ready": "Patch",
        }
        for key in _AI_READY_KEYS:
            scores.append((labels.get(key, key), self._pipeline_ready(status, key)))
        if not scores:
            return 0, "—"
        bottleneck_label, bottleneck_n = min(scores, key=lambda x: x[1])
        heavy = int(status.get("heavy_done", 0) or 0)
        if heavy > 0 and heavy < bottleneck_n:
            return heavy, "Ağır analiz tamam"
        return bottleneck_n, bottleneck_label

    def _compute_health(self, status: dict, total: int) -> tuple[int, list[str]]:
        if total <= 0:
            return 0, []
        weighted = 0.0
        weight_sum = 0.0
        badges: list[str] = []
        for title, key, _ in _PIPELINE_ROWS:
            ready = self._pipeline_ready(status, key)
            pct = int(100 * ready / total)
            w = float(_HEALTH_WEIGHTS.get(key, 0))
            if w <= 0:
                continue
            weighted += pct * w
            weight_sum += w
            badges.append(f"{_health_emoji(pct)} {title}")
        # Hızlı İndeks katkısı
        light = self._pipeline_ready(status, "light_done")
        light_pct = int(100 * light / total)
        w_hash = float(_HEALTH_WEIGHTS.get("light_done", 5))
        weighted += light_pct * w_hash
        weight_sum += w_hash
        badges.append(f"{_health_emoji(light_pct)} Hızlı İndeks")
        health = int(weighted / weight_sum) if weight_sum else 0
        return health, badges

    def _apply_pipeline(self, status: dict) -> None:
        total = int(
            status.get("total")
            or status.get("total_searchable")
            or status.get("archive_total")
            or 0
        )
        session_deltas = (
            self._session_db_deltas(status) if self._session_active else {}
        )
        legacy = status.get("legacy_pools") or {}
        stage_stats = status.get("stage_stats") or {}
        for title, key in getattr(self, "_ai_display_rows", []):
            ready = self._pipeline_ready(status, key)
            extra = ""
            row_total = total
            ready = max(0, min(ready, total))
            rem = max(0, total - ready)
            pct = int(100 * ready / total) if total > 0 else 0
            # stage_stats remaining tercih
            pool_key = {
                "thumbnail_ready": "thumbnail",
                "preview_ready": "preview",
                "db_dino_embeddings": "dino",
                "db_clip_embeddings": "openclip",
                "texture_done": "texture",
                "semantic_tag_count": "semantic",
                "pattern_dna_count": "dna",
                "object_concept_ready": "object_concept",
                "patch_embedding_ready": "patch",
                "ocr_done": "ocr",
                "ai_final_ready": "ai_final",
            }.get(key)
            if pool_key and pool_key in stage_stats:
                rem = int(stage_stats[pool_key].get("remaining", rem) or rem)
                pct = int(stage_stats[pool_key].get("percent", pct) or pct)
            leg = 0
            if pool_key:
                leg_key = "clip" if pool_key == "openclip" else pool_key
                leg = int(legacy.get(leg_key, 0) or 0)
            if key in self._pipeline_rows:
                owl_color = _owl_bar_color(status, int(pct)) if key == "owlv2_ready" else None
                self._pipeline_rows[key].set_values(
                    ready,
                    total,
                    int(pct),
                    session_delta=(
                        session_deltas.get(key) if self._session_active else None
                    ),
                    remaining=rem,
                    legacy=leg,
                    bar_color=owl_color,
                )

        health, badges = self._compute_health(status, total)
        if self.maint_group.isVisible():
            self.lbl_health.show()
            self.lbl_health_checks.show()
            self.lbl_health.setText(f"Sistem Sağlığı  %{health}")
            self.lbl_health_checks.setText("  ·  ".join(badges))
        else:
            self.lbl_health.hide()
            self.lbl_health_checks.hide()

        self.lbl_total_files.setText(f"Toplam: {total:,}")
        basic = int(status.get("light_done", 0) or 0)
        ai_ready, bottleneck = self._ai_ready_count(status)
        ai_ready = max(0, min(ai_ready, total))
        self.lbl_basic_ready.setText(f"Temel Hazır: {basic:,}")
        self.lbl_ai_ready.setText(f"AI Hazır: {ai_ready:,}")
        self.lbl_bottleneck.setText(f"Darboğaz: {bottleneck}")
        search_ready = int(
            status.get("search_ready")
            or status.get("embedding_ready")
            or status.get("db_dino_embeddings")
            or 0
        )
        self.lbl_search_ready.setText(f"Search Ready: {search_ready:,}")
        self.bar_full_ready.setFormat(f"FULL READY {ai_ready:,}/{total:,}")

    def _lane_speed_per_min(
        self, done: int, lane: str, *, job_speed_pm: float = 0.0
    ) -> float:
        """Done (coverage) Δ + JobStore recent completes — ikisinden yüksek olan.

        Formül:
          cov_spm = Δdone / Δt_sec * 60
          job_spm = son 60s tamamlanan job / dk  (coverage düzken bile akar)
          return max(cov_ema, job_spm)
        """
        now = time.monotonic()
        samples = self._lane_speed_samples.setdefault(lane, deque())
        done = max(0, int(done))
        if not samples or samples[-1][1] != done:
            samples.append((now, done))
        window = 180.0 if lane == "heavy" else 60.0
        cutoff = now - window
        while len(samples) > 2 and samples[0][0] < cutoff:
            samples.popleft()
        cov_ema = 0.0
        if len(samples) >= 2:
            dt = samples[-1][0] - samples[0][0]
            dd = samples[-1][1] - samples[0][1]
            min_dt = 3.0 if lane == "heavy" else 1.5
            if dt >= min_dt and dd > 0:
                raw = (dd / dt) * 60.0
                prev = float(self._lane_speed_ema.get(lane, 0.0) or 0.0)
                if prev <= 0:
                    cov_ema = raw
                else:
                    cov_ema = 0.25 * raw + 0.75 * prev
                    cov_ema = min(cov_ema, max(prev * 2.5, 0.5 * raw + 0.5 * prev))
                self._lane_speed_ema[lane] = cov_ema
            elif dd == 0:
                prev = float(self._lane_speed_ema.get(lane, 0.0) or 0.0)
                cov_ema = prev * 0.85
                self._lane_speed_ema[lane] = cov_ema
            else:
                cov_ema = float(self._lane_speed_ema.get(lane, 0.0) or 0.0)
        else:
            cov_ema = float(self._lane_speed_ema.get(lane, 0.0) or 0.0)
        job_spm = max(0.0, float(job_speed_pm or 0.0))
        return max(cov_ema, job_spm)

    def _update_lane_perf(
        self,
        *,
        light_done: int,
        light_total: int,
        heavy_done: int,
        heavy_total: int,
    ) -> None:
        _ = (light_done, light_total, heavy_done, heavy_total)

    def _apply_db_status(self, status: dict) -> None:
        self.lbl_source_count.setText(
            f"Toplam kaynak: {int(status.get('source_count', 0) or 0):,}"
        )
        searchable = int(
            status.get("total", 0)
            or status.get("total_searchable", 0)
            or status.get("archive_total", 0)
            or 0
        )
        self.lbl_searchable.setText(f"İşlenebilir görsel: {searchable:,}")
        self.lbl_embedding_status.setText(
            f"Embedding: {status.get('embedding_status_label', '—')}"
        )
        self.lbl_ocr_status.setText(f"OCR: {status.get('ocr_status_label', '—')}")
        ga = str(status.get("general_ai_status_label") or "Bekliyor")
        patch_l = str(status.get("patch_status_label") or "Bekliyor")
        ocr_l = str(status.get("ocr_status_label") or "Bekliyor")
        if hasattr(self, "lbl_general_ai_status"):
            self.lbl_general_ai_status.setText(f"Genel AI: {ga}")
            self.lbl_patch_lane_status.setText(f"PATCH: {patch_l}")
            self.lbl_ocr_lane_status.setText(f"OCR: {ocr_l}")
            en = bool(status.get("post_ga_buttons_enabled"))
            self._apply_post_ga_button_state(en)
        self.lbl_full_done_status.setText(
                status.get("full_done_label", "Tamamlanan aşama: —")
        )
        self.lbl_light_block.setText(self._format_light_db_block(status))
        self.lbl_heavy_block.setText(self._format_heavy_db_block(status))
        self.lbl_db_block.setText(self._format_db_block(status))
        self.lbl_dev_block.setText(
            self._format_light_db_block(status)
            + "\n\n"
            + self._format_heavy_db_block(status)
            + "\n\n"
            + self._format_db_block(status)
        )
        self._apply_simple_summaries(status)
        self._apply_pipeline(status)
        if not self._session_active:
            full_net = int(status.get("full_network_original_open_count", 0) or 0)
            violation = int(status.get("deep_original_open_violation_count", 0) or 0)
            self.lbl_deep_violation.setText(
                f"Deep pass ağ orijinal: {full_net:,} · ihlal: {violation:,}"
            )

    def update_status(self, status: dict) -> None:
        if (
            self._session_active
            and status.get("lanes_only")
            and not status.get("pipeline_counts_exact")
            and self._db_status_snapshot
        ):
            # Eski lane yenilemesi ağır DB toplamlarını eski cache'e düşürmesin
            merged = dict(status)
            for key in (
                "preview_ready",
                "db_dino_embeddings",
                "db_clip_embeddings",
                "embedding_ready",
                "patch_embedding_ready",
                "semantic_tag_count",
                "pattern_dna_count",
                "texture_done",
                "ocr_done",
                "ai_final_ready",
                "artifact_pools",
                "legacy_pools",
                "stage_stats",
            ):
                if key in self._db_status_snapshot:
                    merged[key] = self._db_status_snapshot[key]
            status = merged
        # Index oturumunda: StatusWorker processing=0 ile claim'i ezmesin.
        # Havuz sayıları DB SSOT'tan gelir (geri gidebilir — physical reclassify).
        if self._session_active and self._db_status_snapshot:
            prev = self._db_status_snapshot
            status = dict(status)
            in_light = int(status.get("light_processing") or 0)
            in_heavy = int(status.get("heavy_processing") or 0)
            prev_light = int(prev.get("light_processing") or 0)
            prev_heavy = int(prev.get("heavy_processing") or 0)
            # Lane bağımsız: yalnız 0 gelen şeridi önceki canlı değerle koru
            if in_light == 0 and prev_light > 0 and not self._progress_frozen:
                # Status claimed=0 ama worker hâlâ dosya gösteriyorsa koru
                cur = str(prev.get("current_filename") or status.get("current_filename") or "")
                if cur or int(prev.get("claimed_light_jobs") or 0) > 0:
                    status["light_processing"] = prev_light
            if in_heavy == 0 and prev_heavy > 0 and not self._progress_frozen:
                cur = str(prev.get("current_filename") or status.get("current_filename") or "")
                phase = str(
                    (status.get("current_file_info") or prev.get("current_file_info") or {}).get(
                        "phase"
                    )
                    or ""
                ).lower()
                if phase == "claimed" or cur or int(prev.get("claimed_heavy_jobs") or 0) > 0:
                    status["heavy_processing"] = prev_heavy
            # Job throughput alanlarını da birleştir
            for k in (
                "light_job_speed_pm",
                "heavy_job_speed_pm",
                "jobs_completed_light_1m",
                "jobs_completed_heavy_1m",
                "claimed_light_jobs",
                "claimed_heavy_jobs",
            ):
                if status.get(k) is None and prev.get(k) is not None:
                    status[k] = prev[k]
        self._db_status_snapshot = dict(status)
        if self._session_active:
            self.lbl_source_count.setText(
                f"Toplam kaynak: {int(status.get('source_count', 0) or 0):,}"
            )
            searchable = int(
                status.get("total", 0)
                or status.get("total_searchable", 0)
                or status.get("archive_total", 0)
                or 0
            )
            self.lbl_searchable.setText(f"İşlenebilir görsel: {searchable:,}")
            ga = str(status.get("general_ai_status_label") or "Bekliyor")
            patch_l = str(status.get("patch_status_label") or "Bekliyor")
            ocr_l = str(status.get("ocr_status_label") or "Bekliyor")
            self.lbl_general_ai_status.setText(f"Genel AI: {ga}")
            self.lbl_patch_lane_status.setText(f"PATCH: {patch_l}")
            self.lbl_ocr_lane_status.setText(f"OCR: {ocr_l}")
            en = bool(status.get("post_ga_buttons_enabled"))
            self._apply_post_ga_button_state(en)
            self.lbl_light_block.setText(self._format_light_db_block(status))
            self.lbl_heavy_block.setText(self._format_heavy_db_block(status))
            self.lbl_db_block.setText(self._format_db_block(status))
            self.lbl_session_block.setText(self._format_session_summary({}))
            if self._progress_frozen or self._control_state in ("paused", "stopped"):
                # Duraklat/Durdur: saf DB — hız/şişirme yok
                self._lane_processing = {"light": 0, "heavy": 0}
                self._apply_simple_summaries(status)
                self.lbl_fast_perf.setText("Hız         : —\nTahmini     : —")
                self.lbl_general_perf.setText("Hız         : —\nTahmini     : —")
            else:
                self._paint_session_lanes(status)
            self._apply_pipeline(status)
            return
        self._apply_db_status(status)

    def _window_speed(self, done: int) -> float:
        now = time.monotonic()
        samples = self._speed_samples
        if not samples or done != samples[-1][1]:
            samples.append((now, done))
        cutoff = now - self._speed_window_sec
        while len(samples) > 2 and samples[0][0] < cutoff:
            samples.popleft()
        if len(samples) < 2:
            return 0.0
        t0, d0 = samples[0]
        t1, d1 = samples[-1]
        dt = t1 - t0
        dd = d1 - d0
        if dt <= 0 or dd <= 0:
            return 0.0
        return dd / dt

    @staticmethod
    def _guess_next_stage(task: str, phase: str) -> str:
        t = (task or "").lower()
        if "thumb" in t or "önizleme" in t or "preview" in t:
            return "Hash"
        if "hash" in t or "metadata" in t:
            return "Preview"
        if "embed" in t or "dino" in t or "clip" in t:
            return "Pattern DNA"
        if "dna" in t:
            return "Semantic / Texture"
        if "semantic" in t:
            return "OCR / Patch"
        if "ocr" in t:
            return "Texture / Patch"
        if "texture" in t:
            return "Patch"
        if "patch" in t:
            return "FAISS / Sonraki dosya"
        if phase in ("Hızlı Index",):
            return "Genel Index (AI)"
        if phase in ("Ağır Analiz", "Genel Index"):
            return "Sonraki dosya"
        return "—"

    @staticmethod
    def _flow_chain(task: str, phase: str) -> str:
        steps = [
            "Tarama",
            "Thumbnail",
            "Hash",
            "Preview",
            "Embedding",
            "DNA",
            "Semantic",
            "Texture",
            "OCR",
            "Patch",
            "FAISS",
        ]
        t = (task or "").lower()
        idx = 0
        if "tarama" in t or "scan" in t:
            idx = 0
        elif "thumb" in t or "önizleme" in t:
            idx = 1
        elif "hash" in t or "metadata" in t:
            idx = 2
        elif "preview" in t:
            idx = 3
        elif "embed" in t or "dino" in t or "clip" in t:
            idx = 4
        elif "dna" in t:
            idx = 5
        elif "semantic" in t:
            idx = 6
        elif "texture" in t:
            idx = 7
        elif "ocr" in t:
            idx = 8
        elif "patch" in t:
            idx = 9
        elif "faiss" in t:
            idx = 10
        elif phase in ("Hızlı Index",):
            idx = 1
        elif phase in ("Ağır Analiz", "Genel Index"):
            idx = 4
        prev = steps[idx - 1] if idx > 0 else "—"
        cur = steps[idx]
        nxt = steps[idx + 1] if idx + 1 < len(steps) else "—"
        return f"{prev}  →  {cur}  →  {nxt}"

    def update_progress(self, data: dict) -> None:
        if self._progress_frozen and self._control_state in ("paused", "stopped"):
            # Duraklat/Durdur sonrası gelen geç olaylar sayaç/hız şişirmesin
            current = str(data.get("current_file", "") or "").strip()
            if current and self._control_state == "paused":
                self.lbl_current_file.setText(f"Duraklatıldı: {current}")
            return
        stage = str(data.get("stage", "") or "").strip()
        phase = str(data.get("active_phase", "") or "").strip()
        mode_from = str(data.get("index_mode") or "").strip()
        if mode_from:
            self.set_active_index_mode(
                mode_from, status=self._control_state or "running"
            )
        elif data.get("dual_pipeline") or data.get("independent_lanes"):
            if self._active_index_mode in ("", "standard"):
                self.set_active_index_mode(
                    "complete", status=self._control_state or "running"
                )
        if not phase and data.get("index_pass") == "light":
            phase = "Hızlı Index"
        elif not phase and data.get("index_pass") == "full":
            phase = "Ağır Analiz"
        elif not phase and stage == "ai_loading":
            phase = "AI yükleniyor"
        elif not phase and stage in ("backlog_only", "process_batch_start", "scan_start"):
            phase = "Hızlı Index" if data.get("index_pass") != "full" else "Güçlü Analiz"
        elif not phase and stage == "queue_prepare":
            phase = str(data.get("active_phase") or "Kuyruk hazırlanıyor")
        elif not phase and self._session_active and stage:
            # "Başlatılıyor…" da kalmasın — herhangi bir index olayı
            phase = "Index çalışıyor"
        if phase:
            label = _PHASE_LABELS.get(phase, phase)
            self.lbl_active_phase.setText(f"Mod: {label}")

        task = str(data.get("pipeline_task", "") or "").strip()
        prev_task = self.lbl_current_task.text()
        if task:
            if f"Aşama: {task}" != prev_task:
                self._task_started = time.monotonic()
            self.lbl_current_task.setText(f"Aşama: {task}")
            self.lbl_next_stage.setText(
                f"Sonraki: {self._guess_next_stage(task, phase)}"
            )
            self.lbl_flow.setText(f"Akış: {self._flow_chain(task, phase)}")
        elif phase:
            self.lbl_flow.setText(f"Akış: {self._flow_chain('', phase)}")
        elif stage:
            self.lbl_current_task.setText(f"Aşama: {stage}")

        current = str(data.get("current_file", "") or "").strip()
        fname = str(data.get("current_filename") or "").strip()
        src = str(data.get("current_source") or "").strip()
        stage_name = self._display_stage_name(str(data.get("current_stage") or "").strip())
        worker = data.get(
            "current_worker", data.get("worker_id", data.get("worker", data.get("thread_name")))
        )
        elapsed = float(data.get("current_elapsed_sec") or 0)
        proc_n = int(data.get("processing") or 0)
        if fname and proc_n:
            parts = [f"İşleniyor: {fname}"]
            if src:
                parts.append(f"Kaynak: {src}")
            if stage_name:
                parts.append(f"Aşama: {stage_name}")
            if worker is not None and str(worker).strip():
                parts.append(f"Worker: {worker}")
            if elapsed > 0:
                parts.append(f"Süre: {elapsed:.1f} sn")
            if data.get("stall_warning"):
                parts.append("STALL / TIMEOUT riski")
            self.lbl_current_file.setText(" · ".join(parts))
        elif current:
            self.lbl_current_file.setText(f"İşleniyor: {current}")

        if worker is not None and str(worker).strip() and not fname:
            self.lbl_worker.setText(f"İş parçacığı: {worker}")
        elif self._session_active:
            self.lbl_worker.setText(
                f"İş parçacığı: {worker or 'IndexWorker'}"
                + (f" · Processing={proc_n}" if proc_n else "")
            )

        if self._task_started:
            ms = int((time.monotonic() - self._task_started) * 1000)
            self.lbl_task_ms.setText(f"Süre: {ms:,} ms")

        jobs_total = int(data.get("jobs_total", 0) or 0)
        jobs_done = int(data.get("jobs_done", 0) or 0)
        if jobs_total > 0:
            self._jobs_total = jobs_total
        if jobs_done >= 0:
            self._jobs_done = max(self._jobs_done, jobs_done)

        scanned = int(data.get("scanned", 0) or 0)
        processed = int(data.get("processed", 0) or 0)
        total_for_bar = self._jobs_total or scanned or 0
        done_for_bar = self._jobs_done if self._jobs_total else processed
        _ = (total_for_bar, done_for_bar)

        light_delta = int(
            data.get("light_processed", data.get("fast_index_done", 0)) or 0
        )
        heavy_delta = int(data.get("heavy_processed", 0) or 0)
        index_pass = str(data.get("index_pass", "") or "")
        # jobs_done fallback KALDIRILDI — atlanan işler kuyruğu sahte düşürüyordu
        # (106841 DB iken 110k / kuyruk 6021 gibi)
        light_delta = max(int(self._session_delta.get("light", 0) or 0), light_delta)
        heavy_delta = max(int(self._session_delta.get("heavy", 0) or 0), heavy_delta)
        self._session_delta["light"] = light_delta
        self._session_delta["heavy"] = heavy_delta

        # İşleniyor: claim stage'e göre şerit (complete ≠ yalnız light)
        if current:
            stage_l = str(data.get("current_stage") or "").strip().lower()
            heavy_stages = {
                "hash",
                "metadata",
                "dino",
                "clip",
                "texture",
                "semantic",
                "dna",
                "patch",
                "ocr",
                "object_concept",
                "owlv2",
            }
            if index_pass == "full" or stage_l in heavy_stages or phase in (
                "Ağır Analiz",
                "Genel Index",
                "Genel AI",
                "Güçlü Analiz",
            ):
                self._lane_processing = {"light": 0, "heavy": 1}
            elif index_pass == "light" or stage_l in ("preview", "thumbnail") or phase in (
                "Hızlı Index",
            ):
                self._lane_processing = {"light": 1, "heavy": 0}
            elif phase in ("Hızlı + Genel", "Hızlı + Genel AI", "complete"):
                # Bilinmeyen stage: her iki şeridi de işaretleme — SSOT processing kullan
                pass
        elif not self._session_active:
            self._lane_processing = {"light": 0, "heavy": 0}

        # Event telemetry birikir (debug); kullanıcı sayaçları yalnız havuz SSOT.
        session_data = self._accumulate_session_counters(data)

        # Önce V3 SSOT birleştir — sonra boya (stale flicker yok)
        snap = dict(self._db_status_snapshot or {})
        if data.get("engine") == "index_v3" and (
            data.get("total") is not None
            or data.get("light_done") is not None
            or data.get("artifact_pools") is not None
        ):
            for key in (
                "total",
                "ssot_total",
                "archive_total",
                "total_searchable",
                "light_done",
                "ai_final_ready",
                "heavy_done",
                "light_queue_display",
                "heavy_queue_display",
                "light_pending",
                "heavy_pending",
                "light_processing",
                "heavy_processing",
                "ui_remaining_light",
                "ui_remaining_heavy",
                "status_scope_source_ids",
                "scope_mode",
                "source_count",
                "pending_jobs",
                "v3_executable_jobs",
                "preview_ready",
                "verified_preview_ready",
                "thumbnail_ready",
                "hash_ready",
                "metadata_ready",
                "db_dino_embeddings",
                "db_clip_embeddings",
                "patch_embedding_ready",
                "texture_done",
                "semantic_tag_count",
                "pattern_dna_count",
                "artifact_pools",
                "legacy_pools",
                "stage_stats",
                "fast_completed",
                "general_completed",
                "session_plus",
                "live_stages",
                "remaining_display",
                "executable_display",
                "claimed_jobs",
                "claimed_fresh_jobs",
                "claimed_light_jobs",
                "claimed_heavy_jobs",
                "light_job_speed_pm",
                "heavy_job_speed_pm",
                "jobs_completed_light_1m",
                "jobs_completed_heavy_1m",
                "retry_light_jobs",
                "retry_heavy_jobs",
                "failed_permanent_jobs",
            ):
                if key in data and data.get(key) is not None:
                    snap[key] = data[key]
            # İşleniyor: aktif claim → light/heavy; idle/done → 0
            # StatusWorker processing=0 ile canlı claim'i ezmesin.
            art = str(data.get("current_stage") or data.get("pipeline_task") or "").lower()
            proc_live = int(data.get("processing") or 0)
            light_live = data.get("light_processing")
            heavy_live = data.get("heavy_processing")
            phase_claim = str(
                (data.get("current_file_info") or {}).get("phase")
                or ""
            ).lower()
            has_live_claim = phase_claim == "claimed" or (
                proc_live > 0 and bool(str(data.get("current_filename") or "").strip())
            )
            if light_live is not None or heavy_live is not None:
                if light_live is not None:
                    snap["light_processing"] = max(0, int(light_live or 0))
                if heavy_live is not None:
                    snap["heavy_processing"] = max(0, int(heavy_live or 0))
            elif proc_live and art:
                light_arts = ("thumbnail", "preview")
                is_light = any(a in art for a in light_arts)
                snap["light_processing"] = 1 if is_light else 0
                snap["heavy_processing"] = 0 if is_light else 1
            elif has_live_claim:
                # processing alanı 0 olsa bile claim canlı — şeridi koru / stage'den yaz
                light_arts = ("thumbnail", "preview")
                is_light = any(a in art for a in light_arts) if art else False
                if is_light:
                    snap["light_processing"] = max(
                        1, int(snap.get("light_processing") or 0)
                    )
                else:
                    snap["heavy_processing"] = max(
                        1, int(snap.get("heavy_processing") or 0)
                    )
            elif phase_claim == "idle" or str(data.get("stage") or "") in (
                "index_done",
                "complete",
            ):
                snap["light_processing"] = 0
                snap["heavy_processing"] = 0
            # else: processing=0 ama idle değil → önceki snap değerlerini koru
            self._db_status_snapshot = snap

        if snap and not (
            self._progress_frozen and self._control_state in ("paused", "stopped")
        ):
            self._paint_session_lanes(snap)
            self._paint_retry_and_failed_files(snap)
            self._apply_pipeline(snap)
            # Worker emit session_plus varsa onu kullan (tek SSOT)
            if isinstance(snap.get("session_plus"), dict) and snap.get("session_plus"):
                sp = snap["session_plus"]
                self.lbl_session_block.setText(
                    f"Thumb: {int(sp.get('thumbnail', 0) or 0):+,}\n"
                    f"Preview: {int(sp.get('preview', 0) or 0):+,}\n"
                    f"DINO: {int(sp.get('dino', 0) or 0):+,}\n"
                    f"OpenCLIP: {int(sp.get('clip', 0) or 0):+,}\n"
                    f"Texture: {int(sp.get('texture', 0) or 0):+,}\n"
                    f"Semantic: {int(sp.get('semantic', 0) or 0):+,}\n"
                    f"DNA: {int(sp.get('dna', 0) or 0):+,}\n"
                    f"Patch: {int(sp.get('patch', 0) or 0):+,}\n"
                    f"AI Final: {int(sp.get('ai_final', 0) or 0):+,}"
                )
            else:
                self.lbl_session_block.setText(self._format_session_summary(session_data))
            self.lbl_session_debug.setText(self._format_session_block(session_data))
            if self.maint_group.isVisible():
                self.lbl_dev_block.setText(
                    self._format_light_db_block(snap)
                    + "\n\n"
                    + self._format_heavy_db_block(snap)
                )

        if str(data.get("stage", "")) in ("complete", "index_done"):
            self.set_control_state("completed")
            self.lbl_current_file.setText("İşleniyor: —")
            self._lane_processing = {"light": 0, "heavy": 0}
            if self._db_status_snapshot is not None:
                self._db_status_snapshot["light_processing"] = 0
                self._db_status_snapshot["heavy_processing"] = 0

    def freeze_progress(self, frozen: bool = True) -> None:
        """Durdur/Duraklat: hız ve şişen sayaç güncellemesi kesilir."""
        self._progress_frozen = bool(frozen)
        if frozen:
            self._lane_processing = {"light": 0, "heavy": 0}
            self._lane_speed_ema = {"light": 0.0, "heavy": 0.0}
            self._lane_speed_samples = {"light": deque(), "heavy": deque()}
            self.lbl_fast_perf.setText("Hız         : —\nTahmini     : —")
            self.lbl_general_perf.setText("Hız         : —\nTahmini     : —")
            cur = self.lbl_current_file.text()
            if cur.startswith("İşleniyor:"):
                self.lbl_current_file.setText(
                    "Duraklatıldı:" + cur.split(":", 1)[-1]
                )
            if self._db_status_snapshot:
                self._apply_simple_summaries(self._db_status_snapshot)

    def set_indexing(
        self,
        active: bool,
        *,
        paused: bool = False,
        terminal: str | None = None,
    ) -> None:
        if active:
            self.set_control_state("paused" if paused else "running")
            if paused:
                self.freeze_progress(True)
            else:
                self._progress_frozen = False
        else:
            self.set_control_state(terminal or "stopped")
            # Tamamlandı/Durdu: sayaçlar donsun ama Başlat butonları açık kalsın
            self.freeze_progress(True)
            self.end_session()
            # Güvence: start butonları her terminal durumda tıklanabilir
            self._apply_button_state(self._control_state)
