"""Sağ inspector paneli — önizleme, skor, doku, dosya, kaynak."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.category_predictions import category_predictions_from_result, prediction_label
from core.pattern_explanation import build_result_explanation
from core.search_explanation_ui import format_why_html, format_why_lines
from core.search_engine import SearchResult
from core.utils import format_file_size


def _scroll_text(content: str) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lbl = QLabel(content)
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
    lay.addWidget(lbl)
    lay.addStretch()
    scroll.setWidget(inner)
    return scroll


def _scroll_form(rows: list[tuple[str, QLabel]]) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    inner = QWidget()
    form = QFormLayout(inner)
    form.setContentsMargins(8, 8, 8, 8)
    form.setSpacing(10)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignTop)
    for title, value_lbl in rows:
        key = QLabel(f"<b>{title}</b>")
        value_lbl.setWordWrap(True)
        value_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(key, value_lbl)
    form.addRow(QWidget())
    scroll.setWidget(inner)
    return scroll


class _RelationLoadSignals(QObject):
    finished = Signal(int, int, object)  # gen, file_id, bundle


class _RelationLoadJob(QRunnable):
    def __init__(self, gen: int, file_id: int, db_path: str, stage: str):
        super().__init__()
        self.gen = int(gen)
        self.file_id = int(file_id)
        self.db_path = str(db_path or "")
        self.stage = str(stage or "full")
        self.signals = _RelationLoadSignals()

    def run(self) -> None:
        bundle = None
        try:
            from core.db import Database
            from core.pattern_relations import get_pattern_relations

            db = Database(self.db_path, read_only=True)
            bundle = get_pattern_relations(
                db,
                self.file_id,
                db_path=self.db_path,
                stage=self.stage,
                include_similar=True,
            )
        except Exception as exc:
            from core.pattern_relations import RelationBundle

            bundle = RelationBundle(
                source_file_id=self.file_id, error=str(exc)[:200]
            )
        self.signals.finished.emit(self.gen, self.file_id, bundle)


class InspectorPanel(QWidget):
    open_folder = Signal(int)
    open_file = Signal(int)
    search_similar = Signal(int)
    feedback_clicked = Signal(str, str)  # action, label
    teach_tag_clicked = Signal(str)  # custom tag text
    learn_saved = Signal(str, str, str)  # parent, child, tag
    ai_prediction_action = Signal(str, str, str)  # action, category_path, scope

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: SearchResult | None = None
        self._family_reject_report: list[dict] = []
        self._simple_mode = True
        self._db_path: str = ""
        self._rel_gen: int = 0
        self._detail_source_pix: QPixmap | None = None
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self._build_ui()
        from ui.thumbnail_scheduler import get_thumbnail_scheduler

        sched = get_thumbnail_scheduler()
        sched.thumbnail_ready.connect(self._on_scheduler_thumb)
        sched.thumbnail_failed.connect(self._on_scheduler_thumb_failed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("<b>Sonuç Detayı</b>"))

        self.tabs = QTabWidget()
        self.tab_preview = QWidget()
        self.tab_preview.setMinimumWidth(0)
        self.tab_preview.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        prev_l = QVBoxLayout(self.tab_preview)
        prev_l.setContentsMargins(0, 0, 0, 0)
        self.thumb = QLabel()
        self.thumb.setMinimumSize(0, 160)
        self.thumb.setMaximumHeight(520)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.thumb.setStyleSheet(
            "background:#232a35;border:1px solid #2a3340;border-radius:6px;color:#7c8698;"
        )
        self._detail_native_size: tuple[int, int] = (0, 0)
        self.lbl_preview_name = QLabel("—")
        self.lbl_preview_name.setWordWrap(True)
        self.lbl_preview_name.setMinimumWidth(0)
        self.lbl_preview_name.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.lbl_preview_src = QLabel("—")
        self.lbl_preview_src.setWordWrap(True)
        self.lbl_preview_src.setMinimumWidth(0)
        self.lbl_preview_src.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        prev_l.addWidget(self.thumb)
        prev_l.addWidget(self.lbl_preview_name)
        prev_l.addWidget(self.lbl_preview_src)

        self.grp_ai_prediction = QGroupBox("AI Tahmini")
        ai_l = QVBoxLayout(self.grp_ai_prediction)
        self.lbl_ai_prediction = QLabel("Kararsız")
        self.lbl_ai_prediction.setWordWrap(True)
        self.lbl_ai_prediction.setStyleSheet("font-weight:600;")
        ai_l.addWidget(self.lbl_ai_prediction)
        ai_buttons = QGridLayout()
        ai_buttons.setHorizontalSpacing(6)
        ai_buttons.setVerticalSpacing(6)
        self.btn_ai_correct = QPushButton("Doğru")
        self.btn_ai_wrong = QPushButton("Yanlış")
        self.btn_ai_edit = QPushButton("Düzenle")
        self.btn_ai_correct.hide()
        self.btn_ai_wrong.hide()
        self.btn_ai_correct.clicked.connect(
            lambda: self.ai_prediction_action.emit(
                "correct", self._top_prediction_path(), self._prediction_scope()
            )
        )
        self.btn_ai_wrong.clicked.connect(
            lambda: self.ai_prediction_action.emit(
                "wrong", self._top_prediction_path(), self._prediction_scope()
            )
        )
        self.btn_ai_edit.clicked.connect(
            lambda: self.ai_prediction_action.emit(
                "edit", self._top_prediction_path(), self._prediction_scope()
            )
        )
        for btn in (self.btn_ai_correct, self.btn_ai_wrong, self.btn_ai_edit):
            btn.setMinimumHeight(30)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
            )
        ai_buttons.addWidget(self.btn_ai_correct, 0, 0)
        ai_buttons.addWidget(self.btn_ai_wrong, 0, 1)
        ai_buttons.addWidget(self.btn_ai_edit, 1, 0, 1, 2)
        ai_l.addLayout(ai_buttons)
        self._ai_advanced = QWidget()
        ai_adv_l = QVBoxLayout(self._ai_advanced)
        ai_adv_l.setContentsMargins(0, 0, 0, 0)
        self.cmb_ai_scope = QComboBox()
        self.cmb_ai_scope.addItem("Sadece bu dosya", "single")
        self.cmb_ai_scope.addItem("Birebir kopyalar", "exact")
        self.cmb_ai_scope.addItem("Aynı desen ailesindeki benzerler", "family")
        self.cmb_ai_scope.addItem("Gelecekteki tahminleri etkilesin", "future")
        self.btn_ai_undo = QPushButton("Son Düzeltmeyi Geri Al")
        self.btn_ai_undo.clicked.connect(
            lambda: self.ai_prediction_action.emit("undo", "", self._prediction_scope())
        )
        self.cmb_ai_scope.hide()
        ai_adv_l.addWidget(self.btn_ai_undo)
        ai_l.addWidget(self._ai_advanced)
        prev_l.addWidget(self.grp_ai_prediction)

        self.grp_pattern_dna = QGroupBox("Pattern DNA")
        dna_l = QVBoxLayout(self.grp_pattern_dna)
        self.lbl_pattern_dna = QLabel("—")
        self.lbl_pattern_dna.setWordWrap(True)
        self.lbl_pattern_dna.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        dna_l.addWidget(self.lbl_pattern_dna)
        prev_l.addWidget(self.grp_pattern_dna)

        self.grp_ai_explanation = QGroupBox("Neden bu sonuç ▸")
        self.grp_ai_explanation.setObjectName("grp_ai_explanation")
        self.grp_ai_explanation.setCheckable(True)
        self.grp_ai_explanation.setChecked(False)  # start COLLAPSED
        self.grp_ai_explanation.setFlat(False)
        expl_l = QVBoxLayout(self.grp_ai_explanation)
        self.lbl_ai_explanation = QLabel("")
        self.lbl_ai_explanation.setWordWrap(True)
        self.lbl_ai_explanation.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        expl_l.addWidget(self.lbl_ai_explanation)
        self._why_base_title = "Neden bu sonuç"
        self.grp_ai_explanation.toggled.connect(self._on_why_toggled)
        self._sync_why_group_title()
        prev_l.addWidget(self.grp_ai_explanation)

        self.grp_relations = QGroupBox("BU DESENİN DİĞERLERİ")
        self._rel_layout = QVBoxLayout(self.grp_relations)
        self._rel_layout.setSpacing(6)
        self.lbl_relations_status = QLabel("")
        self.lbl_relations_status.setWordWrap(True)
        self.lbl_relations_status.setStyleSheet("color:#94a3b8;font-size:11px;")
        self._rel_layout.addWidget(self.lbl_relations_status)
        self._rel_sections_host = QVBoxLayout()
        self._rel_layout.addLayout(self._rel_sections_host)
        self.grp_relations.hide()
        prev_l.addWidget(self.grp_relations)

        self.txt_teach_tag = QLineEdit()
        self.txt_teach_tag.hide()
        self.lbl_teach_tags = QLabel("")
        self.lbl_teach_tags.hide()
        prev_l.addStretch()

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.preview_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.preview_scroll.setWidget(self.tab_preview)
        self.tabs.addTab(self.preview_scroll, "Önizleme")
        self._tab_scores = _scroll_text("Sonuç seçin…")
        self.lbl_pattern_family = QLabel("—")
        self.lbl_animal_type = QLabel("—")
        self.lbl_color_family = QLabel("—")
        self.lbl_cluster = QLabel("—")
        self.lbl_blob = QLabel("—")
        self.lbl_stripe = QLabel("—")
        self.lbl_scale = QLabel("—")
        self.lbl_density = QLabel("—")
        self.lbl_reason = QLabel("—")
        self._tab_texture = _scroll_form(
            [
                ("Pattern family", self.lbl_pattern_family),
                ("Animal print type", self.lbl_animal_type),
                ("Color family", self.lbl_color_family),
                ("Küme grubu", self.lbl_cluster),
                ("Blob score", self.lbl_blob),
                ("Stripe score", self.lbl_stripe),
                ("Scale score", self.lbl_scale),
                ("Texture density", self.lbl_density),
                ("Neden bu gruba girdi?", self.lbl_reason),
            ]
        )
        self._tab_file = _scroll_text("Sonuç seçin…")
        self._tab_source = _scroll_text("Sonuç seçin…")
        self.tabs.addTab(self._tab_scores, "Skor Detayı")
        self.tabs.addTab(self._tab_texture, "Doku Haritası")
        self.tabs.addTab(self._tab_file, "Dosya Bilgisi")
        self.tabs.addTab(self._tab_source, "Kaynak / Kopyalar")
        layout.addWidget(self.tabs, stretch=1)

        # Her sekmede görünür — Önizleme scroll içinde kaybolmasın; wrap, yatay taşırma yok.
        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(6)
        action_grid.setVerticalSpacing(6)
        self.btn_folder = QPushButton("Klasörde Aç")
        self.btn_file = QPushButton("Görseli Aç")
        self.btn_similar = QPushButton("Benzerlerini Ara")
        self.btn_folder.setToolTip("Dosyanın bulunduğu klasörü açar")
        self.btn_file.setToolTip("Görseli varsayılan uygulamada açar")
        self.btn_folder.clicked.connect(self._emit_folder)
        self.btn_file.clicked.connect(self._emit_file)
        self.btn_similar.clicked.connect(self._emit_similar)
        for btn in (self.btn_folder, self.btn_file, self.btn_similar):
            btn.setMinimumHeight(34)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
            )
            btn.setEnabled(False)
        action_grid.addWidget(self.btn_folder, 0, 0)
        action_grid.addWidget(self.btn_file, 0, 1)
        action_grid.addWidget(self.btn_similar, 1, 0, 1, 2)
        layout.addLayout(action_grid)

        self.set_simple_mode(True)

    def set_simple_mode(self, simple: bool) -> None:
        self._simple_mode = simple
        self._ai_advanced.setVisible(not simple)
        # Designer view: hide technical DNA / keep "why" + learn
        self.grp_pattern_dna.setVisible(not simple)
        self.grp_ai_prediction.setTitle("Desen tahmini" if simple else "AI Tahmini")
        self._why_base_title = "Neden benzer" if simple else "Neden bu sonuç"
        self._sync_why_group_title()
        for i in range(1, self.tabs.count()):
            self.tabs.setTabVisible(i, not simple)

    def set_db_path(self, db_path: str) -> None:
        self._db_path = str(db_path or "")

    def set_edit_target_count(self, count: int) -> None:
        n = int(count or 0)
        if n > 1:
            self.btn_ai_edit.setText(f"Düzenle ({n})")
            self.btn_ai_edit.setToolTip(f"{n} seçili görsel toplu düzenlenir")
        else:
            self.btn_ai_edit.setText("Düzenle")
            self.btn_ai_edit.setToolTip("")

    def set_result(self, result: SearchResult | None) -> None:
        self._result = result
        if not result:
            self._clear_inspector()
            return
        self._set_result_fast(result)
        self._schedule_relations(int(result.file_id))
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._set_result_heavy(result))

    def _clear_inspector(self) -> None:
        self._detail_source_pix = None
        self.thumb.clear()
        self.lbl_preview_name.setText("Sonuç seçin…")
        self.lbl_preview_src.setText("")
        self.lbl_ai_prediction.setText("Kararsız")
        self._predictions = []
        self.lbl_ai_explanation.setText("")
        self.grp_ai_explanation.hide()
        self.grp_ai_explanation.setChecked(False)
        self._sync_why_group_title()
        self._set_tab_text(self._tab_scores, "Sonuç seçin…")
        for lbl in (
            self.lbl_pattern_family,
            self.lbl_animal_type,
            self.lbl_color_family,
            self.lbl_cluster,
            self.lbl_blob,
            self.lbl_stripe,
            self.lbl_scale,
            self.lbl_density,
            self.lbl_reason,
        ):
            lbl.setText("—")
        self._set_tab_text(self._tab_file, "Sonuç seçin…")
        self._set_tab_text(self._tab_source, "Sonuç seçin…")
        for btn in (self.btn_folder, self.btn_file, self.btn_similar):
            btn.setEnabled(False)
        self._clear_relations_ui()

    def _clear_relations_ui(self, *, bump: bool = True) -> None:
        if bump:
            self._rel_gen += 1
        while self._rel_sections_host.count():
            item = self._rel_sections_host.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            lay = item.layout()
            if lay is not None:
                while lay.count():
                    child = lay.takeAt(0)
                    cw = child.widget()
                    if cw is not None:
                        cw.deleteLater()
        self.lbl_relations_status.setText("")
        self.grp_relations.hide()

    def _schedule_relations(self, file_id: int) -> None:
        self._rel_gen += 1
        gen = self._rel_gen
        self._clear_relations_ui(bump=False)
        if not self._db_path or int(file_id or 0) <= 0:
            return
        self.grp_relations.show()
        self.lbl_relations_status.setText("İlişkiler yükleniyor…")
        # Fast meta first, then full — UI never waits on the click path
        self._start_relation_job(gen, file_id, "meta")
        from PySide6.QtCore import QTimer

        QTimer.singleShot(
            30, lambda g=gen, f=file_id: self._start_relation_job(g, f, "full")
        )

    def _start_relation_job(self, gen: int, file_id: int, stage: str) -> None:
        if gen != self._rel_gen:
            return
        job = _RelationLoadJob(gen, file_id, self._db_path, stage)
        job.signals.finished.connect(self._on_relations_loaded)
        QThreadPool.globalInstance().start(job)

    def _on_relations_loaded(self, gen: int, file_id: int, bundle: object) -> None:
        if gen != self._rel_gen:
            return
        if not self._result or int(self._result.file_id) != int(file_id):
            return
        self._render_relations(bundle)

    def _render_relations(self, bundle: object) -> None:
        from core.pattern_relations import RelationBundle

        while self._rel_sections_host.count():
            item = self._rel_sections_host.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not isinstance(bundle, RelationBundle):
            self.lbl_relations_status.setText("")
            self.grp_relations.hide()
            return
        if bundle.error and not bundle.sections:
            self.lbl_relations_status.setText("İlişki bulunamadı.")
            self.grp_relations.show()
            return
        sections = bundle.non_empty()
        if not sections:
            self.lbl_relations_status.setText("Bu desen için bağlı kayıt yok.")
            self.grp_relations.show()
            return
        self.lbl_relations_status.setText("")
        self.grp_relations.show()
        for sec in sections:
            block = QWidget()
            bl = QVBoxLayout(block)
            bl.setContentsMargins(0, 0, 0, 4)
            bl.setSpacing(4)
            title = QLabel(f"<b>{sec.label}</b> ({sec.count})")
            bl.addWidget(title)
            row = QHBoxLayout()
            row.setSpacing(6)
            for it in sec.items[:12]:
                cell = QLabel()
                cell.setFixedSize(64, 64)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setStyleSheet(
                    "background:#1e2530;border:1px solid #2a3340;border-radius:4px;"
                    "color:#94a3b8;font-size:10px;"
                )
                tip = it.filename or Path(it.path).name
                if it.customer:
                    tip = f"{tip}\n{it.customer}"
                cell.setToolTip(tip)
                cell.setCursor(Qt.CursorShape.PointingHandCursor)
                shown = False
                if it.preview_path and Path(it.preview_path).is_file():
                    pix = QPixmap(it.preview_path)
                    if not pix.isNull():
                        cell.setPixmap(
                            pix.scaled(
                                62,
                                62,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )
                        )
                        shown = True
                if not shown:
                    name = (it.filename or Path(it.path).name or "?")[:10]
                    cell.setText(name)
                fid = int(it.file_id)

                def _click(_ev=None, file_id=fid):
                    self.open_file.emit(file_id)

                cell.mousePressEvent = _click  # type: ignore[method-assign]
                row.addWidget(cell)
            row.addStretch()
            bl.addLayout(row)
            self._rel_sections_host.addWidget(block)

    def _set_result_fast(self, result: SearchResult) -> None:
        """Aninda: isim + cache/preview; hi-res/source arka planda (UI disk yok)."""
        from ui.thumbnail_scheduler import get_thumbnail_scheduler

        sched = get_thumbnail_scheduler()
        fid = int(result.file_id)
        soft_placeholder = (
            "background:#232a35;border:1px solid #2a3340;border-radius:6px;"
            "color:#7c8698;font-size:13px;"
        )
        # Instant paint from RAM caches before any worker hop.
        cached = sched.peek_detail_image(fid)
        if cached is not None and not cached.isNull():
            pix = QPixmap.fromImage(cached)
            if not pix.isNull():
                self._apply_fit_preview(pix, smooth=False)
                self.thumb.setStyleSheet(soft_placeholder)
            else:
                self._detail_source_pix = None
                self.thumb.clear()
                self.thumb.setText("···")
                self.thumb.setStyleSheet(soft_placeholder)
        else:
            self._detail_source_pix = None
            self.thumb.clear()
            self.thumb.setText("···")
            self.thumb.setStyleSheet(soft_placeholder)
        self.thumb.setToolTip("")
        self._detail_native_size = (0, 0)

        if result.thumbnail_path or result.path:
            # Hidden inspector: no FeaturePreviewCache.create storms at startup.
            if self.isVisible():
                # Candidate feature path only — worker validates; no isfile on UI thread.
                fp = ""
                try:
                    from core.preview_cache import FeaturePreviewCache

                    cache_dir = str(getattr(sched, "_cache_dir", "") or "")
                    if cache_dir and result.path:
                        fp = str(
                            FeaturePreviewCache(cache_dir).preview_path_for(
                                str(result.path)
                            )
                        )
                except Exception:
                    fp = ""

                sched.request_detail_preview(
                    fid,
                    str(result.thumbnail_path or ""),
                    source_path=str(result.path or ""),
                    filename=str(result.filename or ""),
                    feature_preview_path=fp,
                    size=1024,
                    priority=0,
                )
        else:
            self.thumb.clear()
            self.thumb.setText("!")
            self.thumb.setToolTip("FAIL\nSebep: Cache yolu bos")
            self.thumb.setStyleSheet(
                "background:#2a2228;border:1px solid #3d2a32;border-radius:6px;"
                "color:#f87171;font-size:14px;"
            )
        from ui.designer_labels import (
            brand_badge_text,
            color_badge_text,
            family_badge_for_result,
            family_badge_text,
            independent_feature_badge,
            similarity_tier_label,
        )

        fam = family_badge_for_result(result)
        extra = independent_feature_badge(result)
        color = color_badge_text(result.color_family)
        brand = brand_badge_text(result)
        tier = similarity_tier_label(result)
        chips = []
        if fam:
            chips.append(
                f"<span style='background:#0f766e;color:#fff;padding:2px 8px;"
                f"border-radius:4px;font-size:11px;'>{fam}</span>"
            )
        if extra:
            chips.append(
                f"<span style='background:#334155;color:#e2e8f0;padding:2px 8px;"
                f"border-radius:4px;font-size:11px;'>{extra}</span>"
            )
        if color:
            chips.append(
                f"<span style='background:#9a3412;color:#fff;padding:2px 8px;"
                f"border-radius:4px;font-size:11px;'>{color}</span>"
            )
        if brand:
            chips.append(
                f"<span style='background:#1e3a5f;color:#e2e8f0;padding:2px 8px;"
                f"border-radius:4px;font-size:11px;'>{brand}</span>"
            )
        for tag in result.debug.get("user_tags") or []:
            chips.append(
                f"<span style='background:#334155;color:#e2e8f0;padding:2px 8px;"
                f"border-radius:4px;font-size:11px;'>{tag}</span>"
            )
        chip_html = " ".join(chips)
        self.lbl_preview_name.setText(
            f"<b style='font-size:14px;'>{result.filename}</b><br>"
            f"<span style='color:#cbd5e1;font-weight:600;'>{tier}</span>"
            f" <span style='color:#64748b;font-size:12px;'>%{result.score_percent:.0f}</span>"
            + (f"<br>{chip_html}" if chip_html else "")
        )
        self.lbl_preview_src.setText(f"Kaynak: {result.source_name or '—'}")
        for btn in (self.btn_folder, self.btn_file, self.btn_similar):
            btn.setEnabled(True)

    def _set_result_heavy(self, result: SearchResult) -> None:
        """Arka planda: DNA, skor, texture sekmeleri."""
        if not self._result or int(self._result.file_id) != int(result.file_id):
            return
        b, d = result.breakdown, result.debug
        self._predictions = category_predictions_from_result(result)
        top_label = prediction_label(self._predictions)
        if self._simple_mode:
            confidence = (
                float(self._predictions[0].get("confidence", 0))
                if self._predictions
                else 0
            )
            self.lbl_ai_prediction.setText(
                f"{top_label} %{confidence * 100:.0f}"
                if top_label != "Kararsız"
                else "Kararsız"
            )
        else:
            self.lbl_ai_prediction.setText(
                "<br>".join(
                    f"{row['label']} %{float(row.get('confidence', 0)) * 100:.0f}"
                    for row in self._predictions
                )
                or "Kararsız"
            )
        explanation = build_result_explanation(result)
        dna = explanation.get("dna") or {}
        dna_lines = []
        for key, label in (
            ("main_family", "Main Family"),
            ("family", "Family"),
            ("subfamily", "Sub Family"),
            ("collection", "Collection"),
            ("series", "Series"),
            ("motif", "Motif"),
            ("motif_class", "Motif Class"),
            ("repeat", "Repeat"),
            ("repeat_class", "Repeat Class"),
            ("style", "Style"),
            ("designer_style", "Designer Style"),
            ("density", "Density"),
            ("complexity", "Complexity"),
            ("texture", "Texture"),
            ("texture_class", "Texture Class"),
            ("visual_signature", "Visual Signature"),
            ("color_family", "Color Style"),
        ):
            val = dna.get(key)
            if val:
                dna_lines.append(f"<b>{label}:</b> {val}")
        conf = float(dna.get("confidence") or explanation.get("confidence") or 0)
        if conf:
            dna_lines.append(f"<b>Confidence:</b> %{conf * 100:.0f}")
        tm = d.get("texture_map") or {}
        auto_tags = tm.get("auto_tags") if isinstance(tm, dict) else []
        if auto_tags:
            dna_lines.append(
                "<b>AI Etiketler:</b> " + ", ".join(str(t) for t in auto_tags)
            )
        ent_lines = d.get("entity_debug_lines") or []
        if ent_lines:
            dna_lines.append("<b>Nesne:</b> " + ", ".join(str(x) for x in ent_lines))
        tree = tm.get("pattern_family_tree") if isinstance(tm, dict) else None
        if isinstance(tree, dict) and tree.get("root"):
            from core.pattern_family_tree import format_family_tree_text

            tree_txt = format_family_tree_text(tree).replace("\n", "<br>")
            dna_lines.append(f"<b>Pattern Family:</b><br>{tree_txt}")
        dup = tm.get("duplicate_info") if isinstance(tm, dict) else None
        if isinstance(dup, dict) and dup.get("label"):
            dna_lines.append(
                f"<b>Duplicate:</b> {dup.get('label')} "
                f"(%{float(dup.get('score', 0) or 0) * 100:.0f})"
            )
        self.lbl_pattern_dna.setText("<br>".join(dna_lines) or "—")
        dna_match = d.get("dna_match_fields") or {}
        if dna_match:
            dna_lines_check = ["<b>Eşleşen DNA</b>"]
            for label in (
                "Family",
                "SubFamily",
                "Collection",
                "Series",
                "Motif",
                "Motif Class",
                "Repeat",
                "Repeat Class",
                "Style",
                "Designer Style",
                "Pattern Type",
                "Brand Style",
                "Material Hint",
                "Texture",
                "Texture Class",
                "Complexity",
                "Density",
                "Color Style",
                "Semantic",
                "Visual Signature",
                "Structure",
            ):
                dna_lines_check.append(
                    f"{'✔' if dna_match.get(label) else '✖'} {label}"
                )
            dna_check_text = "<br><br>" + "<br>".join(dna_lines_check)
        else:
            dna_check_text = ""
        family_explanations = d.get("family_explanations") or []
        family_checks = [
            ("Aynı Koleksiyon", bool(d.get("same_collection")) or bool(dna_match.get("Collection"))),
            ("Aynı Seri", bool(d.get("same_series")) or bool(dna_match.get("Series"))),
            ("Aynı Tasarım Dili", bool(d.get("same_designer_style")) or bool(dna_match.get("Designer Style"))),
            ("Aynı Family", bool(d.get("same_family")) or bool(dna_match.get("Family"))),
            ("Aynı Motif", bool(d.get("motif_match")) or bool(dna_match.get("Motif"))),
            ("Aynı Repeat", bool(d.get("repeat_match")) or bool(dna_match.get("Repeat"))),
            ("Aynı Style", bool(dna_match.get("Style")) or float(d.get("family_variant_score", 0) or 0) >= 0.60),
            ("Aynı Texture", bool(d.get("texture_match")) or bool(dna_match.get("Texture"))),
            ("Aynı Density", bool(dna_match.get("Density"))),
            ("Aynı Semantic", bool(d.get("semantic_match")) or bool(dna_match.get("Semantic"))),
            ("Aynı Pattern DNA", float(d.get("dna_score", 0) or 0) >= 0.55),
            ("Aynı Structure", bool(d.get("structure_match"))),
            ("Aynı Brand Style", bool(dna_match.get("Brand Style"))),
            ("Aynı Material Hint", bool(dna_match.get("Material Hint"))),
        ]
        why_checks = "<br><br><b>Eşleşen Özellikler</b><br>" + "<br>".join(
            f"{'✔' if ok else '✖'} {label}" for label, ok in family_checks
        )
        natural_why = ""
        if family_explanations:
            natural_why = "<br><br><b>Neden aynı aile?</b><br>" + "<br>".join(
                f"• {line}" for line in family_explanations
            )
        # Neden box: real-evidence-only helper (no fabricated similarity text).
        why_lines = format_why_lines(result)
        why_html = format_why_html(result)
        if why_lines:
            self.lbl_ai_explanation.setText(
                self._append_technical_why(
                    why_html, dna_check_text, why_checks, natural_why
                )
            )
            # Keep collapsed by default; content is below preview (no overlay).
            if not self.grp_ai_explanation.isChecked():
                self.grp_ai_explanation.setChecked(False)
            self.grp_ai_explanation.show()
            self.lbl_ai_explanation.setVisible(self.grp_ai_explanation.isChecked())
            self._sync_why_group_title()
        else:
            self.lbl_ai_explanation.setText("")
            # Hide when empty; keep collapsed for next selection with evidence.
            self.grp_ai_explanation.hide()
            self.grp_ai_explanation.setChecked(False)
            self._sync_why_group_title()

        tb = result.text_score_breakdown or {}
        pattern_family = result.pattern_family or d.get("pattern_family") or "unknown"
        contrib = d.get("contribution_scores") or {}
        engine_scores = (
            f"<b>Exact Score:</b> {float(d.get('exact_score', d.get('exact_search_score', 0)) or 0) * 100:.1f}%<br>"
            f"<b>DNA Score:</b> {float(d.get('dna_score', b.get('dna', 0)) or 0) * 100:.1f}%<br>"
            f"<b>Semantic Score:</b> {float(d.get('semantic_score', b.get('semantic', 0)) or 0) * 100:.1f}%<br>"
            f"<b>Texture Score:</b> {float(d.get('texture_score', b.get('texture', 0)) or 0) * 100:.1f}%<br>"
            f"<b>Family Score:</b> {float(d.get('family_variant_score', d.get('pattern_family_score', 0)) or 0) * 100:.1f}%<br>"
            f"<b>Final Score:</b> %{result.score_percent}<br>"
            f"<b>Katkılar:</b> "
            f"Patch +{float(contrib.get('patch', 0) or 0) * 22:.0f}, "
            f"DNA +{float(contrib.get('dna', 0) or 0) * 16:.0f}, "
            f"Semantic +{float(contrib.get('semantic', 0) or 0) * 12:.0f}, "
            f"Texture +{float(contrib.get('texture', 0) or 0) * 16:.0f}, "
            f"Repeat +{float(d.get('repeat_score', 0) or 0) * 8:.0f}, "
            f"Color +{float(d.get('color_score', 0) or 0) * 4:.0f}<br><br>"
        )
        scores_text = (
            engine_scores
            +
            f"<b>Final skor:</b> %{result.score_percent}<br><br>"
            f"<b>Metin skoru:</b> {tb.get('final_score', b.get('text_score', 0)) * 100:.1f}%<br>"
            f"Dosya adı: {tb.get('filename_score', b.get('filename_text', 0)) * 100:.1f}%<br>"
            f"Klasör: {tb.get('folder_score', 0) * 100:.1f}%<br>"
            f"Aile: {tb.get('family_score', b.get('family_text', 0)) * 100:.1f}%<br>"
            f"Doku etiketi: {tb.get('texture_score', 0) * 100:.1f}%<br>"
            f"Feedback: {tb.get('feedback_score', 0) * 100:.1f}%<br>"
            f"Pattern group: {tb.get('group_score', 0) * 100:.1f}%<br>"
            f"OCR: {tb.get('ocr_score', b.get('ocr_text', 0)) * 100:.1f}%<br><br>"
            f"pHash: {b.get('phash', 0) * 100:.1f}%<br>"
            f"dHash: {b.get('dhash', 0) * 100:.1f}%<br>"
            f"wHash: {b.get('whash', 0) * 100:.1f}%<br>"
            f"Patch: {b.get('patch', d.get('patch_score', 0)) * 100:.1f}%<br>"
            f"Texture: {b.get('texture', 0) * 100:.1f}%<br>"
            f"Renk: {b.get('color', 0) * 100:.1f}%<br>"
            f"AI/DINO: {b.get('dino', 0) * 100:.1f}%<br>"
            f"AI/CLIP: {b.get('clip', 0) * 100:.1f}%<br>"
        )
        uvi_ex = (result.debug or {}).get("uvi_explain") or {}
        if uvi_ex:
            scores_text += (
                f"<br><b>UVI ranking</b><br>"
                f"Object Match: {uvi_ex.get('object_match', 'None')}<br>"
                f"Object Evidence: {uvi_ex.get('object_evidence', '')}<br>"
                f"Pattern Match: {uvi_ex.get('pattern_match', 0)}<br>"
                f"CLIP: {uvi_ex.get('clip', 0)}<br>"
                f"UVI: {uvi_ex.get('uvi', 0)}<br>"
                f"Semantic: {uvi_ex.get('semantic', 0)}<br>"
                f"Tier: {uvi_ex.get('tier', '')}<br>"
            )
            if uvi_ex.get("penalty"):
                scores_text += f"Penalty: {uvi_ex.get('penalty')}<br>"
        qev = d.get("query_evidence_report") or tb.get("query_evidence_report") or {}
        if qev:
            verdict = str(d.get("visual_verdict") or qev.get("visual_verdict") or "")
            scores_text += "<br><b>Query Evidence</b><br>"
            if verdict:
                scores_text += f"<b>Görsel:</b> {verdict}<br>"
            grade = str(d.get("visual_grade") or qev.get("visual_grade") or "")
            if grade:
                scores_text += f"visual_grade: {grade}"
                if qev.get("visual_rival"):
                    scores_text += f" · rival={qev.get('visual_rival')}"
                scores_text += "<br>"
            for cid, ev in (qev.get("concepts") or {}).items():
                state = str(ev.get("state") or "?")
                mark = "✓" if state == "SUPPORTED" else ("✗" if state == "CONTRADICTED" else "?")
                ch = ev.get("channels") or {}
                chs = " ".join(f"{k}={float(v):.2f}" for k, v in ch.items()) or "—"
                scores_text += (
                    f"{mark} {cid} {state} {float(ev.get('score') or 0):.2f} [{chs}]<br>"
                )
            scores_text += (
                f"COMPOSITE: {qev.get('composite') or '—'} · "
                f"CONFIDENCE: {qev.get('confidence') or '—'}<br>"
                f"Δ {float(qev.get('delta') or 0):+.3f} "
                f"(q={float(qev.get('query_evidence') or 0):+.3f} "
                f"comp={float(qev.get('composite_evidence') or 0):+.3f} "
                f"pen={float(qev.get('contradiction_penalty') or 0):.3f})<br>"
            )
            if qev.get("conflict"):
                scores_text += "CONFLICT: true<br>"
            scores_text += (
                f"visual_object={float(qev.get('visual_object') or 0):.2f} "
                f"visual_attribute={float(qev.get('visual_attribute') or 0):.2f} "
                f"visual_similarity={float(qev.get('visual_similarity') or 0):.2f} "
                f"visual_family={float(qev.get('visual_family') or 0):.2f}"
            )
            if qev.get("visual_conflict"):
                scores_text += " · visual_conflict=true"
            scores_text += "<br>"
        scores_text += (
            f"<br><br><b>Family:</b> {d.get('query_family', pattern_family)} "
            f"({d.get('query_confidence', 0) * 100:.0f}%)<br>"
            f"<b>Subtype:</b> {d.get('query_subtype', '—')}<br>"
            f"<b>Sonuç family:</b> {d.get('result_family', pattern_family)} "
            f"({d.get('result_confidence', 0) * 100:.0f}%)<br>"
            f"Animal: {d.get('animal_score', 0) * 100:.0f}% | "
            f"Floral: {d.get('floral_score', 0) * 100:.0f}% | "
            f"Marble: {d.get('marble_score', 0) * 100:.0f}%<br>"
        )
        if d.get("why_animal_rejected"):
            scores_text += f"<b>Animal red:</b> {d.get('why_animal_rejected')}<br>"
        if d.get("why_family_selected"):
            scores_text += f"<b>Family seçimi:</b> {d.get('why_family_selected')}<br>"
        if d.get("reject_reason"):
            scores_text += f"<b>Reject Reason:</b> {d.get('reject_reason')}<br>"
        if self._family_reject_report:
            scores_text += "<br><b>Aynı family — elenen adaylar</b><br>"
            for row in self._family_reject_report[:20]:
                scores_text += (
                    f"• {row.get('filename', '')} | "
                    f"Final {float(row.get('final_score', 0) or 0) * 100:.1f}% | "
                    f"Exact {float(row.get('exact_score', 0) or 0) * 100:.1f}% | "
                    f"DNA {float(row.get('dna_score', 0) or 0) * 100:.1f}% | "
                    f"Semantic {float(row.get('semantic_score', 0) or 0) * 100:.1f}% | "
                    f"Texture {float(row.get('texture_score', 0) or 0) * 100:.1f}% | "
                    f"Family {float(row.get('family_score', 0) or 0) * 100:.1f}% | "
                    f"<i>{row.get('reject_reason', '')}</i><br>"
                )
        self._set_tab_text(self._tab_scores, scores_text)

        animal_type = result.animal_print_type or d.get("animal_print_type") or "—"
        color_family = result.color_family or d.get("color_family") or "unknown"
        cluster = result.cluster_group or d.get("cluster_group") or "—"
        blob = float(d.get("organic_blob_score", 0) or d.get("texture_score", 0) or 0)
        stripe = float(d.get("stripe_score", 0) or 0)
        scale = float(d.get("scale_score", 0) or 0)
        density = result.texture_family_score or float(
            d.get("texture_family_score", 0) or 0
        )
        reason = result.cluster_reason or d.get("matched_reason") or "—"

        self.lbl_pattern_family.setText(str(pattern_family))
        self.lbl_animal_type.setText(str(animal_type) if animal_type else "—")
        self.lbl_color_family.setText(str(color_family))
        self.lbl_cluster.setText(f"{cluster} ({result.cluster_label or cluster})")
        self.lbl_blob.setText(f"%{blob * 100:.0f}")
        self.lbl_stripe.setText(f"%{stripe * 100:.0f}")
        self.lbl_scale.setText(f"%{scale * 100:.0f}")
        self.lbl_density.setText(f"%{density * 100:.0f}")
        self.lbl_reason.setText(str(reason).replace("- ", "• "))

        mtime = ""
        if result.mtime:
            mtime = datetime.fromtimestamp(result.mtime).strftime("%d.%m.%Y %H:%M")
        file_text = (
            f"<b>Tam yol:</b><br>{result.path}<br><br>"
            f"Boyut: {result.width}×{result.height}<br>"
            f"Dosya: {format_file_size(result.file_size)}<br>"
            f"Format: {Path(result.path).suffix}<br>"
            f"Tarih: {mtime}<br>"
            f"Reader: {d.get('reader_used', 'pillow')}<br>"
            f"Thumbnail: {'var' if d.get('has_thumbnail') else 'yok'}<br>"
            f"Feature: {'var' if d.get('has_features') else 'yok'}"
        )
        self._set_tab_text(self._tab_file, file_text)

        source_text = (
            f"Kaynak: {result.source_name or '—'}<br>"
            f"Tür: {result.source_type or '—'}<br>"
            f"Müşteri: {result.customer or '—'}<br>"
            f"Klasör: {Path(result.path).parent}<br><br>"
            f"<i>Aynı desen farklı konumlarda — tam kopya listesi için arama raporuna bakın.</i>"
        )
        self._set_tab_text(self._tab_source, source_text)

    @staticmethod
    def _set_tab_text(scroll: QScrollArea, html: str) -> None:
        lbl = scroll.widget().findChild(QLabel)
        if lbl:
            lbl.setText(html)

    def _why_title_base(self) -> str:
        return getattr(self, "_why_base_title", None) or "Neden bu sonuç"

    def _sync_why_group_title(self) -> None:
        base = self._why_title_base()
        open_ = bool(self.grp_ai_explanation.isChecked())
        mark = "▾" if open_ else "▸"
        self.grp_ai_explanation.setTitle(f"{base} {mark}")

    def _on_why_toggled(self, checked: bool) -> None:
        self._sync_why_group_title()
        # Content sits in the group layout below preview — never overlay.
        self.lbl_ai_explanation.setVisible(bool(checked))

    def _emit_folder(self) -> None:
        if self._result:
            self.open_folder.emit(self._result.file_id)

    @staticmethod
    def _build_similarity_explanation_html(
        result: SearchResult,
        d: dict,
        dna_check_text: str,
        why_checks: str,
        natural_why: str,
        explanation: dict,
    ) -> str:
        """Neden box content — real evidence only (no fabricated fallbacks)."""
        # d / dna_check / why_checks / explanation kept for call-site compat;
        # invented chip/format_explanation_text path removed.
        _ = (d, dna_check_text, why_checks, natural_why, explanation)
        return format_why_html(result)

    def _append_technical_why(
        self,
        designer_html: str,
        dna_check_text: str,
        why_checks: str,
        natural_why: str,
    ) -> str:
        if self._simple_mode:
            return designer_html
        return designer_html + dna_check_text + why_checks + natural_why

    def _emit_file(self) -> None:
        if self._result:
            self.open_file.emit(self._result.file_id)

    def _emit_similar(self) -> None:
        if self._result:
            self.search_similar.emit(self._result.file_id)

    def _emit_teach_tag(self) -> None:
        tag = self.txt_teach_tag.text().strip()
        if tag:
            self.teach_tag_clicked.emit(tag)

    def set_learned_tags(self, tags: list[str]) -> None:
        if tags:
            self.lbl_teach_tags.setText("Öğrenilen etiketler: " + ", ".join(tags))
        else:
            self.lbl_teach_tags.setText("")

    def _top_prediction_path(self) -> str:
        rows = getattr(self, "_predictions", [])
        return str(rows[0].get("category_path", "")) if rows else ""

    def _prediction_scope(self) -> str:
        if self._simple_mode:
            return "single"
        return str(self.cmb_ai_scope.currentData() or "single")

    def set_family_reject_report(self, rows: list[dict] | None) -> None:
        self._family_reject_report = list(rows or [])
        if self._result:
            self.set_result(self._result)

    def _on_scheduler_thumb_failed(self, file_id: int, filename: str, reason: str) -> None:
        if not self._result or int(self._result.file_id) != int(file_id):
            return
        self.thumb.clear()
        self.thumb.setText("!")
        self.thumb.setToolTip(
            f"FAIL\nDosya: {filename or self._result.filename}\nSebep: ☒ {reason}"
        )

    def _on_scheduler_thumb(self, file_id: int, image, size: int) -> None:
        if not self._result or int(self._result.file_id) != int(file_id):
            return
        # Sol kart küçük thumb sinyali (size<=120) detail'i ezmesin
        if int(size or 0) < 768:
            return
        if image is None:
            self.thumb.setText("!")
            self.thumb.setToolTip(
                f"FAIL\nDosya: {self._result.filename}\nSebep: ☒ Decode başarısız"
            )
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            self.thumb.setText("!")
            self.thumb.setToolTip(
                f"FAIL\nDosya: {self._result.filename}\nSebep: ☒ Decode başarısız"
            )
            return
        nw, nh = int(pix.width()), int(pix.height())
        self._detail_native_size = (nw, nh)
        meta = ""
        try:
            from ui.thumbnail_scheduler import get_thumbnail_scheduler

            meta = get_thumbnail_scheduler().peek_detail_meta(int(file_id))
        except Exception:
            meta = {}
        path = str((meta or {}).get("path") or "")
        self.thumb.setToolTip(
            f"{self._result.filename}\n"
            f"Detail preview: {nw}×{nh}px"
            + (f"\n{path}" if path else "")
        )
        tw, th = self._apply_fit_preview(pix, smooth=True)
        logger_msg = f"RESULT_DETAIL file_id={file_id} display={tw}x{th} native={nw}x{nh}"
        try:
            from core.logger import setup_logger

            setup_logger(__name__).info(logger_msg)
        except Exception:
            pass

    def _preview_avail_width(self) -> int:
        """Usable width inside the preview scroll viewport (never expands the panel)."""
        widths: list[int] = []
        try:
            vp = self.preview_scroll.viewport()
            if vp is not None and int(vp.width()) > 1:
                widths.append(int(vp.width()))
        except Exception:
            pass
        # Walk parents: dock QScrollArea viewport is the hard horizontal bound.
        try:
            p = self.parent()
            while p is not None:
                if isinstance(p, QScrollArea):
                    pvp = p.viewport()
                    if pvp is not None and int(pvp.width()) > 1:
                        widths.append(int(pvp.width()))
                p = p.parent()
        except Exception:
            pass
        for w in (
            int(self.tab_preview.width() or 0),
            int(self.thumb.width() or 0),
            int(self.width() or 0),
        ):
            if w > 1:
                widths.append(w)
        if not widths:
            return 1
        return max(1, min(widths) - 8)

    def _preview_avail_height(self) -> int:
        h = int(self.thumb.height() or 0)
        if h <= 1:
            h = 360
        return max(120, min(520, h))

    def _constrain_text_widths(self) -> None:
        """Cap label max width so long filenames cannot force horizontal overflow."""
        w = max(40, self._preview_avail_width())
        for lbl in (
            self.lbl_preview_name,
            self.lbl_preview_src,
            self.lbl_ai_prediction,
            self.lbl_pattern_dna,
            self.lbl_ai_explanation,
            self.lbl_relations_status,
        ):
            lbl.setMaximumWidth(w)

    def _apply_fit_preview(
        self, pix: QPixmap, *, smooth: bool = True
    ) -> tuple[int, int]:
        """Contain-fit preview into available panel width; never force panel wider."""
        if pix.isNull():
            return (0, 0)
        self._constrain_text_widths()
        self._detail_source_pix = QPixmap(pix)
        nw, nh = int(pix.width()), int(pix.height())
        box_w = self._preview_avail_width()
        box_h = self._preview_avail_height()
        # Do not upscale above native; do not use floors that expand the layout.
        scale = min(1.0, box_w / max(nw, 1), box_h / max(nh, 1))
        tw = max(1, int(nw * scale))
        th = max(1, int(nh * scale))
        mode = (
            Qt.TransformationMode.SmoothTransformation
            if smooth
            else Qt.TransformationMode.FastTransformation
        )
        self.thumb.setText("")
        self.thumb.setPixmap(
            pix.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio, mode)
        )
        return (tw, th)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._constrain_text_widths()
        if self._detail_source_pix is not None and not self._detail_source_pix.isNull():
            self._apply_fit_preview(self._detail_source_pix, smooth=True)
