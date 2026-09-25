"""Sonuç listesi — dinamik gruplar, görünüm modları."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.dynamic_groups import (
    G_COLOR,
    G_CROP,
    G_EXACT,
    G_FAR,
    G_FORMAT,
    G_RELATED,
    G_RESOLUTION,
    G_SAME_CLOSE,
    G_SAME_STYLE,
    G_UNRELATED,
    QueryContext,
    group_checked_by_default,
    group_definitions,
    groups_relevant_for_query,
    normalize_cluster_key,
)
from core.display_match_percent import (
    merge_results_in_engine_order,
    stamp_display_match_percent,
)
from core.search_engine import SearchResult
from ui.result_card import VIEW_CARD, VIEW_COMPACT, VIEW_LARGE, VIEW_LIST, ResultCard
from ui.thumbnail_scheduler import ThumbnailScheduler
from ui.preview_selection import (
    PreviewSelectionStore,
    build_copy_plan,
    build_original_copy_plan,
    copy_originals_to_clipboard,
    copy_previews_to_clipboard,
)
from ui.virtual_results_list import VirtualResultsList

RANKED_VIEW = VIEW_LIST
INITIAL_RENDER_COUNT = 50
INFINITE_PAGE_SIZE = 50

GROUP_ALL = "all"
GROUP_FAMILY = "family"
GROUP_SAME = "same_pattern"
GROUP_SOURCE = "source"

GROUP_TIER_LABELS = {
    "exact": "Aynı desen",
    "near_variant": "Yakın varyant",
    "same_family": "Aynı aile",
    "similar_motif": "Benzer motif",
    "similar_texture": "Benzer doku",
    "similar_color": "Benzer renk",
}

RESULT_LAYER_LABELS = {
    "same_person": "Aynı Kişi",
    "same_files": "Aynı Dosyalar",
    "same_pattern_family": "Aynı Desen Ailesi",
    "similar_patterns": "Benzer Desenler",
    "other_results": "Diğer Sonuçlar",
}
RESULT_LAYER_ORDER = {
    "same_person": 0,
    "same_files": 1,
    "same_pattern_family": 2,
    "similar_patterns": 3,
    "other_results": 4,
}

GROUP_LABELS = {
    GROUP_ALL: "Tüm sonuçlar",
    GROUP_FAMILY: "Aileye göre grupla",
    GROUP_SAME: "Aynı desenleri grupla",
    GROUP_SOURCE: "Kaynağa göre grupla",
}


class ResultsPanel(QWidget):
    result_selected = Signal(object)
    open_folder = Signal(int)
    open_file = Signal(int)
    load_more_clicked = Signal()
    group_filter_changed = Signal()
    view_mode_changed = Signal(str)
    hover_preview = Signal(int, str, str, str)  # file_id, path, filename, source
    hover_preview_clear = Signal()
    preview_selection_changed = Signal(int)
    edit_selected_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._group_checks: dict[str, QCheckBox] = {}
        self._view_mode = VIEW_LARGE
        self._query_ctx = QueryContext()
        self._simple_mode = True
        self._group_mode = GROUP_ALL
        self._searching = False
        self._selected_file_id = 0
        self._search_activity_lbl: QLabel | None = None
        self._thumb_fail_reasons: dict[int, str] = {}
        self._virtual_ranked_results: list[SearchResult] = []
        self._virtual_rendered = 0
        self._virtual_cursor = 0
        self._full_results: list[SearchResult] = []
        self._page_limit = 0
        self._page_size = 500
        self._virtual_entries: list[tuple] = []
        self._ranked = False
        self._score_floor = 0.0
        self._below_threshold: list[SearchResult] = []
        self._total_above = 0
        self._thumb_scheduler: ThumbnailScheduler | None = None
        self._cards_by_id: dict[int, ResultCard] = {}
        self._preview_sel = PreviewSelectionStore()
        self._last_thumb_expected_ids: tuple[int, ...] = ()
        self._last_visible_thumb_ids: tuple[int, ...] = ()
        self._thumb_schedule_timer = QTimer(self)
        self._thumb_schedule_timer.setSingleShot(True)
        self._thumb_schedule_timer.setInterval(40)
        self._thumb_schedule_timer.timeout.connect(
            self._schedule_visible_thumbnails_now
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        group = QGroupBox("Sonuçlar")
        group.setStyleSheet(
            "QGroupBox { font-weight:600; color:#94a3b8; border:1px solid #252b36; }"
        )
        layout.addWidget(group, stretch=1)
        inner = QVBoxLayout(group)

        toolbar = QHBoxLayout()
        self.lbl_group = QLabel("Grupla:")
        self.cmb_group = QComboBox()
        self.cmb_group.setMinimumWidth(200)
        for key, label in GROUP_LABELS.items():
            self.cmb_group.addItem(label, key)
        self.cmb_group.currentIndexChanged.connect(self._on_group_mode_changed)
        toolbar.addWidget(self.lbl_group)
        toolbar.addWidget(self.cmb_group)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Görünüm:"))
        self.cmb_view = QComboBox()
        self.cmb_view.setMinimumWidth(120)
        self.cmb_view.addItem("Kart", VIEW_CARD)
        self.cmb_view.addItem("Liste", VIEW_LIST)
        self.cmb_view.addItem("Kompakt", VIEW_COMPACT)
        self.cmb_view.addItem("Büyük thumbnail", VIEW_LARGE)
        idx_large = self.cmb_view.findData(VIEW_LARGE)
        if idx_large >= 0:
            self.cmb_view.setCurrentIndex(idx_large)
        self.cmb_view.currentIndexChanged.connect(self._on_view_changed)
        toolbar.addWidget(self.cmb_view)
        inner.addLayout(toolbar)

        self.lbl_search_activity = QLabel("")
        self.lbl_search_activity.setWordWrap(True)
        self.lbl_search_activity.setStyleSheet(
            "color:#fbbf24;font-weight:600;padding:8px 10px;"
            "background:#2a2418;border:1px solid #4a3f1f;border-radius:6px;"
        )
        self.lbl_search_activity.setVisible(False)
        inner.addWidget(self.lbl_search_activity)

        # Thumbnail Durumu — sabit/küçük yükseklik; FAIL detayları scrollable
        self.thumb_health_box = QFrame()
        self.thumb_health_box.setObjectName("thumb_health_box")
        self.thumb_health_box.setStyleSheet(
            "QFrame#thumb_health_box{background:#0f172a;border:1px solid #334155;"
            "border-radius:4px;}"
            "QLabel{color:#cbd5e1;font-size:11px;background:transparent;border:none;}"
            "QToolButton{color:#94a3b8;font-size:11px;background:transparent;"
            "border:none;text-align:left;padding:0;}"
            "QToolButton:hover{color:#e2e8f0;}"
        )
        self.thumb_health_box.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.thumb_health_box.setVisible(False)
        th_lay = QVBoxLayout(self.thumb_health_box)
        th_lay.setContentsMargins(8, 4, 8, 4)
        th_lay.setSpacing(2)
        self.lbl_thumb_health = QLabel("")
        self.lbl_thumb_health.setWordWrap(False)
        self.lbl_thumb_health.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        th_lay.addWidget(self.lbl_thumb_health)
        self.btn_thumb_fails = QToolButton()
        self.btn_thumb_fails.setText("FAIL örnekleri ▸")
        self.btn_thumb_fails.setCheckable(True)
        self.btn_thumb_fails.setChecked(False)
        self.btn_thumb_fails.setVisible(False)
        self.btn_thumb_fails.toggled.connect(self._on_thumb_fails_toggled)
        th_lay.addWidget(self.btn_thumb_fails)
        self.txt_thumb_fails = QTextEdit()
        self.txt_thumb_fails.setReadOnly(True)
        self.txt_thumb_fails.setAcceptRichText(False)
        self.txt_thumb_fails.setMaximumHeight(72)
        self.txt_thumb_fails.setMinimumHeight(0)
        self.txt_thumb_fails.setVisible(False)
        self.txt_thumb_fails.setStyleSheet(
            "QTextEdit{color:#94a3b8;font-size:10px;background:#020617;"
            "border:1px solid #1e293b;border-radius:3px;padding:2px 4px;}"
        )
        self.txt_thumb_fails.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        th_lay.addWidget(self.txt_thumb_fails)
        self.thumb_health_box.setMaximumHeight(110)
        inner.addWidget(self.thumb_health_box)

        self.lbl_ai_health = QLabel("")
        self.lbl_ai_health.setWordWrap(False)
        self.lbl_ai_health.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.lbl_ai_health.setStyleSheet(
            "color:#cbd5e1;font-size:11px;background:#0f172a;"
            "border:1px solid #334155;border-radius:4px;padding:4px 8px;"
        )
        self.lbl_ai_health.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.lbl_ai_health.setMaximumHeight(28)
        self.lbl_ai_health.setVisible(False)
        inner.addWidget(self.lbl_ai_health)

        self.grp_scroll = QScrollArea()
        self.grp_scroll.setWidgetResizable(True)
        self.grp_scroll.setMaximumHeight(160)
        self.grp_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.grp_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.grp_inner = QWidget()
        self.grp_grid = QGridLayout(self.grp_inner)
        self.grp_grid.setContentsMargins(4, 2, 4, 2)
        self.grp_grid.setHorizontalSpacing(10)
        self.grp_grid.setVerticalSpacing(6)
        self.grp_scroll.setWidget(self.grp_inner)
        inner.addWidget(self.grp_scroll)

        self.scroll = VirtualResultsList()
        self.scroll.card_selected.connect(self._select_result)
        self.scroll.open_folder.connect(self.open_folder.emit)
        self.scroll.open_file.connect(self.open_file.emit)
        self.scroll.hover_preview.connect(self.hover_preview.emit)
        self.scroll.hover_preview_clear.connect(self.hover_preview_clear.emit)
        self.scroll.near_bottom.connect(self._on_near_bottom)
        self.scroll.scrolled.connect(self._schedule_visible_thumbnails)
        self.scroll.set_preview_selected_fn(self._preview_sel.is_selected)
        self.scroll.preview_menu_requested.connect(self._show_preview_context_menu)
        self.scroll.preview_click.connect(self._on_preview_card_click)
        self.scroll.preview_checkbox.connect(self._on_preview_checkbox)
        inner.addWidget(self.scroll, stretch=1)
        # Geriye uyumluluk
        self.container = self.scroll

        bottom = QHBoxLayout()
        self.btn_load_more = QPushButton("Daha fazla yükle")
        self.btn_load_more.clicked.connect(self.load_more_clicked.emit)
        self.btn_load_more.setVisible(False)
        self.lbl_page = QLabel("")
        self.lbl_page.setStyleSheet("color: #94a3b8;")
        self.lbl_preview_count = QLabel("")
        self.lbl_preview_count.setObjectName("lbl_preview_count")
        self.lbl_preview_count.setStyleSheet("color: #cbd5e1;font-size:11px;")
        self.lbl_preview_count.setVisible(False)
        bottom.addWidget(self.btn_load_more)
        bottom.addWidget(self.lbl_page)
        bottom.addStretch()
        bottom.addWidget(self.lbl_preview_count)
        inner.addLayout(bottom)

        self._rebuild_group_filters()

    def begin_new_search(self) -> None:
        self._preview_sel.clear()
        self._refresh_preview_checks()

    def _show_preview_context_menu(self, result, global_pos) -> None:
        from PySide6.QtWidgets import QMenu

        from ui.preview_selection import preview_context_menu_spec

        if result is None:
            return
        fid = int(getattr(result, "file_id", 0) or 0)
        if fid <= 0:
            return
        menu = QMenu(self)
        acts = {}
        for aid, label in preview_context_menu_spec(
            this_selected=self._preview_sel.is_selected(fid),
            count=self._preview_sel.count(),
        ):
            act = menu.addAction(label)
            if aid == "count":
                act.setEnabled(False)
            acts[aid] = act
        chosen = menu.exec(global_pos)
        if chosen is acts.get("toggle"):
            self._preview_sel.toggle(fid)
            self._refresh_preview_checks()
        elif chosen is acts.get("visible"):
            self._select_visible_previews()
        elif chosen is acts.get("all"):
            self._select_all_previews()
        elif chosen is acts.get("clear"):
            self._clear_preview_selection()
        elif chosen is acts.get("copy"):
            self._copy_selected_previews()
        elif chosen is acts.get("copy_original"):
            ids = self._preview_sel.ids()
            if not ids:
                ids = {int(fid)}
            self._copy_selected_originals(ids)
        elif chosen is acts.get("edit"):
            self.edit_selected_requested.emit()

    def _on_preview_card_click(self, result, modifiers) -> None:
        fid = int(getattr(result, "file_id", 0) or 0)
        if fid <= 0:
            return
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        self._preview_sel.apply_card_click(
            fid,
            self._all_result_ids(),
            ctrl=ctrl,
            shift=shift,
        )
        self._refresh_preview_checks()

    def _on_preview_checkbox(self, result, checked: bool) -> None:
        fid = int(getattr(result, "file_id", 0) or 0)
        if fid <= 0:
            return
        self._preview_sel.apply_checkbox(fid, bool(checked))
        self._refresh_preview_checks()

    def _refresh_preview_checks(self) -> None:
        for card in self.scroll.visible_cards():
            card.set_preview_checked(self._preview_sel.is_selected(card.result.file_id))
        n = self._preview_sel.count()
        if getattr(self, "lbl_preview_count", None) is None:
            return
        self.lbl_preview_count.setText(f"Seçilen: {n}")
        self.lbl_preview_count.setVisible(n > 0)
        self.preview_selection_changed.emit(n)

    def _visible_result_ids(self) -> list[int]:
        ids: list[int] = []
        seen: set[int] = set()
        for card in self.scroll.visible_cards():
            fid = int(card.result.file_id)
            if fid > 0 and fid not in seen:
                seen.add(fid)
                ids.append(fid)
        return ids

    def _all_result_ids(self) -> list[int]:
        ids: list[int] = []
        seen: set[int] = set()
        for rec in self._full_results:
            fid = int(getattr(rec, "file_id", 0) or 0)
            if fid > 0 and fid not in seen:
                seen.add(fid)
                ids.append(fid)
        return ids

    def _select_visible_previews(self) -> None:
        self._preview_sel.select_many(self._visible_result_ids())
        self._refresh_preview_checks()

    def _select_all_previews(self) -> None:
        self._preview_sel.select_many(self._all_result_ids())
        self._refresh_preview_checks()

    def selected_preview_ids(self) -> list[int]:
        wanted = self._preview_sel.ids()
        if not wanted:
            return []
        return [fid for fid in self._all_result_ids() if fid in wanted]

    def _clear_preview_selection(self) -> None:
        self._preview_sel.clear()
        self._refresh_preview_checks()

    def _copy_selected_previews(self) -> None:
        from PySide6.QtWidgets import QApplication, QMessageBox

        if self._preview_sel.count() <= 0:
            QMessageBox.information(
                self, "Önizleme", "Kopyalamak için önce desen seçin."
            )
            return
        plan = build_copy_plan(self._full_results, self._preview_sel.ids())
        clip = QApplication.clipboard()
        msg = copy_previews_to_clipboard(plan, clip)
        if not plan.paths:
            QMessageBox.warning(self, "Önizleme", msg)
            return
        QMessageBox.information(self, "Önizleme", msg)

    def _copy_selected_originals(self, ids) -> None:
        from PySide6.QtWidgets import QApplication, QMessageBox

        if not ids:
            QMessageBox.information(
                self, "Orijinal desen", "Kopyalamak için önce desen seçin."
            )
            return
        plan = build_original_copy_plan(self._full_results, ids)
        clip = QApplication.clipboard()
        msg = copy_originals_to_clipboard(plan, clip)
        if not plan.paths:
            QMessageBox.warning(self, "Orijinal desen", msg)
            return
        QMessageBox.information(self, "Orijinal desen", msg)

    def set_simple_mode(self, simple: bool) -> None:
        self._simple_mode = simple
        self.grp_scroll.setVisible(not simple)
        self.lbl_group.setVisible(simple)
        self.cmb_group.setVisible(simple)
        self.cmb_view.setVisible(not simple)

    def _on_group_mode_changed(self) -> None:
        self._group_mode = str(self.cmb_group.currentData() or GROUP_ALL)
        self.group_filter_changed.emit()

    def _group_results_simple(
        self, results: list[SearchResult]
    ) -> list[tuple[str, list[SearchResult]]]:
        mode = self._group_mode
        if mode == GROUP_ALL:
            return [("", results)]
        buckets: dict[str, list[SearchResult]] = {}
        for r in results:
            if mode == GROUP_FAMILY:
                from core.group_gates import candidate_family_display_label

                key = candidate_family_display_label(
                    type(
                        "M",
                        (),
                        {
                            "pattern_family": r.pattern_family,
                            "pattern_subtype": r.debug.get("result_subtype", ""),
                            "animal_print_type": r.animal_print_type,
                            "classification_confidence": r.debug.get(
                                "result_confidence", 0
                            ),
                        },
                    )()
                )
            elif mode == GROUP_SOURCE:
                key = r.source_name or "Bilinmeyen kaynak"
            elif mode == GROUP_SAME:
                tier = str(
                    (r.debug.get("texture_map") or {}).get("user_similarity_tier")
                    or r.debug.get("similarity_tier")
                    or ""
                )
                cluster = normalize_cluster_key(r.cluster_group or r.category or "")
                if tier in GROUP_TIER_LABELS:
                    key = GROUP_TIER_LABELS[tier]
                elif cluster in (G_EXACT, G_SAME_CLOSE):
                    key = GROUP_TIER_LABELS["exact"]
                elif cluster in (G_SAME_STYLE, G_RELATED):
                    key = GROUP_TIER_LABELS["same_family"]
                elif cluster == G_COLOR:
                    key = GROUP_TIER_LABELS["similar_color"]
                else:
                    key = GROUP_TIER_LABELS.get("similar_texture", "Benzer doku")
            else:
                key = r.cluster_group or r.pattern_family or r.category or "Diğer"
            buckets.setdefault(key, []).append(r)
        return sorted(
            [(f"{k} — {len(v)} sonuç", v) for k, v in buckets.items()],
            key=lambda x: -len(x[1]),
        )

    @staticmethod
    def _result_layer_key(result: SearchResult) -> str:
        layer = str((result.debug or {}).get("result_layer") or "").strip()
        if layer in RESULT_LAYER_LABELS:
            return layer
        cluster = normalize_cluster_key(result.cluster_group or result.category or "")
        if result.is_self_match or (result.debug or {}).get("protected_exact") or cluster in (
            G_EXACT,
            G_FORMAT,
            G_RESOLUTION,
            G_CROP,
        ):
            return "same_files"
        if cluster in (G_COLOR, G_SAME_CLOSE):
            return "same_pattern_family"
        if cluster in (G_SAME_STYLE, G_RELATED):
            return "similar_patterns"
        return "other_results"

    def _group_results_by_layer(
        self, results: list[SearchResult]
    ) -> list[tuple[str, list[SearchResult]]]:
        buckets: dict[str, list[SearchResult]] = {}
        for result in results:
            buckets.setdefault(self._result_layer_key(result), []).append(result)
        ordered: list[tuple[str, list[SearchResult]]] = []
        for key, label in sorted(
            RESULT_LAYER_LABELS.items(),
            key=lambda item: RESULT_LAYER_ORDER.get(item[0], 99),
        ):
            items = buckets.get(key, [])
            if items:
                ordered.append((f"{label}  ({len(items)})", items))
        return ordered

    def set_thumbnail_scheduler(self, scheduler: ThumbnailScheduler) -> None:
        if self._thumb_scheduler is scheduler:
            return
        if self._thumb_scheduler:
            try:
                self._thumb_scheduler.thumbnail_ready.disconnect(self._on_thumbnail_ready)
            except (RuntimeError, TypeError):
                pass
            try:
                self._thumb_scheduler.thumbnail_failed.disconnect(self._on_thumbnail_failed)
            except (RuntimeError, TypeError, AttributeError):
                pass
            try:
                self._thumb_scheduler.health_changed.disconnect(self._on_thumb_health)
            except (RuntimeError, TypeError, AttributeError):
                pass
        self._thumb_scheduler = scheduler
        self._thumb_fail_reasons: dict[int, str] = {}
        if scheduler:
            scheduler.thumbnail_ready.connect(self._on_thumbnail_ready)
            scheduler.thumbnail_failed.connect(self._on_thumbnail_failed)
            scheduler.health_changed.connect(self._on_thumb_health)

    def _on_near_bottom(self) -> None:
        if self._page_limit < len(self._full_results):
            self.append_more_pages(1)

    def _on_thumbnail_ready(self, file_id: int, image, size: int) -> None:
        applied = self._apply_thumbnail_to_visible(int(file_id), image)
        if self._thumb_scheduler and image is not None and applied:
            self._thumb_scheduler.mark_widget_updated(int(file_id), True)

    def _apply_thumbnail_to_visible(self, file_id: int, image) -> bool:
        fid = int(file_id)
        reason = self._thumb_fail_reasons.get(fid, "")
        applied = False
        for card in self.scroll.visible_cards():
            if int(card.result.file_id) == fid:
                card.apply_thumbnail(image, reason=reason)
                applied = True
        return applied

    def _apply_cached_thumbnails_to_visible(self) -> None:
        if not self._thumb_scheduler:
            return
        for card in self.scroll.visible_cards():
            if not card._thumbnail_pixmap.isNull():
                continue
            img = self._thumb_scheduler.peek_image(int(card.result.file_id))
            if img is not None and not img.isNull():
                card.apply_thumbnail(img)

    def _schedule_visible_thumbnails(self) -> None:
        if self._thumb_schedule_timer.isActive():
            return
        self._thumb_schedule_timer.start()

    def _schedule_visible_thumbnails_now(self) -> None:
        if not self._thumb_scheduler:
            return
        self._apply_cached_thumbnails_to_visible()
        visible_items: list[tuple] = []
        visible_ids: list[int] = []
        for card in self.scroll.visible_cards():
            visible_ids.append(int(card.result.file_id))
            if card._thumbnail_pixmap.isNull():
                req = card.thumbnail_request()
                if req:
                    visible_items.append(req)
        expected_ids = tuple(
            int(r.file_id) for r in self._full_results[: self._page_limit]
        )
        if expected_ids and expected_ids != self._last_thumb_expected_ids:
            self._thumb_scheduler.begin_search_batch(list(expected_ids))
            self._last_thumb_expected_ids = expected_ids
        visible_sig = tuple(visible_ids)
        if visible_sig == self._last_visible_thumb_ids and not visible_items:
            return
        self._last_visible_thumb_ids = visible_sig
        if visible_items:
            self._thumb_scheduler.request_visible(visible_items)

    def _on_thumbnail_failed(self, file_id: int, filename: str, reason: str) -> None:
        self._thumb_fail_reasons[int(file_id)] = str(reason or "")
        self._apply_thumbnail_to_visible(int(file_id), None)

    def set_pipeline_health(self, audit: dict | None = None, *, warning: str = "") -> None:
        """AI / Knowledge / DNA / FAISS sağlık paneli — tek satır özet."""
        if not audit and not warning:
            self.lbl_ai_health.setVisible(False)
            return
        tip_parts: list[str] = []
        if audit:
            comps = []
            check = audit.get("checklist") or {}
            order = (
                ("Embedding", "Embedding"),
                ("FAISS", "FAISS"),
                ("Pattern DNA", "DNA"),
                ("Knowledge Graph", "Knowledge"),
                ("Re-ranker", "Re-rank"),
                ("Family Gate", "Gate"),
            )
            for key, label in order:
                mark = check.get(key, "✗")
                comps.append(f"{label} {mark}")
            ai = "✓" if audit.get("ai_active") else "✗"
            comps.insert(0, f"AI {ai}")
            try:
                from core.face_identity import FaceIdentityEngine
                face_cap = FaceIdentityEngine.capability()
                if face_cap.get("face_recognition") == "supported":
                    comps.append("Yüz ✓")
                else:
                    comps.append("Yüz ↻ Yerel")
            except Exception:
                comps.append("Yüz ✗")
            if audit.get("provider"):
                comps.append(str(audit.get("provider")))
            stages = audit.get("stages") or []
            for st in stages:
                if st.get("count"):
                    tip_parts.append(f"{st.get('name')}: {st.get('count')}")
            body = " | ".join(comps)
            if not audit.get("ai_active") and audit.get("ai_reason"):
                reason = str(audit["ai_reason"]).splitlines()[0]
                tip_parts.insert(0, reason)
            self.lbl_ai_health.setText(f"<b>AI Pipeline</b>  {body}")
            self.lbl_ai_health.setToolTip(
                "\n".join(tip_parts) if tip_parts else body.replace(" | ", "\n")
            )
        elif warning:
            self.lbl_ai_health.setText(
                f"<b>AI Pipeline</b>  <span style='color:#f87171;'>{warning}</span>"
            )
            self.lbl_ai_health.setToolTip(str(warning))
        self.lbl_ai_health.setVisible(True)

    def _on_thumb_fails_toggled(self, checked: bool) -> None:
        self.txt_thumb_fails.setVisible(bool(checked) and bool(self.txt_thumb_fails.toPlainText()))
        self.btn_thumb_fails.setText(
            "FAIL örnekleri ▾" if checked else "FAIL örnekleri ▸"
        )
        # Kapalıyken kutu daha da kısa kalsın
        if checked and self.txt_thumb_fails.isVisible():
            self.thumb_health_box.setMaximumHeight(110)
        else:
            self.thumb_health_box.setMaximumHeight(48)

    def _on_thumb_health(self, snap) -> None:
        if not snap or int(getattr(snap, "total", 0) or 0) <= 0:
            self.thumb_health_box.setVisible(False)
            self.lbl_thumb_health.setVisible(False)
            self.btn_thumb_fails.setVisible(False)
            self.txt_thumb_fails.clear()
            self.txt_thumb_fails.setVisible(False)
            return
        err = (
            int(getattr(snap, "decode_error", 0) or 0)
            + int(getattr(snap, "cache_miss", 0) or 0)
            + int(getattr(snap, "file_missing", 0) or 0)
        )
        # Kısa özet — sayılar korunur; uzun satırlar yok
        parts = [
            f"Toplam: {snap.total}",
            f"Hazır: {snap.loaded}",
            f"Eksik: {snap.missing}",
            f"Decode: {snap.decode_error}",
            f"Cache: {snap.cache_miss}",
            f"Dosya Yok: {snap.file_missing}",
        ]
        detail_tip = (
            f"Toplam Sonuç: {snap.total}\n"
            f"Yüklenen: {snap.loaded}\n"
            f"Eksik: {snap.missing}\n"
            f"Decode Hatası: {snap.decode_error}\n"
            f"Cache Eksik: {snap.cache_miss}\n"
            f"Dosya Yok: {snap.file_missing}\n"
            f"Hata (toplam sayaç): {err}"
        )
        if snap.pending:
            parts.append(f"Kuyruk: {snap.pending}")
            detail_tip += f"\nKuyruk: {snap.pending}"
        self.lbl_thumb_health.setText(
            f"<b>Thumbnail</b>  {' | '.join(parts)}"
        )
        self.lbl_thumb_health.setToolTip(detail_tip)
        self.lbl_thumb_health.setVisible(True)

        fails = list(getattr(snap, "failures", None) or [])
        if fails:
            # Tüm örnekler erişilebilir; scroll alanında gösterilir (listeyi şişirmez)
            lines = []
            for row in fails:
                fn = str(row.get("filename", "") or "")
                reason = str(row.get("reason", "") or "").replace("\n", " ").strip()
                if len(reason) > 220:
                    reason = reason[:217] + "…"
                lines.append(f"☒ {fn}: {reason}" if fn else f"☒ {reason}")
            self.txt_thumb_fails.setPlainText("\n".join(lines))
            n = len(fails)
            self.btn_thumb_fails.setText(
                f"FAIL örnekleri ({n}) ▾"
                if self.btn_thumb_fails.isChecked()
                else f"FAIL örnekleri ({n}) ▸"
            )
            self.btn_thumb_fails.setVisible(True)
            self.txt_thumb_fails.setVisible(self.btn_thumb_fails.isChecked())
            self.thumb_health_box.setMaximumHeight(
                110 if self.btn_thumb_fails.isChecked() else 48
            )
        else:
            self.btn_thumb_fails.setVisible(False)
            self.txt_thumb_fails.clear()
            self.txt_thumb_fails.setVisible(False)
            self.thumb_health_box.setMaximumHeight(36)

        self.thumb_health_box.setVisible(True)

    def set_query_context(self, ctx: QueryContext) -> None:
        self._query_ctx = ctx
        self._rebuild_group_filters()

    def _rebuild_group_filters(self) -> None:
        while self.grp_grid.count():
            item = self.grp_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._group_checks.clear()
        defs = [
            (k, l)
            for k, l in group_definitions(self._query_ctx)
            if groups_relevant_for_query(k, self._query_ctx)
        ]
        for i, (key, label) in enumerate(defs):
            chk = QCheckBox(label)
            chk.setChecked(group_checked_by_default(key))
            chk.setToolTip(label)
            if key == G_EXACT:
                chk.setEnabled(False)
                chk.setToolTip("Exact-first koruması: birebir sonuçlar gizlenemez")
            chk.stateChanged.connect(self.group_filter_changed.emit)
            self._group_checks[key] = chk
            self.grp_grid.addWidget(chk, i // 3, i % 3)

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        idx = self.cmb_view.findData(mode)
        if idx >= 0:
            self.cmb_view.setCurrentIndex(idx)

    def _on_view_changed(self) -> None:
        self._view_mode = self.cmb_view.currentData()
        self.view_mode_changed.emit(self._view_mode)

    def set_scroll_page_size(self, size: int) -> None:
        self._page_size = max(100, int(size or 500))

    def visible_count(self) -> int:
        return min(self._page_limit, len(self._full_results))

    def rendered_card_count(self) -> int:
        return self.scroll.card_count()

    def append_more_pages(self, pages: int = 1) -> bool:
        if not self._full_results:
            return False
        old = self._page_limit
        step = max(INFINITE_PAGE_SIZE, self._page_size // 5)
        self._page_limit = min(
            self._page_limit + pages * step,
            len(self._full_results),
        )
        if self._page_limit <= old:
            return False
        self._rebuild_virtual_entries()
        self.scroll.set_entries(
            self._virtual_entries,
            preserve_scroll=True,
        )
        self.scroll.set_context(
            view_mode=self._view_mode,
            query_ctx=self._query_ctx,
            simple_mode=self._simple_mode,
            ranked=self._ranked,
        )
        self.btn_load_more.setVisible(self._page_limit < len(self._full_results))
        self._schedule_visible_thumbnails()
        return self._page_limit < len(self._full_results)

    def update_page_footer(self, stats=None) -> None:
        above = self._total_above or len(self._full_results)
        shown = self.visible_count()
        rendered = len(self.scroll.visible_cards())
        parts = [f"Gosterilen: {shown:,} / Toplam: {above:,}  (ekranda {rendered} kart)"]
        if self._page_limit < len(self._full_results):
            parts.append(f"Asagi kaydirin veya Daha fazla yukle (+{INFINITE_PAGE_SIZE})")
        self.lbl_page.setText("  |  ".join(parts))
        self.btn_load_more.setVisible(self._page_limit < len(self._full_results))

    def show_loading_skeleton(self, message: str = "Aranıyor…") -> None:
        self.scroll.clear_entries()
        self.set_search_activity(True, message)

    def show_empty_results(self, message: str) -> None:
        self.scroll.clear_entries()
        self.lbl_page.setText(message)
        self.btn_load_more.setVisible(False)

    def has_result_cards(self) -> bool:
        return bool(self.scroll.visible_cards())

    def _clear_result_widgets(self) -> None:
        self._virtual_entries = []
        self._virtual_rendered = 0
        self._virtual_cursor = 0
        self._virtual_ranked_results = []
        self._cards_by_id.clear()
        self._last_visible_thumb_ids = ()
        self.scroll.clear_entries()

    def _rebuild_virtual_entries(self) -> None:
        visible = self._full_results[: self._page_limit]
        entries: list[tuple] = []
        # Sıra numarası görünüm modundan bağımsızdır. Kullanıcı hangi gruplama
        # seçili olursa olsun ekranda gördüğü kartın 1,2,3... sırasını görür.
        rank = 1
        if self._ranked:
            for hdr_text, items in self._group_results_by_layer(visible):
                entries.append(("header", f"▸ {hdr_text}", "ranked"))
                for result in items:
                    entries.append(("card", result, rank))
                    rank += 1
        elif self._simple_mode:
            for hdr_text, items in self._group_results_simple(visible):
                if hdr_text:
                    entries.append(("header", f"▸ {hdr_text}  ({len(items)})", ""))
                for result in items:
                    entries.append(("card", result, rank))
                    rank += 1
        else:
            groups: dict[str, list[SearchResult]] = {}
            for r in visible:
                key = normalize_cluster_key(r.cluster_group or r.category)
                groups.setdefault(key, []).append(r)
            for key, label in group_definitions(self._query_ctx):
                if not self._group_checks.get(key, QCheckBox()).isChecked():
                    continue
                items = groups.get(key, [])
                if not items:
                    continue
                entries.append(("header", f"▸ {label}  ({len(items)})", ""))
                for result in items:
                    entries.append(("card", result, rank))
                    rank += 1
            shown_keys = {
                normalize_cluster_key(k)
                for k in self._group_checks
                if self._group_checks[k].isChecked()
            }
            for key, items in groups.items():
                if key in shown_keys:
                    continue
                if not groups_relevant_for_query(key, self._query_ctx):
                    continue
                if not items:
                    continue
                entries.append(
                    ("header", f"▸ {key.replace('_', ' ').title()}  ({len(items)})", "")
                )
                for result in items:
                    entries.append(("card", result, rank))
                    rank += 1
        if self._below_threshold and not self._ranked:
            entries.append(
                (
                    "header",
                    f"▸ Eşik altı en yakın adaylar ({len(self._below_threshold)})",
                    "near",
                )
            )
            for result in self._below_threshold:
                entries.append(("card", result, rank))
                rank += 1
        self._virtual_entries = entries

    def _select_result(self, result: SearchResult) -> None:
        self._selected_file_id = int(result.file_id)
        self.scroll.set_selected_file_id(self._selected_file_id)
        self.scroll.setFocus(Qt.FocusReason.OtherFocusReason)
        self.result_selected.emit(result)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Forward ↑/↓ to virtual list when focus is not in a text field."""
        from PySide6.QtWidgets import (
            QAbstractSpinBox,
            QComboBox,
            QLineEdit,
            QPlainTextEdit,
            QTextEdit,
        )

        key = event.key()
        if key not in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            super().keyPressEvent(event)
            return
        fw = self.window().focusWidget() if self.window() else None
        if fw is not None:
            if isinstance(
                fw, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)
            ):
                super().keyPressEvent(event)
                return
            if isinstance(fw, QComboBox) and fw.isEditable():
                super().keyPressEvent(event)
                return
        # Only navigate when focus is within this results panel / list.
        if fw is not None and fw is not self and not self.isAncestorOf(fw):
            super().keyPressEvent(event)
            return
        if self.scroll.navigate_by(1 if key == Qt.Key.Key_Down else -1):
            event.accept()
            return
        event.accept()

    def set_search_activity(
        self,
        active: bool,
        message: str = "",
        *,
        result_count: int = 0,
        refining: bool = False,
    ) -> None:
        self._searching = active
        if not active:
            self.lbl_search_activity.setVisible(False)
            return
        parts = [message or "Benzer desenler aranıyor…"]
        if result_count > 0:
            parts.append(f"({result_count:,} sonuç)")
        if refining:
            parts.append("— sıralama güncelleniyor")
        self.lbl_search_activity.setText(" ".join(parts))
        self.lbl_search_activity.setVisible(True)

    def update_scores_in_place(
        self,
        results: list[SearchResult],
        *,
        stage: str,
    ) -> bool:
        """Progressive aşama: mevcut kartları güncelle — rebuild / deleteLater yok.

        Yeni file_id'ler geldiyse False döner; çağıran show_results ile listeyi kurmalı.
        """
        incoming = list(results or [])
        if not incoming:
            return False
        cards = list(self.scroll.visible_cards())
        if not cards:
            return False
        existing_ids = {int(r.file_id) for r in (self._full_results or [])}
        if not existing_ids:
            existing_ids = {int(card.result.file_id) for card in cards}
        if any(int(r.file_id) not in existing_ids for r in incoming):
            return False
        old_order = [int(r.file_id) for r in (self._full_results or [])]
        merged = merge_results_in_engine_order(self._full_results, incoming)
        stamp_display_match_percent(merged)
        self._full_results = merged
        new_order = [int(r.file_id) for r in merged]
        result_map = {int(r.file_id): r for r in merged}
        if old_order != new_order:
            self._rebuild_virtual_entries()
            self.scroll.set_entries(self._virtual_entries, preserve_scroll=True)
        matched = 0
        for card in self.scroll.visible_cards():
            result = result_map.get(int(card.result.file_id))
            if result is None:
                continue
            matched += 1
            card.update_result(result, stage)
            card.set_selected(int(result.file_id) == self._selected_file_id)
        if matched == 0:
            return False
        self.lbl_search_activity.setVisible(False)
        self._apply_cached_thumbnails_to_visible()
        self._schedule_visible_thumbnails()
        self._update_page_label(None, self._ranked, self._score_floor, self._below_threshold)
        return True

    def show_results(
        self,
        results: list[SearchResult],
        stats=None,
        threshold: float = 0.0,
        has_more: bool = False,
        below_threshold: list[SearchResult] | None = None,
        display_limit: int = 0,
        query_ctx: QueryContext | None = None,
        ranked: bool = False,
        score_floor: float = 0.0,
    ) -> None:
        preserve_scroll = self._searching and bool(self._full_results)
        previous_scroll = self.scroll.verticalScrollBar().value()
        if query_ctx:
            self.set_query_context(query_ctx)

        stamped = stamp_display_match_percent(list(results or []))
        incoming_empty = not stamped and not (below_threshold or [])
        if incoming_empty and self._searching and self._full_results:
            return
        self._full_results = stamped
        self._preview_sel.retain(int(r.file_id) for r in self._full_results)
        self._ranked = ranked
        self._score_floor = score_floor
        self._below_threshold = list(below_threshold or [])
        self._total_above = stats.above_threshold if stats else len(self._full_results)
        self._last_thumb_expected_ids = ()
        batch = display_limit or self._page_size
        if batch > 0:
            self._page_size = batch
        self._page_limit = min(
            INITIAL_RENDER_COUNT,
            len(self._full_results),
        )

        if results:
            self.lbl_search_activity.setVisible(False)
            if not preserve_scroll:
                self.scroll.verticalScrollBar().setValue(0)

        self.grp_scroll.setVisible(not ranked and not self._simple_mode)
        self.btn_load_more.setVisible(
            has_more or self._page_limit < len(self._full_results)
        )
        self._update_page_label(stats, ranked, score_floor, below_threshold)

        self.setUpdatesEnabled(False)
        try:
            self._clear_result_widgets()

            if not results and not below_threshold:
                self.show_empty_results(
                    "Sonuç bulunamadı. Semantik aramayı veya kapsamı genişletmeyi deneyin."
                    if ranked
                    else "Eşik üstü sonuç yok. Eşiği düşürün veya «Eşik altı yakınlar»ı açın."
                )
                return

            self._rebuild_virtual_entries()
            self.scroll.set_context(
                view_mode=self._view_mode,
                query_ctx=self._query_ctx,
                simple_mode=self._simple_mode,
                ranked=self._ranked,
            )
            self.scroll.set_entries(self._virtual_entries, preserve_scroll=preserve_scroll)
            self._refresh_preview_checks()
            QTimer.singleShot(0, self._after_results_rendered)
            if preserve_scroll:
                bar = self.scroll.verticalScrollBar()
                bar.setValue(min(previous_scroll, bar.maximum()))
        finally:
            self.setUpdatesEnabled(True)

    def _after_results_rendered(self) -> None:
        self.scroll.refresh_layout()
        self._refresh_preview_checks()
        self._schedule_visible_thumbnails()
        self.update_page_footer()

    def _update_page_label(
        self,
        stats,
        ranked: bool,
        score_floor: float,
        below_threshold,
    ) -> None:
        above = self._total_above
        if self._simple_mode:
            parts = [f"{above:,} sonuç"]
        else:
            parts = [
                f"Gösterilen: {self.visible_count():,} / Toplam: {above:,}"
            ]
        if ranked and score_floor > 0:
            parts[0] = (
                f"Toplam: {above:,} (%{score_floor * 100:.0f} ve üzeri)"
            )
        if self._page_limit < len(self._full_results):
            parts.append(f"Sayfa: {self._page_size:,}")
        if below_threshold and not ranked:
            parts.append(f"Eşik altı önizleme: {len(below_threshold):,} aday")
        self.lbl_page.setText("  |  ".join(parts) if parts else "")

    def is_group_visible(self, category: str) -> bool:
        key = normalize_cluster_key(category)
        chk = self._group_checks.get(key)
        return chk.isChecked() if chk else True
