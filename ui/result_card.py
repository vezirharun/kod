"""Sonuç kartı — önce görsel, sonra tasarımcı etiketleri."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFontMetrics, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from core.display_match_percent import (
    DISPLAY_MATCH_TOOLTIP,
    display_match_percent,
    format_display_match_label,
)
from core.dynamic_groups import QueryContext
from core.search_engine import SearchResult
from ui.designer_labels import (
    brand_badge_text,
    color_badge_text,
    family_badge_for_result,
    independent_feature_badge,
    similarity_tier_label,
    stage_label,
)
from ui.theme import score_color

VIEW_CARD = "card"
VIEW_LIST = "list"
VIEW_COMPACT = "compact"
VIEW_LARGE = "large"

_HEIGHTS = {VIEW_CARD: 108, VIEW_LIST: 88, VIEW_COMPACT: 64, VIEW_LARGE: 140}
_THUMB = {VIEW_CARD: 88, VIEW_LIST: 72, VIEW_COMPACT: 52, VIEW_LARGE: 120}
_CONF_LABEL_CACHE: dict[int, tuple[str, str]] = {}


def _chip(text: str, bg: str, fg: str = "#fff") -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background:{bg};color:{fg};padding:2px 8px;border-radius:4px;"
        "font-size:10px;font-weight:600;"
    )
    lbl.setMaximumWidth(160)
    return lbl


class ResultCard(QFrame):
    selected = Signal(object)
    open_folder = Signal(int)
    open_file = Signal(int)
    hover_entered = Signal(int, str, str, str)  # file_id, thumb_path, filename, source_path
    hover_left = Signal()
    preview_menu_requested = Signal(object, object)
    preview_click = Signal(object, object)  # result, Qt modifiers
    preview_checkbox = Signal(object, bool)

    def __init__(
        self,
        result: SearchResult,
        view_mode: str = VIEW_CARD,
        query_ctx: QueryContext | None = None,
        rank: int = 0,
        simple_mode: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("ResultCard")
        self.result = result
        self.view_mode = view_mode
        self._query_ctx = query_ctx or QueryContext()
        self._rank = rank
        self._simple_mode = simple_mode
        self._selected = False
        self._preview_checked = False
        self._thumbnail_pixmap = QPixmap()
        self._last_bound_fid = int(result.file_id) if result else 0
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        h = _HEIGHTS.get(view_mode, 108)
        self.setFixedHeight(h)
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)
        self.setStyleSheet(
            "ResultCard { background: #1c2129; border: 1px solid #252b36; border-radius: 6px; }"
            "ResultCard:hover { border-color: #3d4654; background:#1e2430; }"
        )
        self._build_ui()

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        border = "#64748b" if self._selected else "#252b36"
        self.setStyleSheet(
            f"ResultCard {{ background:#1c2129;border:1px solid {border};border-radius:6px; }}"
            "ResultCard:hover { border-color:#3d4654; background:#1e2430; }"
        )

    def set_preview_checked(self, checked: bool) -> None:
        self._preview_checked = bool(checked)
        chk = getattr(self, "chk_preview", None)
        if chk is None:
            return
        chk.blockSignals(True)
        chk.setChecked(self._preview_checked)
        chk.blockSignals(False)

    def _on_preview_checkbox(self, checked: bool) -> None:
        if self.result and int(self.result.file_id) > 0:
            self.preview_checkbox.emit(self.result, bool(checked))

    def _emit_preview_menu(self, global_pos) -> None:
        if self.result and int(self.result.file_id) > 0:
            self.preview_menu_requested.emit(self.result, global_pos)

    def _build_ui(self) -> None:
        layout = self.layout()
        if not isinstance(layout, QHBoxLayout):
            layout = QHBoxLayout(self)
            layout.setContentsMargins(8, 6, 8, 6)
            layout.setSpacing(12)
        else:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # 1) Thumbnail first — dominant visual
        ts = _THUMB.get(self.view_mode, 88)
        self.thumb = QLabel()
        self.thumb.setFixedSize(ts, ts)
        self.thumb.setScaledContents(True)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet(
            "background:#232a35;border:1px solid #2a3340;border-radius:6px;"
            "color:#7c8698;font-size:12px;"
        )
        self.thumb.setText("···")
        self.thumb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.thumb.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.thumb.customContextMenuRequested.connect(
            lambda pos: self._emit_preview_menu(self.thumb.mapToGlobal(pos))
        )
        self.thumb.installEventFilter(self)

        self.chk_preview = QCheckBox()
        self.chk_preview.setObjectName("chk_preview")
        self.chk_preview.setToolTip("Seç")
        self.chk_preview.setCursor(Qt.CursorShape.ArrowCursor)
        self.chk_preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_preview.setFixedSize(18, 18)
        self.chk_preview.setChecked(self._preview_checked)
        self.chk_preview.clicked.connect(self._on_preview_checkbox)
        layout.addWidget(self.chk_preview, alignment=Qt.AlignmentFlag.AlignTop)

        # Sıra numarası yalnızca bilgi satırında (#1, #2, #3...) gösterilir.
        # Thumbnail üzerinde ikinci bir numara/badge gösterilmez; bu, aynı
        # sıra bilgisinin iki kez görünmesini ve görseli kapatmasını önler.
        layout.addWidget(self.thumb)

        info = QVBoxLayout()
        info.setSpacing(4)

        # Designer badges: family / color / brand
        badges = QHBoxLayout()
        badges.setSpacing(6)
        fam = family_badge_for_result(self.result)
        if fam:
            badges.addWidget(_chip(fam, "#0f766e"))
        extra = independent_feature_badge(self.result)
        if extra:
            badges.addWidget(_chip(extra, "#334155"))
        color = color_badge_text(self.result.color_family)
        if color:
            badges.addWidget(_chip(color, "#9a3412"))
        brand = brand_badge_text(self.result)
        if brand:
            badges.addWidget(_chip(brand, "#1e3a5f", "#e2e8f0"))
        badges.addStretch()
        if self.view_mode != VIEW_COMPACT:
            info.addLayout(badges)

        # Filename — readable, not overpowered by scores
        fname_raw = self.result.filename
        fm = QFontMetrics(self.font())
        width = 280 if self.view_mode != VIEW_COMPACT else 200
        fname = fm.elidedText(fname_raw, Qt.TextElideMode.ElideMiddle, width)
        fname_lbl = QLabel(fname)
        fname_lbl.setToolTip(fname_raw)
        fname_lbl.setStyleSheet("font-weight:600;font-size:13px;color:#f1f5f9;")
        self.fname_lbl = fname_lbl
        info.addWidget(fname_lbl)

        # Face Intelligence is a separate evidence layer; never label a face
        # match as a pattern-family/doku result.
        is_face_match = bool((self.result.debug or {}).get("face_match"))
        try:
            from core.confidence_engine import compute_confidence
            _mode = "text" if self.result.debug.get("text_mode") else "image"
            _conf = compute_confidence(self.result, mode=_mode)
        except Exception:
            _conf = None
        if is_face_match:
            tier = "Aynı Kişi"
        elif _conf is not None and _conf.match_type:
            tier = _conf.match_type
        else:
            tier = similarity_tier_label(self.result)
        self.stage_lbl = QLabel(tier)
        self.stage_lbl.setStyleSheet(("color:#22c55e;font-size:11px;font-weight:700;" if is_face_match else "color:#cbd5e1;font-size:11px;font-weight:600;"))
        pct = display_match_percent(self.result)
        sc = score_color(pct)
        score_text = format_display_match_label(self.result, self._rank)
        self.score_lbl = QLabel(score_text)
        self.score_lbl.setStyleSheet(
            f"color:{sc};font-size:11px;font-weight:500;opacity:0.85;"
        )
        self.score_lbl.setToolTip(
            "Yüz eşleşmesi skoru" if is_face_match else DISPLAY_MATCH_TOOLTIP
        )

        # Confidence from evidence kind — not a second copy of similarity %.
        try:
            if _conf is None:
                raise RuntimeError("no conf")
            _conf_colors = {
                "Çok Yüksek": "#22c55e",
                "Yüksek":     "#84cc16",
                "Orta":       "#f59e0b",
                "Düşük":      "#ef4444",
            }
            _conf_color = _conf_colors.get(_conf.level, "#94a3b8")
            self.conf_lbl = QLabel(f"Güven: {_conf.level}")
            self.conf_lbl.setStyleSheet(
                f"color:{_conf_color};font-size:10px;font-weight:600;"
            )
            _tooltip_lines = [
                l.replace("+ ", "\u2713 ").replace("- ", "\u2717 ")
                for l in (_conf.reasons + _conf.warnings)[:6]
            ]
            tip = _conf.summary
            if _tooltip_lines:
                tip = tip + "\n" + "\n".join(_tooltip_lines)
            self.conf_lbl.setToolTip(tip)
        except Exception:
            self.conf_lbl = QLabel("")

        meta = QHBoxLayout()
        meta.setSpacing(10)
        meta.addWidget(self.stage_lbl)
        meta.addWidget(self.score_lbl)
        meta.addWidget(self.conf_lbl)
        meta.addStretch()
        info.addLayout(meta)

        # Keep attrs for update_result compatibility (hidden technical)
        self.engine_lbl = QLabel("")
        self.engine_lbl.hide()
        self.contrib_lbl = QLabel("")
        self.contrib_lbl.hide()
        self.badge = self.stage_lbl

        if self.view_mode != VIEW_COMPACT:
            src = QLabel(self.result.source_name or "—")
            src.setStyleSheet("color:#64748b;font-size:10px;")
            self.src_lbl = src
            info.addWidget(src)

        layout.addLayout(info, stretch=1)

        if self.view_mode != VIEW_COMPACT:
            btn_row = QVBoxLayout()
            btn_row.setSpacing(4)
            btn_folder = QPushButton("📁")
            btn_folder.setFixedSize(28, 28)
            btn_folder.setToolTip("Klasörde aç")
            btn_folder.setStyleSheet(
                "QPushButton{background:#1e293b;border:1px solid #334155;border-radius:4px;}"
            )
            btn_folder.clicked.connect(
                lambda: self.open_folder.emit(self.result.file_id)
            )
            btn_file = QPushButton("↗")
            btn_file.setFixedSize(28, 28)
            btn_file.setToolTip("Dosyayı aç")
            btn_file.setStyleSheet(
                "QPushButton{background:#1e293b;border:1px solid #334155;border-radius:4px;}"
            )
            btn_file.clicked.connect(lambda: self.open_file.emit(self.result.file_id))
            btn_row.addWidget(btn_folder)
            btn_row.addWidget(btn_file)
            btn_row.addStretch(1)
            layout.addLayout(btn_row)

    def rebind(
        self,
        result: SearchResult,
        *,
        view_mode: str | None = None,
        query_ctx: QueryContext | None = None,
        rank: int = 0,
        simple_mode: bool = False,
    ) -> None:
        """Widget recycling — yeni QWidget oluşturmadan veri bağla."""
        new_mode = view_mode or self.view_mode
        if new_mode != self.view_mode:
            self.view_mode = new_mode
            self._rank = rank
            self._simple_mode = simple_mode
            self._query_ctx = query_ctx or self._query_ctx
            h = _HEIGHTS.get(self.view_mode, 108)
            self.setFixedHeight(h)
            self.setMinimumHeight(h)
            self.setMaximumHeight(h)
            # Layout temizle ve yeniden kur (mod değişimi nadir)
            while self.layout().count():
                item = self.layout().takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._build_ui()
        prev_fid = int(self.result.file_id) if self.result else 0
        self.result = result
        self._rank = rank
        self._last_bound_fid = int(result.file_id)
        self._update_rank_badge()
        self._update_labels()
        if prev_fid != int(result.file_id) or self._thumbnail_pixmap.isNull():
            self._thumbnail_pixmap = QPixmap()
            self.thumb.setPixmap(QPixmap())
            self.thumb.setText("···")
            self.thumb.setStyleSheet(
                "background:#232a35;border:1px solid #2a3340;border-radius:6px;"
                "color:#7c8698;font-size:12px;"
            )
            self._try_cached_thumbnail()

    def _try_cached_thumbnail(self) -> None:
        try:
            from core.thumbnail_cache import get_thumbnail_cache

            _img, pix = get_thumbnail_cache().get(
                int(self.result.file_id), self.thumb_size()
            )
            if pix is not None and not pix.isNull():
                self._apply_pixmap(pix)
            elif _img is not None and not _img.isNull():
                self._apply_pixmap(QPixmap.fromImage(_img))
        except Exception:
            pass

    def _update_rank_badge(self) -> None:
        # Geriye dönük uyumluluk: virtual list bu metodu çağırmaya devam
        # edebilir. Sıra numarası artık yalnızca #N skor satırındadır.
        return

    def _update_labels(self) -> None:
        """Mevcut label'ları güncelle — layout yeniden kurma yok."""
        self._update_rank_badge()
        fname_raw = self.result.filename
        fm = QFontMetrics(self.font())
        width = 280 if self.view_mode != VIEW_COMPACT else 200
        fname = fm.elidedText(fname_raw, Qt.TextElideMode.ElideMiddle, width)
        if hasattr(self, "fname_lbl"):
            self.fname_lbl.setText(fname)
            self.fname_lbl.setToolTip(fname_raw)
        if hasattr(self, "stage_lbl"):
            is_face_match = bool((self.result.debug or {}).get("face_match"))
            if is_face_match:
                self.stage_lbl.setText("Aynı Kişi")
            else:
                try:
                    from core.confidence_engine import compute_confidence

                    _mode = "text" if self.result.debug.get("text_mode") else "image"
                    _mt = compute_confidence(self.result, mode=_mode).match_type
                    self.stage_lbl.setText(_mt or similarity_tier_label(self.result))
                except Exception:
                    self.stage_lbl.setText(similarity_tier_label(self.result))
        pct = display_match_percent(self.result)
        sc = score_color(pct)
        score_text = format_display_match_label(self.result, self._rank)
        if hasattr(self, "score_lbl"):
            self.score_lbl.setText(score_text)
            self.score_lbl.setStyleSheet(
                f"color:{sc};font-size:11px;font-weight:500;opacity:0.85;"
            )
            is_face = bool((self.result.debug or {}).get("face_match"))
            self.score_lbl.setToolTip(
                "Yüz eşleşmesi skoru" if is_face else DISPLAY_MATCH_TOOLTIP
            )
        if hasattr(self, "conf_lbl"):
            fid = int(self.result.file_id)
            cached = _CONF_LABEL_CACHE.get(fid)
            if cached:
                text, style = cached
                self.conf_lbl.setText(text)
                self.conf_lbl.setStyleSheet(style)
            else:
                try:
                    from core.confidence_engine import compute_confidence

                    _mode = "text" if self.result.debug.get("text_mode") else "image"
                    _conf = compute_confidence(self.result, mode=_mode)
                    _conf_colors = {
                        "Cok Yuksek": "#22c55e",
                        "Yuksek": "#84cc16",
                        "Orta": "#f59e0b",
                        "Dusuk": "#ef4444",
                        "Çok Yüksek": "#22c55e",
                        "Yüksek": "#84cc16",
                        "Düşük": "#ef4444",
                    }
                    _conf_color = _conf_colors.get(_conf.level, "#94a3b8")
                    text = f"Güven: {_conf.level}"
                    style = f"color:{_conf_color};font-size:10px;font-weight:600;"
                    _CONF_LABEL_CACHE[fid] = (text, style)
                    self.conf_lbl.setText(text)
                    self.conf_lbl.setStyleSheet(style)
                except Exception:
                    self.conf_lbl.setText("")
        if hasattr(self, "src_lbl"):
            self.src_lbl.setText(self.result.source_name or "—")

    def thumb_size(self) -> int:
        return _THUMB.get(self.view_mode, 88)

    def thumbnail_request(self) -> tuple | None:
        path = str(self.result.thumbnail_path or "")
        source = str(self.result.path or "")
        fp = str(getattr(self.result, "feature_preview_path", "") or "")
        if not path and not source and not fp:
            return None
        return (
            int(self.result.file_id),
            path,
            self.thumb_size(),
            source,
            str(self.result.filename or ""),
            fp,
        )

    def _apply_pixmap(self, pix: QPixmap) -> None:
        if pix.isNull():
            return
        self._thumbnail_pixmap = pix
        self.thumb.setText("")
        self.thumb.setToolTip(self.result.filename)
        self.thumb.setStyleSheet(
            "background:#232a35;border:1px solid #2a3340;border-radius:6px;"
            "color:#7c8698;font-size:12px;"
        )
        self.thumb.setPixmap(pix)

    def apply_thumbnail(self, image, reason: str = "") -> None:
        if image is None:
            msg = "!"
            tip = reason or "Önizleme yüklenemedi"
            self.thumb.setText(msg)
            self.thumb.setToolTip(f"FAIL\nDosya: {self.result.filename}\nSebep: ☒ {tip}")
            self.thumb.setStyleSheet(
                "background:#2a2228;border:1px solid #3d2a32;border-radius:6px;"
                "color:#f87171;font-size:14px;font-weight:700;"
            )
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            self.apply_thumbnail(None, reason or "Decode başarısız")
            return
        self._apply_pixmap(pix)

    def update_result(self, result: SearchResult, stage: str = "Tam Analiz") -> None:
        self.result = result
        pct = display_match_percent(result)
        score_text = format_display_match_label(result, self._rank)
        self.score_lbl.setText(score_text)
        self.score_lbl.setStyleSheet(
            f"color:{score_color(pct)};font-size:11px;font-weight:500;"
        )
        is_face = bool((result.debug or {}).get("face_match"))
        self.score_lbl.setToolTip(
            "Yüz eşleşmesi skoru" if is_face else DISPLAY_MATCH_TOOLTIP
        )
        display = stage_label(stage) if stage else similarity_tier_label(result)
        self.stage_lbl.setText(display)
        self.stage_lbl.setStyleSheet(
            "color:#cbd5e1;font-size:11px;font-weight:600;"
        )
        if self._thumbnail_pixmap.isNull():
            self._try_cached_thumbnail()

    def contextMenuEvent(self, event) -> None:
        self._emit_preview_menu(event.globalPos())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            child = self.childAt(pos)
            on_chk = child is getattr(self, "chk_preview", None)
            if not on_chk:
                self.selected.emit(self.result)
                if self.result and int(self.result.file_id) > 0:
                    self.preview_click.emit(self.result, event.modifiers())
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_file.emit(self.result.file_id)
        super().mouseDoubleClickEvent(event)

    def enterEvent(self, event) -> None:
        path = str(self.result.thumbnail_path or "")
        source = str(self.result.path or "")
        if path or source:
            self.hover_entered.emit(
                int(self.result.file_id),
                path,
                self.result.filename,
                source,
            )
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.hover_left.emit()
        super().leaveEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if watched is getattr(self, "thumb", None):
            if event.type() == QEvent.Type.ContextMenu:
                self._emit_preview_menu(event.globalPos())
                return True
            if event.type() == QEvent.Type.Enter:
                path = str(self.result.thumbnail_path or "")
                source = str(self.result.path or "")
                if path or source:
                    self.hover_entered.emit(
                        int(self.result.file_id),
                        path,
                        self.result.filename,
                        source,
                    )
        return super().eventFilter(watched, event)
