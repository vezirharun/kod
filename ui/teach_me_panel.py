"""Bana Öğret paneli — toplu inceleme ve kalıcı kavram öğretimi. Düzenle değildir."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize, QRect, QThread, QTimer, Signal
from PySide6.QtGui import QFontMetrics, QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.autonomous_learn import (
    confirm_reviews,
    correct_reviews,
    reject_reviews,
    undecided_candidate_labels,
)
from core.concept_registry import concepts
from core.concept_query_normalize import sorted_turkish
from core.db import Database
from core.logger import setup_logger
from core.visual_family_candidates import expand_member_ids, reject_and_split_family
from core.teach_me import (
    TeachMeCard,
    best_preview_path,
    common_overlay_from_snapshots,
    dismiss_files,
    enrich_teach_card_paths,
    list_inbox_pools,
    snapshot_file_classification,
    summarize_classification_changes,
    teach_files,
)
from core.user_feedback import UserFeedbackStore
from ui.preview_dialog import ImagePreviewDialog
from ui.result_metadata_dialog import ResultMetadataDialog
from ui.thumbnail_scheduler import ThumbnailScheduler, get_thumbnail_scheduler

logger = setup_logger(__name__)

_THUMB_EDGE = 160
_CARD_TEXT_WIDTH = 168
_CARD_H_PAD = 12
_CARD_V_GAP = 10
_REASON_MAX_CHARS = 140
_CARD_BATCH = 40
_ICON_BATCH = 12


def _active_concept_names(db_path: str) -> list[str]:
    """Active concept_registry labels — call off the UI thread when possible."""
    names: list[str] = []
    for row in concepts(db_path):
        status = str(row.get("status") or "active")
        if status in ("inactive", "retired"):
            continue
        name = str(row.get("canonical") or "")
        if name:
            names.append(name)
    return sorted_turkish(names)


class _TeachMeInboxWorker(QThread):
    """SQLite + preview path resolve — UI thread dışında."""

    finished_ok = Signal(int, object, object)  # req_id, pools, concept_names
    failed = Signal(int, str)

    def __init__(
        self,
        db_path: str,
        cache_dir: str,
        req_id: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._db_path = str(db_path)
        self._cache_dir = str(cache_dir or "")
        self._req_id = int(req_id)

    def run(self) -> None:
        try:
            db = Database(self._db_path)
            pools = list_inbox_pools(db, self._db_path)
            pools.setdefault("new_concept", [])
            for cards in pools.values():
                for card in cards or []:
                    enrich_teach_card_paths(
                        card, db=db, cache_dir=self._cache_dir
                    )
            names = _active_concept_names(self._db_path)
            self.finished_ok.emit(self._req_id, pools, names)
        except Exception as exc:
            logger.exception("TeachMe inbox worker failed")
            self.failed.emit(self._req_id, str(exc))


def _card_list_text(card: TeachMeCard) -> str:
    conf_pct = int(round(float(card.confidence or 0) * 100))
    extra = ""
    if int(getattr(card, "cluster_size", 1) or 1) > 1:
        extra = f" · {int(card.cluster_size)} görsel"
    reason = " ".join(str(card.reason or "").split())
    if len(reason) > _REASON_MAX_CHARS:
        reason = reason[: _REASON_MAX_CHARS - 1].rstrip() + "…"
    return (
        f"{card.filename}\n"
        f"Öneri: {card.guess} · Güven %{conf_pct}{extra}\n"
        f"{reason}"
    )


def _card_item_size_hint(text: str, *, thumb_edge: int = _THUMB_EDGE) -> QSize:
    """IconMode: ikon + wrap metin yüksekliği — satır overlap önlenir."""
    fm = QFontMetrics(QLabel().font())
    text_w = max(int(thumb_edge), _CARD_TEXT_WIDTH)
    bounds = fm.boundingRect(
        QRect(0, 0, text_w, 10_000),
        int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
        str(text or ""),
    )
    text_h = max(fm.height() * 3, int(bounds.height()) + fm.leading())
    width = text_w + _CARD_H_PAD
    height = int(thumb_edge) + _CARD_V_GAP + text_h + _CARD_H_PAD
    return QSize(width, height)


def _items_overlap(rects: list[QRect]) -> bool:
    for i, a in enumerate(rects):
        if not a.isValid() or a.isEmpty():
            continue
        for b in rects[i + 1 :]:
            if b.isValid() and not b.isEmpty() and a.intersects(b):
                return True
    return False


def _thumb_path(card: TeachMeCard, *, cache_dir: str = "", db=None) -> str:
    """Liste ikonu: worker'da enrich edilmiş path'i tercih et (UI'de DB yok)."""
    for raw in (card.preview_path, card.feature_preview_path):
        p = Path(str(raw or ""))
        try:
            if p.is_file() and p.stat().st_size > 0:
                return str(p)
        except OSError:
            continue
    # Fallback (eski kart / sync test): tek seferlik resolve — üretimde nadir.
    from core.thumb_resolve import resolve_thumb_path

    if db is not None:
        enrich_teach_card_paths(card, db=db, cache_dir=str(cache_dir or ""))
    resolved, status = resolve_thumb_path(
        str(card.preview_path or ""),
        cache_dir=str(cache_dir or ""),
        feature_preview_path=str(card.feature_preview_path or ""),
    )
    if status == "hit" and resolved:
        return resolved
    return ""


def _preview_pix(path: str) -> QPixmap:
    p = Path(str(path or ""))
    empty = QPixmap(_THUMB_EDGE, _THUMB_EDGE)
    empty.fill(Qt.GlobalColor.darkGray)
    if not p.is_file() or p.stat().st_size <= 0:
        return empty
    reader = QImageReader(str(p))
    reader.setAutoTransform(True)
    orig = reader.size()
    if orig.width() > 0 and orig.height() > 0:
        reader.setScaledSize(
            orig.scaled(_THUMB_EDGE, _THUMB_EDGE, Qt.AspectRatioMode.KeepAspectRatio)
        )
    img = reader.read()
    if img.isNull():
        return empty
    return QPixmap.fromImage(img)


class TeachMePanel(QWidget):
    taught = Signal(str, int)

    def __init__(self, settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._cards: dict[str, list[TeachMeCard]] = {
            "undefined": [],
            "suspicious": [],
            "undecided": [],
            "new_concept": [],
        }
        self._reload_req = 0
        self._fill_gen = 0
        self._inbox_worker: _TeachMeInboxWorker | None = None
        self._concept_names: list[str] = []
        self._thumb_scheduler: ThumbnailScheduler | None = None
        self._thumb_wait_gen: dict[int, int] = {}
        self._build()

    def set_thumbnail_scheduler(self, scheduler: ThumbnailScheduler | None) -> None:
        """Results ile aynı ThumbnailScheduler — miss'te async ensure (UI/NAS yok)."""
        if self._thumb_scheduler is scheduler:
            return
        if self._thumb_scheduler is not None:
            try:
                self._thumb_scheduler.thumbnail_ready.disconnect(self._on_teach_thumb_ready)
            except (TypeError, RuntimeError):
                pass
        self._thumb_scheduler = scheduler
        if scheduler is not None:
            scheduler.thumbnail_ready.connect(self._on_teach_thumb_ready)

    def _active_thumb_scheduler(self) -> ThumbnailScheduler | None:
        if self._thumb_scheduler is not None:
            return self._thumb_scheduler
        try:
            return get_thumbnail_scheduler()
        except Exception:
            return None

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        hint = QLabel(
            "Sistem emin olduğu görselleri kendi sınıflandırır. "
            "Emin değilse burada sorar: Doğru / Yanlış / Şu. "
            "Düşük güven kesin kavram olarak kaydedilmez. "
            "Görsele tıklayınca önizleme açılır."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.tabs = QTabWidget()
        self.list_undefined = self._make_list()
        self.list_suspicious = self._make_list()
        self.list_undecided = self._make_list()
        self.list_new_concept = self._make_list()
        self.tabs.addTab(self.list_undefined, "Tanımsızlar")
        self.tabs.addTab(self.list_suspicious, "Şüpheliler")
        self.tabs.addTab(self.list_undecided, "Kararsızlar")
        self.tabs.addTab(self.list_new_concept, "Yeni Kavram")
        layout.addWidget(self.tabs, 1)

        self.lbl_review = QLabel("Bir görsel seçin — öneri, güven ve gerekçe burada.")
        self.lbl_review.setWordWrap(True)
        layout.addWidget(self.lbl_review)

        # AKILLI KARARSIZLAR — one-click öğret (mevcut Teach önünde karar katmanı)
        self.smart_undecided = QWidget()
        smart_lay = QVBoxLayout(self.smart_undecided)
        smart_lay.setContentsMargins(0, 4, 0, 4)
        smart_lay.setSpacing(6)
        self.lbl_smart_title = QLabel("BU GÖRSEL NE?")
        title_font = self.lbl_smart_title.font()
        title_font.setBold(True)
        self.lbl_smart_title.setFont(title_font)
        smart_lay.addWidget(self.lbl_smart_title)
        smart_mid = QHBoxLayout()
        self.lbl_smart_thumb = QLabel()
        self.lbl_smart_thumb.setFixedSize(72, 72)
        self.lbl_smart_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_smart_thumb.setStyleSheet("background:#2a2a2a;")
        smart_mid.addWidget(self.lbl_smart_thumb)
        self.smart_candidates_host = QWidget()
        self.smart_candidates_row = QHBoxLayout(self.smart_candidates_host)
        self.smart_candidates_row.setContentsMargins(0, 0, 0, 0)
        self.smart_candidates_row.setSpacing(6)
        smart_mid.addWidget(self.smart_candidates_host, 1)
        smart_lay.addLayout(smart_mid)
        self.btn_hicbiri = QPushButton("Hiçbiri")
        self.btn_hicbiri.setToolTip("Aday yok / yanlış — mevcut öğret diyaloğunu aç")
        self.btn_hicbiri.clicked.connect(self._on_hicbiri)
        smart_lay.addWidget(self.btn_hicbiri)
        self.smart_undecided.hide()
        layout.addWidget(self.smart_undecided)

        btns = QHBoxLayout()
        self.btn_select_all = QPushButton("Tümünü seç")
        self.btn_clear = QPushButton("Seçimi bırak")
        self.btn_confirm = QPushButton("Doğru")
        self.btn_confirm.setToolTip("Önerilen kavramı kalıcı öğren")
        self.btn_reject = QPushButton("Yanlış")
        self.btn_reject.setToolTip("Öneriyi reddet; kesin kavram yazılmaz")
        self.btn_correct = QPushButton("Şu…")
        self.btn_correct.setToolTip("Başka kavram ver; o öğrenilir")
        self.btn_teach = QPushButton("Seçilenlere öğret")
        self.btn_teach.setToolTip("Seç → Sınıflandır → Kaydet → Sisteme öğret")
        self.btn_dismiss = QPushButton("Havuzdan çıkar")
        self.btn_refresh = QPushButton("Yenile")
        btns.addWidget(self.btn_select_all)
        btns.addWidget(self.btn_clear)
        btns.addWidget(self.btn_confirm)
        btns.addWidget(self.btn_reject)
        btns.addWidget(self.btn_correct)
        btns.addWidget(self.btn_teach)
        btns.addWidget(self.btn_dismiss)
        btns.addWidget(self.btn_refresh)
        layout.addLayout(btns)

        self.btn_select_all.clicked.connect(self._select_all)
        self.btn_clear.clicked.connect(self._clear_sel)
        self.btn_confirm.clicked.connect(self._on_confirm)
        self.btn_reject.clicked.connect(self._on_reject)
        self.btn_correct.clicked.connect(self._on_correct)
        self.btn_teach.clicked.connect(self._on_teach)
        self.btn_dismiss.clicked.connect(self._on_dismiss)
        self.btn_refresh.clicked.connect(self.reload)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._last_click_id = 0
        self._preview_open = False
        # Geriye dönük testler: kavram alanı artık sınıflandırma diyaloğunda.
        self.txt_label = QLineEdit()
        self.txt_label.hide()
        self.cmb_known = QComboBox()
        self.cmb_known.hide()

    def _make_list(self) -> QListWidget:
        w = QListWidget()
        w.setViewMode(QListWidget.ViewMode.IconMode)
        w.setResizeMode(QListWidget.ResizeMode.Adjust)
        w.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        w.setIconSize(QSize(_THUMB_EDGE, _THUMB_EDGE))
        w.setWordWrap(True)
        w.setSpacing(14)
        w.setUniformItemSizes(False)
        w.setMovement(QListWidget.Movement.Static)
        w.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        w.itemClicked.connect(self._on_item_clicked)
        w.itemDoubleClicked.connect(self._on_item_preview)
        w.itemSelectionChanged.connect(self._update_review_detail)
        w.verticalScrollBar().valueChanged.connect(
            lambda _v: self._schedule_active_viewport_thumbs()
        )
        return w

    def _on_tab_changed(self, _i: int = 0) -> None:
        self._update_review_detail()
        self._schedule_active_viewport_thumbs()

    def _active_viewport_cards(self) -> list[TeachMeCard]:
        """Active tab only — cards whose icon rect intersects the viewport."""
        widget = self._active_list()
        key = self._pool_key()
        by_id = {int(c.file_id): c for c in (self._cards.get(key) or [])}
        vp = widget.viewport().rect()
        out: list[TeachMeCard] = []
        for i in range(widget.count()):
            item = widget.item(i)
            if item is None:
                continue
            if vp.width() > 0 and vp.height() > 0:
                if not widget.visualItemRect(item).intersects(vp):
                    continue
            fid = int(item.data(Qt.ItemDataRole.UserRole) or 0)
            card = by_id.get(fid)
            if card is not None:
                out.append(card)
        # Layout/viewport not ready (tests / first paint): fall back to active tab.
        if not out and widget.count() > 0:
            for i in range(widget.count()):
                item = widget.item(i)
                if item is None:
                    continue
                fid = int(item.data(Qt.ItemDataRole.UserRole) or 0)
                card = by_id.get(fid)
                if card is not None:
                    out.append(card)
        return out

    def _schedule_active_viewport_thumbs(self) -> None:
        """One request_visible for all active-tab viewport misses (no sibling drop)."""
        gen = self._fill_gen
        sched = self._active_thumb_scheduler()
        if sched is None:
            return
        cache_dir = str(getattr(self.settings, "cache_dir", "") or "")
        items: list[tuple] = []
        for card in self._active_viewport_cards():
            if _thumb_path(card, cache_dir=cache_dir, db=None):
                continue
            fid = int(card.file_id or 0)
            if fid <= 0:
                continue
            source = str(card.path or "").strip()
            preview = str(card.preview_path or "").strip()
            fp = str(card.feature_preview_path or "").strip()
            if not source and not preview and not fp:
                continue
            self._thumb_wait_gen[fid] = gen
            items.append(
                (fid, preview, _THUMB_EDGE, source, str(card.filename or ""), fp)
            )
        if items:
            sched.request_visible(items)

    def _active_list(self) -> QListWidget:
        idx = self.tabs.currentIndex()
        if idx == 1:
            return self.list_suspicious
        if idx == 2:
            return self.list_undecided
        if idx == 3:
            return self.list_new_concept
        return self.list_undefined

    def _pool_key(self) -> str:
        idx = self.tabs.currentIndex()
        if idx == 1:
            return "suspicious"
        if idx == 2:
            return "undecided"
        if idx == 3:
            return "new_concept"
        return "undefined"

    def reload(self, *, blocking: bool | None = None) -> None:
        """Havuzları yenile. Görünür panelde async worker; testlerde sync.

        blocking=None → görünür değilse sync (pytest), görünürse worker.
        """
        if blocking is None:
            blocking = not bool(self.isVisible())
        self._reload_req += 1
        req = self._reload_req
        db_path = str(self.settings.db_path)
        cache_dir = str(getattr(self.settings, "cache_dir", "") or "")
        if blocking:
            pools, names = self._fetch_inbox_sync(db_path, cache_dir)
            self._apply_inbox(req, pools, names, batch=False)
            return
        self.btn_refresh.setEnabled(False)
        self.lbl_review.setText("Havuzlar yükleniyor…")
        # Önceki worker sonucunu ezmesin: req_id ile stale reddedilir.
        w = _TeachMeInboxWorker(db_path, cache_dir, req, parent=self)
        self._inbox_worker = w
        w.finished_ok.connect(self._on_inbox_loaded)
        w.failed.connect(self._on_inbox_failed)
        w.finished.connect(w.deleteLater)
        w.start()

    @staticmethod
    def _fetch_inbox_sync(
        db_path: str, cache_dir: str
    ) -> tuple[dict[str, list[TeachMeCard]], list[str]]:
        db = Database(db_path)
        pools = list_inbox_pools(db, db_path)
        pools.setdefault("new_concept", [])
        for cards in pools.values():
            for card in cards or []:
                enrich_teach_card_paths(card, db=db, cache_dir=cache_dir)
        names = _active_concept_names(db_path)
        return pools, names

    def _on_inbox_loaded(
        self, req_id: int, pools: object, names: object
    ) -> None:
        if int(req_id) != self._reload_req:
            return
        self.btn_refresh.setEnabled(True)
        self._apply_inbox(
            int(req_id),
            pools if isinstance(pools, dict) else {},
            list(names) if isinstance(names, list) else [],
            batch=True,
        )

    def _on_inbox_failed(self, req_id: int, message: str) -> None:
        if int(req_id) != self._reload_req:
            return
        self.btn_refresh.setEnabled(True)
        self.lbl_review.setText(f"Havuz yüklenemedi: {message}")

    def _apply_inbox(
        self,
        req_id: int,
        pools: dict[str, list[TeachMeCard]],
        concept_names: list[str],
        *,
        batch: bool = True,
    ) -> None:
        if int(req_id) != self._reload_req:
            return
        self._db = Database(self.settings.db_path)
        self._cards = {
            "undefined": list(pools.get("undefined") or []),
            "suspicious": list(pools.get("suspicious") or []),
            "undecided": list(pools.get("undecided") or []),
            "new_concept": list(pools.get("new_concept") or []),
        }
        self._fill_gen += 1
        gen = self._fill_gen
        self._thumb_wait_gen.clear()
        self._set_concept_names(concept_names)
        self.tabs.setTabText(0, f"Tanımsızlar ({len(self._cards['undefined'])})")
        self.tabs.setTabText(1, f"Şüpheliler ({len(self._cards['suspicious'])})")
        self.tabs.setTabText(2, f"Kararsızlar ({len(self._cards['undecided'])})")
        self.tabs.setTabText(3, f"Yeni Kavram ({len(self._cards['new_concept'])})")
        for w in (
            self.list_undefined,
            self.list_suspicious,
            self.list_undecided,
            self.list_new_concept,
        ):
            w.clear()
        pairs = (
            (self.list_undefined, self._cards["undefined"]),
            (self.list_suspicious, self._cards["suspicious"]),
            (self.list_undecided, self._cards["undecided"]),
            (self.list_new_concept, self._cards["new_concept"]),
        )
        if batch:
            for widget, cards in pairs:
                self._fill_list(widget, cards, gen, 0)
        else:
            for widget, cards in pairs:
                self._fill_list_sync(widget, cards, gen)
        self._update_review_detail()

    def _fill_list_sync(
        self,
        widget: QListWidget,
        cards: list[TeachMeCard],
        gen: int,
    ) -> None:
        """Test / küçük sync yol — tek seferde kart + ikon."""
        if gen != self._fill_gen:
            return
        cache_dir = str(getattr(self.settings, "cache_dir", "") or "")
        for card in cards:
            text = _card_list_text(card)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, int(card.file_id))
            item.setToolTip(
                f"{card.path}\n{card.category}\n{card.reason}"
            )
            item.setTextAlignment(
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            )
            item.setSizeHint(_card_item_size_hint(text))
            path = _thumb_path(card, cache_dir=cache_dir, db=None)
            item.setIcon(QIcon(_preview_pix(path)))
            widget.addItem(item)
            if not path:
                self._request_thumb_miss(card, gen)
        widget.doItemsLayout()

    def _set_concept_names(self, names: list[str]) -> None:
        self._concept_names = sorted_turkish(names or [])
        self.cmb_known.blockSignals(True)
        self.cmb_known.clear()
        self.cmb_known.addItem("Kayıtlı kavram…", "")
        for name in self._concept_names:
            if name:
                self.cmb_known.addItem(name, name)
        self.cmb_known.blockSignals(False)

    def _reload_concepts(self) -> None:
        """Refresh known labels from in-session cache; DB only if empty."""
        if self._concept_names:
            self._set_concept_names(self._concept_names)
            return
        # Cold path (tests / never reloaded) — keep off hot UI open path.
        self._set_concept_names(_active_concept_names(str(self.settings.db_path)))

    def _known_concepts_for_dialog(self) -> list[str]:
        """Reuse inbox-loaded names — no sync concepts() on Teach classify open."""
        if self._concept_names:
            return list(self._concept_names)
        # Fallback: read from hidden combo already filled by last reload.
        names: list[str] = []
        for i in range(self.cmb_known.count()):
            val = str(self.cmb_known.itemData(i) or "")
            if val:
                names.append(val)
        if names:
            return sorted_turkish(names)
        return []

    def _fill_from_combo(self, _idx: int = 0) -> None:
        val = str(self.cmb_known.currentData() or "")
        if val:
            self.txt_label.setText(val)

    def _fill_list(
        self,
        widget: QListWidget,
        cards: list[TeachMeCard],
        gen: int,
        start: int = 0,
    ) -> None:
        if gen != self._fill_gen:
            return
        end = min(int(start) + _CARD_BATCH, len(cards))
        for i in range(int(start), end):
            card = cards[i]
            text = _card_list_text(card)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, int(card.file_id))
            item.setToolTip(
                f"{card.path}\n"
                f"{card.category}\n"
                f"{card.reason}"
            )
            item.setTextAlignment(
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            )
            item.setSizeHint(_card_item_size_hint(text))
            widget.addItem(item)
        if end < len(cards):
            QTimer.singleShot(
                0, lambda: self._fill_list(widget, cards, gen, end)
            )
            return
        QTimer.singleShot(0, lambda: self._load_icons(widget, cards, 0, gen))

    def _load_icons(
        self,
        widget: QListWidget,
        cards: list[TeachMeCard],
        start: int,
        gen: int,
    ) -> None:
        if gen != self._fill_gen:
            return
        end = min(int(start) + _ICON_BATCH, len(cards))
        cache_dir = str(getattr(self.settings, "cache_dir", "") or "")
        for i in range(int(start), end):
            item = widget.item(i)
            if item is None:
                continue
            card = cards[i]
            path = _thumb_path(card, cache_dir=cache_dir, db=None)
            item.setIcon(QIcon(_preview_pix(path)))
            item.setSizeHint(
                _card_item_size_hint(item.text() or _card_list_text(card))
            )
            if not path:
                self._request_thumb_miss(card, gen)
        if end < len(cards):
            QTimer.singleShot(
                0, lambda: self._load_icons(widget, cards, end, gen)
            )
        else:
            widget.doItemsLayout()

    def _request_thumb_miss(self, card: TeachMeCard, gen: int) -> None:
        """Miss → batched request_visible for active-tab viewport (async, dedupe)."""
        if gen != self._fill_gen:
            return
        # Individual card only triggers a flush; inactive tabs are ignored inside.
        self._schedule_active_viewport_thumbs()

    def _on_teach_thumb_ready(
        self, file_id: int, image: object, size: int
    ) -> None:
        fid = int(file_id or 0)
        wait_gen = self._thumb_wait_gen.get(fid)
        if wait_gen is None or wait_gen != self._fill_gen:
            return
        if image is None or not isinstance(image, QImage) or image.isNull():
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            return
        scaled = pix.scaled(
            _THUMB_EDGE,
            _THUMB_EDGE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        icon = QIcon(scaled)
        for widget in (
            self.list_undefined,
            self.list_suspicious,
            self.list_undecided,
            self.list_new_concept,
        ):
            for i in range(widget.count()):
                item = widget.item(i)
                if item is None:
                    continue
                if int(item.data(Qt.ItemDataRole.UserRole) or 0) != fid:
                    continue
                item.setIcon(icon)
                item.setSizeHint(
                    _card_item_size_hint(item.text() or "")
                )


    def selected_ids(self) -> list[int]:
        out: list[int] = []
        for item in self._active_list().selectedItems():
            fid = int(item.data(Qt.ItemDataRole.UserRole) or 0)
            if fid > 0:
                out.append(fid)
        return out

    def _expanded_selected_ids(self) -> list[int]:
        """Teach/confirm: expand visual-family member_ids so one teach covers the group."""
        ids = self.selected_ids()
        if not ids:
            return []
        seen: set[int] = set()
        out: list[int] = []
        for fid in ids:
            card = self._card_by_id(fid)
            for mid in expand_member_ids(card, [fid]):
                if mid not in seen:
                    seen.add(mid)
                    out.append(mid)
        return out

    def _card_by_id(self, file_id: int) -> TeachMeCard | None:
        for pool in self._cards.values():
            for card in pool:
                if int(card.file_id) == int(file_id):
                    return card
                members = getattr(card, "member_ids", None) or []
                if int(file_id) in {int(m) for m in members}:
                    return card
        return None

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        fid = int(item.data(Qt.ItemDataRole.UserRole) or 0)
        selected = self.selected_ids()
        if fid > 0 and fid == self._last_click_id and selected == [fid]:
            self._open_preview(item)
        self._last_click_id = fid

    def _on_item_preview(self, item: QListWidgetItem) -> None:
        self._open_preview(item)

    def _open_preview(
        self,
        item: QListWidgetItem | None,
        *,
        exec_dialog: bool = True,
    ) -> ImagePreviewDialog | None:
        if item is None:
            return None
        fid = int(item.data(Qt.ItemDataRole.UserRole) or 0)
        card = self._card_by_id(fid)
        path = best_preview_path(card) if card else ""
        dlg = ImagePreviewDialog(
            path,
            title=f"Önizleme — {card.filename if card else fid}",
            parent=self,
        )
        if exec_dialog:
            if self._preview_open:
                return dlg
            self._preview_open = True
            try:
                dlg.exec()
            finally:
                self._preview_open = False
                self._keep_host_visible()
        return dlg

    def _update_review_detail(self) -> None:
        ids = self.selected_ids()
        if not hasattr(self, "lbl_review"):
            return
        if not ids:
            self.lbl_review.setText("Bir görsel seçin — öneri, güven ve gerekçe burada.")
            self._refresh_smart_undecided(None)
            return
        cards = [self._card_by_id(i) for i in ids]
        cards = [c for c in cards if c is not None]
        if not cards:
            self.lbl_review.setText(f"{len(ids)} görsel seçildi.")
            self._refresh_smart_undecided(None)
            return
        card = cards[0]
        conf_pct = int(round(float(card.confidence or 0) * 100))
        rivals = getattr(card, "rivals", None) or []
        rival_txt = ""
        if rivals:
            bits = []
            for item in rivals[:4]:
                if isinstance(item, (list, tuple)) and item:
                    lab = item[0]
                    sc = item[1] if len(item) > 1 else 0
                    bits.append(f"{lab} %{int(round(float(sc) * 100))}")
            rival_txt = " · ".join(bits)
        expanded = self._expanded_selected_ids()
        n = max(len(expanded), int(getattr(card, "cluster_size", 1) or 1))
        family_note = ""
        if int(getattr(card, "cluster_size", 1) or 1) > 1:
            family_note = (
                f"\nAile adayı: aynı aile mi? Öğret/Doğru → {n} dosya birlikte; "
                "Yanlış → grup bölünür."
            )
        self.lbl_review.setText(
            f"Önerilen kavram: {card.suggested or card.guess}\n"
            f"Güven: %{conf_pct}"
            + (f" · Alternatifler: {rival_txt}" if rival_txt else "")
            + f"\nGörsel sayısı: {n}\n"
            f"Neden: {card.reason}"
            + family_note
        )
        # Smart undecided: tek seçim + Kararsızlar sekmesi
        show_card = card if (
            self._pool_key() == "undecided" and len(ids) == 1
        ) else None
        self._refresh_smart_undecided(show_card)

    def _clear_smart_candidate_buttons(self) -> None:
        row = getattr(self, "smart_candidates_row", None)
        if row is None:
            return
        while row.count():
            item = row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _refresh_smart_undecided(self, card: TeachMeCard | None) -> None:
        """Kararsızlar: BU GÖRSEL NE? + aday butonları + Hiçbiri."""
        box = getattr(self, "smart_undecided", None)
        if box is None:
            return
        self._clear_smart_candidate_buttons()
        if card is None or self._pool_key() != "undecided":
            box.hide()
            return
        labels = undecided_candidate_labels(
            getattr(card, "rivals", None),
            str(getattr(card, "suggested", "") or ""),
        )
        # Thumb
        thumb = getattr(self, "lbl_smart_thumb", None)
        if thumb is not None:
            path = ""
            cache = str(getattr(self.settings, "cache_dir", "") or "")
            try:
                path = _thumb_path(card, cache_dir=cache)
            except Exception:
                path = ""
            if path:
                pix = _preview_pix(path)
                thumb.setPixmap(
                    pix.scaled(
                        72,
                        72,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                thumb.setPixmap(QPixmap())
        for lab in labels:
            btn = QPushButton(lab)
            btn.setToolTip(f"Tek tıkla öğret: {lab} (onay yok)")
            btn.clicked.connect(
                lambda _checked=False, name=lab: self._on_one_click_teach(name)
            )
            self.smart_candidates_row.addWidget(btn)
        self.smart_candidates_row.addStretch(1)
        box.show()

    def _select_first_undecided(self) -> None:
        w = self.list_undecided
        if w.count() <= 0:
            return
        self.tabs.setCurrentWidget(w)
        w.setCurrentRow(0)
        item = w.item(0)
        if item is not None:
            w.setCurrentItem(item)
            item.setSelected(True)

    def _on_one_click_teach(self, label: str) -> None:
        """Aday tık → anında mevcut Teach/Learning (confirm_reviews). Onay diyaloğu yok."""
        name = " ".join(str(label or "").strip().split())
        if not name:
            return
        ids = self._expanded_selected_ids()
        if not ids:
            return
        # Anlık UI geri bildirimi; öğrenme bir tur sonra (paint sonrası)
        self.lbl_review.setText(f"Öğretiliyor: {name}…")
        self._clear_smart_candidate_buttons()
        if hasattr(self, "smart_undecided"):
            self.smart_undecided.setEnabled(False)
        self._set_busy(True)

        def _finish() -> None:
            try:
                db = Database(self.settings.db_path)
                stats = confirm_reviews(
                    db, str(self.settings.db_path), ids, label=name
                )
            except Exception:
                logger.exception("one-click teach failed")
                stats = {"taught": 0}
            finally:
                self._set_busy(False)
                if hasattr(self, "smart_undecided"):
                    self.smart_undecided.setEnabled(True)
            self.taught.emit(name, int(stats.get("taught") or 0))
            self._keep_host_visible()
            # Kararsız kartını hemen kapat; sonraki öğeye geç (async reload gecikmesin)
            self.reload(blocking=True)
            self._select_first_undecided()

        QTimer.singleShot(0, _finish)

    def _on_hicbiri(self) -> None:
        """Hiçbiri → mevcut Teach (Şu…) diyaloğu; paralel öğrenme yok."""
        self._on_correct()

    def _on_confirm(self) -> None:
        ids = self._expanded_selected_ids()
        if not ids:
            return
        card = self._card_by_id(ids[0])
        label = str((card.suggested if card else "") or (card.guess if card else "") or "").strip()
        if " / " in label:
            label = label.split(" / ")[0].strip()
        if not label or label.lower() in ("belirsiz", "yeni kavram"):
            self._on_correct()
            return
        self._set_busy(True)
        try:
            db = Database(self.settings.db_path)
            stats = confirm_reviews(
                db, str(self.settings.db_path), ids, label=label
            )
        finally:
            self._set_busy(False)
        self.taught.emit(label, int(stats.get("taught") or 0))
        self._keep_host_visible()
        self.reload()

    def _on_reject(self) -> None:
        ids = self.selected_ids()
        if not ids:
            return
        path = str(self.settings.db_path)
        # Visual family: Yanlış = split group (not dismiss); ban fingerprint from reforming.
        family_cards = []
        for fid in ids:
            card = self._card_by_id(fid)
            if card is not None and int(getattr(card, "cluster_size", 1) or 1) > 1:
                family_cards.append(card)
        if family_cards:
            for card in family_cards:
                reject_and_split_family(path, card)
            self.reload()
            return
        reject_reviews(path, ids)
        self.reload()

    def _on_correct(self) -> None:
        ids = self._expanded_selected_ids()
        if not ids:
            return
        db = Database(self.settings.db_path)
        path = str(self.settings.db_path)
        snaps = [snapshot_file_classification(db, fid) for fid in ids]
        known = self._known_concepts_for_dialog()
        dlg = TeachClassifyDialog(
            self,
            db_path=path,
            overlay=common_overlay_from_snapshots(snaps),
            file_count=len(ids),
            change_summary="Şu: girdiğiniz kavram kalıcı öğrenilir.",
            concepts=known,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            self._keep_host_visible()
            return
        label = dlg.concept_label()
        if not label:
            return
        self._set_busy(True)
        try:
            stats = correct_reviews(
                db, path, ids, label, overlay=dlg.values()
            )
        finally:
            self._set_busy(False)
        self.taught.emit(label, int(stats.get("taught") or 0))
        self.reload()

    def _select_all(self) -> None:
        self._active_list().selectAll()

    def _clear_sel(self) -> None:
        self._active_list().clearSelection()

    def _host_dock(self) -> QDockWidget | None:
        w = self.parentWidget()
        while w is not None:
            if isinstance(w, QDockWidget):
                return w
            w = w.parentWidget()
        return None

    def _keep_host_visible(self) -> None:
        dock = self._host_dock()
        if dock is None:
            return
        if dock.isHidden() or not dock.isVisible():
            dock.show()
            dock.raise_()

    def _apply_teach(self, ids: list[int], label: str, overlay: dict | None = None) -> dict:
        self._set_busy(True)
        try:
            db = Database(self.settings.db_path)
            stats = teach_files(
                db, str(self.settings.db_path), ids, label, overlay=overlay
            )
        finally:
            self._set_busy(False)
        self.taught.emit(label, int(stats.get("taught") or 0))
        self._keep_host_visible()
        self.reload()
        return stats

    def _set_busy(self, busy: bool) -> None:
        for btn in (
            self.btn_select_all,
            self.btn_clear,
            self.btn_confirm,
            self.btn_reject,
            self.btn_correct,
            self.btn_teach,
            self.btn_dismiss,
            self.btn_refresh,
        ):
            btn.setEnabled(not busy)
        if hasattr(self, "btn_hicbiri"):
            self.btn_hicbiri.setEnabled(not busy)
        if hasattr(self, "smart_undecided"):
            self.smart_undecided.setEnabled(not busy)

    def _on_teach(self) -> None:
        ids = self._expanded_selected_ids()
        if not ids:
            return
        db = Database(self.settings.db_path)
        path = str(self.settings.db_path)
        snaps = [snapshot_file_classification(db, fid) for fid in ids]
        store = UserFeedbackStore(db)
        overlays = []
        for snap in snaps:
            ov = store.metadata_overlay_for_file(int(snap["file_id"])) or {}
            overlays.append({**snap, **ov} if ov else snap)
        common = common_overlay_from_snapshots(overlays)
        known = self._known_concepts_for_dialog()
        dlg = TeachClassifyDialog(
            self,
            db_path=path,
            overlay=common,
            file_count=len(ids),
            change_summary=summarize_classification_changes(snaps, common),
            concepts=known,
        )
        dlg.cmb_known.currentIndexChanged.connect(
            lambda _i: dlg.lbl_changes.setText(
                summarize_classification_changes(snaps, dlg.values())
            )
        )
        for combo in (
            dlg.cmb_parent,
            dlg.cmb_child,
            dlg.cmb_family,
            dlg.cmb_color,
            dlg.cmb_brand,
        ):
            combo.editTextChanged.connect(
                lambda _t, d=dlg, s=snaps: d.lbl_changes.setText(
                    summarize_classification_changes(s, d.values())
                )
            )
        if dlg.exec() != dlg.DialogCode.Accepted:
            self._keep_host_visible()
            return
        overlay = dlg.values()
        dlg.lbl_changes.setText(summarize_classification_changes(snaps, overlay))
        label = dlg.concept_label()
        if not label:
            return
        self._apply_teach(ids, label, overlay)

    def _on_dismiss(self) -> None:
        ids = self.selected_ids()
        if not ids:
            return
        dismiss_files(str(self.settings.db_path), ids)
        self.reload()


class TeachClassifyDialog(ResultMetadataDialog):
    """Düzenle formu + kavram satırı. Toplu öğret; tek dosya Düzenle değildir."""

    def __init__(
        self,
        parent=None,
        *,
        db_path: str = "",
        overlay: dict | None = None,
        file_count: int = 1,
        change_summary: str = "",
        concepts: list[str] | None = None,
    ):
        super().__init__(parent, result=None, db_path=db_path, overlay=overlay)
        self.setWindowTitle("Seçilenlere öğret — sınıflandır")
        n = max(1, int(file_count or 1))
        self.lbl_intro.setText(
            f"{n} görsel seçildi. Düzenle ile aynı kategoriler uygulanır; "
            "boş alanlar mevcut metadata'yı korur. Kaydet hem dosya sınıflandırmasını "
            "günceller hem doğrulanmış örnek olarak sisteme öğretir."
        )
        self.btn_teach.hide()
        self.btn_save.setText("Kaydet")
        layout = getattr(self, "_body_layout", None) or self.layout()
        known_row = QWidget()
        known_l = QHBoxLayout(known_row)
        known_l.setContentsMargins(0, 0, 0, 0)
        self.txt_concept = QLineEdit()
        self.txt_concept.setPlaceholderText("Kavram adı (ör. Zincir, Parfüm)")
        self.cmb_known = QComboBox()
        self.cmb_known.addItem("Kayıtlı kavram…", "")
        for name in sorted_turkish(concepts or []):
            self.cmb_known.addItem(name, name)
        self.cmb_known.currentIndexChanged.connect(self._fill_from_combo)
        known_l.addWidget(QLabel("Kavram:"))
        known_l.addWidget(self.txt_concept, 1)
        known_l.addWidget(self.cmb_known)
        self.lbl_changes = QLabel(change_summary or "Boş alanlar mevcut dosya bilgilerini korur.")
        self.lbl_changes.setWordWrap(True)
        layout.insertWidget(1, known_row)
        layout.insertWidget(2, self.lbl_changes)
        from ui.responsive_dialog import fit_dialog_to_available_screen

        fit_dialog_to_available_screen(self, min_width=440, prefer_width=520)

    def _fill_from_combo(self, _idx: int = 0) -> None:
        val = str(self.cmb_known.currentData() or "")
        if val:
            self.txt_concept.setText(val)

    def concept_label(self) -> str:
        typed = self.txt_concept.text().strip()
        known = str(self.cmb_known.currentData() or "")
        if typed:
            return typed
        if known:
            return known
        vals = self.values()
        return str(vals.get("child") or vals.get("parent") or "").strip()
