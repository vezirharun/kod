"""Virtual scrolling + widget recycling for result cards.

Only VISIBLE_POOL_SIZE cards exist regardless of result count (261 / 1000+).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QScrollArea, QWidget

from core.dynamic_groups import QueryContext
from core.search_engine import SearchResult
from ui.result_card import (
    VIEW_CARD,
    VIEW_COMPACT,
    VIEW_LARGE,
    VIEW_LIST,
    ResultCard,
)

RANKED_VIEW = VIEW_LIST
VISIBLE_POOL_SIZE = 15
HEADER_HEIGHT = 34
_ROW_HEIGHTS = {VIEW_CARD: 108, VIEW_LIST: 88, VIEW_COMPACT: 64, VIEW_LARGE: 140}


class VirtualResultsList(QScrollArea):
    """Recycle ResultCard widgets on scroll — no append/delete storm."""

    card_selected = Signal(object)
    open_folder = Signal(int)
    open_file = Signal(int)
    hover_preview = Signal(int, str, str, str)
    hover_preview_clear = Signal()
    near_bottom = Signal()
    scrolled = Signal()
    preview_menu_requested = Signal(object, object)
    preview_click = Signal(object, object)
    preview_checkbox = Signal(object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Absolute child geometry — resizable=True viewport yüksekliğine sıkıştırır.
        self.setWidgetResizable(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view_mode = VIEW_LARGE
        self._query_ctx = QueryContext()
        self._simple_mode = True
        self._ranked = False
        self._selected_file_id = 0
        self._preview_selected_fn = None

        self._entries: list[tuple] = []
        self._row_heights: list[int] = []
        self._row_offsets: list[int] = []
        self._total_height = 0

        self._content = QWidget()
        self._content.setMinimumHeight(0)
        self.setWidget(self._content)

        self._pool: list[ResultCard] = []
        self._header_pool: list[QLabel] = []
        self._slot_index: list[int] = [-1] * VISIBLE_POOL_SIZE
        self._slot_header_index: list[int] = [-1] * 8
        self._layout_pending = False
        self._visible_signature: tuple[int, int] = (-1, -1)

        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self._build_pool()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)

    def _build_pool(self) -> None:
        for _ in range(VISIBLE_POOL_SIZE):
            card = ResultCard(
                _dummy_result(),
                view_mode=self._view_mode,
                simple_mode=self._simple_mode,
            )
            card.hide()
            card.selected.connect(self.card_selected.emit)
            card.open_folder.connect(self.open_folder.emit)
            card.open_file.connect(self.open_file.emit)
            card.hover_entered.connect(self.hover_preview.emit)
            card.hover_left.connect(self.hover_preview_clear.emit)
            card.preview_menu_requested.connect(self.preview_menu_requested.emit)
            card.preview_click.connect(self.preview_click.emit)
            card.preview_checkbox.connect(self.preview_checkbox.emit)
            self._pool.append(card)
            card.setParent(self._content)
        for _ in range(8):
            hdr = QLabel(self._content)
            hdr.hide()
            hdr.setStyleSheet(
                "font-weight:bold;color:#e2e8f0;background:#2d3544;"
                "padding:6px 8px;border-radius:4px;"
            )
            self._header_pool.append(hdr)

    def set_context(
        self,
        *,
        view_mode: str,
        query_ctx: QueryContext,
        simple_mode: bool,
        ranked: bool,
    ) -> None:
        self._view_mode = view_mode
        self._query_ctx = query_ctx
        self._simple_mode = simple_mode
        self._ranked = ranked

    def set_selected_file_id(self, file_id: int) -> None:
        self._selected_file_id = int(file_id or 0)
        for card in self._pool:
            if card.isVisible():
                card.set_selected(int(card.result.file_id) == self._selected_file_id)

    def set_preview_selected_fn(self, fn) -> None:
        self._preview_selected_fn = fn

    def _preview_is_selected(self, file_id: int) -> bool:
        fn = self._preview_selected_fn
        if fn is None:
            return False
        try:
            return bool(fn(int(file_id)))
        except Exception:
            return False

    def _content_width(self) -> int:
        return max(200, self.viewport().width() - 12)

    def _schedule_layout(self) -> None:
        if self._layout_pending:
            return
        self._layout_pending = True
        QTimer.singleShot(0, self._deferred_layout)

    def _deferred_layout(self) -> None:
        self._layout_pending = False
        self._layout_visible()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_layout()

    def set_entries(self, entries: list[tuple], *, preserve_scroll: bool = False) -> None:
        """entries: ('header', text, style) | ('card', SearchResult, rank)"""
        prev = self.verticalScrollBar().value() if preserve_scroll else 0
        self._entries = list(entries)
        self._slot_index = [-1] * VISIBLE_POOL_SIZE
        self._visible_signature = (-1, -1)
        self._rebuild_geometry()
        if preserve_scroll:
            self.verticalScrollBar().setValue(min(prev, self.verticalScrollBar().maximum()))
        else:
            self.verticalScrollBar().setValue(0)
        if self.viewport().height() > 0 and self._content_width() > 0:
            self._layout_visible()
        else:
            self._schedule_layout()
        self.scrolled.emit()

    def refresh_layout(self) -> None:
        """Viewport boyutu hazir oldugunda kartlari yeniden yerlestir."""
        self._layout_visible()

    def clear_entries(self) -> None:
        self._entries = []
        self._row_heights = []
        self._row_offsets = []
        self._total_height = 0
        self._content.setFixedHeight(0)
        for card in self._pool:
            card.hide()
        for hdr in self._header_pool:
            hdr.hide()
        self._slot_index = [-1] * VISIBLE_POOL_SIZE
        self._slot_header_index = [-1] * len(self._header_pool)

    def visible_cards(self) -> list[ResultCard]:
        return [c for c in self._pool if c.isVisible()]

    def cards_by_id(self) -> dict[int, ResultCard]:
        out: dict[int, ResultCard] = {}
        for card in self._pool:
            if card.isVisible():
                out[int(card.result.file_id)] = card
        return out

    def card_count(self) -> int:
        return sum(1 for e in self._entries if e[0] == "card")

    def _card_positions(self) -> list[int]:
        """Entry indices that are result cards (skip headers)."""
        return [i for i, e in enumerate(self._entries) if e[0] == "card"]

    def _selected_card_pos(self) -> int:
        """Index into _card_positions() for current selection, or -1."""
        positions = self._card_positions()
        if not positions:
            return -1
        fid = int(self._selected_file_id or 0)
        for pos, entry_i in enumerate(positions):
            result = self._entries[entry_i][1]
            if int(getattr(result, "file_id", 0) or 0) == fid:
                return pos
        return -1

    def result_at_card_pos(self, pos: int):
        positions = self._card_positions()
        if pos < 0 or pos >= len(positions):
            return None
        return self._entries[positions[pos]][1]

    def ensure_card_pos_visible(self, pos: int) -> None:
        positions = self._card_positions()
        if pos < 0 or pos >= len(positions):
            return
        entry_i = positions[pos]
        if entry_i >= len(self._row_offsets):
            return
        y = int(self._row_offsets[entry_i])
        h = int(self._row_heights[entry_i] if entry_i < len(self._row_heights) else 80)
        bar = self.verticalScrollBar()
        view_h = max(1, int(self.viewport().height() or self.height() or 1))
        top = int(bar.value())
        bottom = top + view_h
        target = top
        if y < top:
            target = max(0, y - 8)
        elif y + h > bottom:
            target = max(0, y + h - view_h + 8)
        else:
            return
        bar.setValue(min(target, bar.maximum()))
        self._layout_visible()

    def navigate_by(self, delta: int) -> bool:
        """Move active/selected card by delta. Returns True if selection changed."""
        if delta == 0:
            return False
        positions = self._card_positions()
        if not positions:
            return False
        cur = self._selected_card_pos()
        if cur < 0:
            # No selection yet — Down selects first, Up selects last.
            new_pos = 0 if delta > 0 else len(positions) - 1
        else:
            new_pos = cur + int(delta)
            if new_pos < 0 or new_pos >= len(positions):
                return False  # clamp at ends — no change
        result = self.result_at_card_pos(new_pos)
        if result is None:
            return False
        new_fid = int(getattr(result, "file_id", 0) or 0)
        if new_fid == int(self._selected_file_id or 0) and cur == new_pos:
            return False
        self.set_selected_file_id(new_fid)
        self.ensure_card_pos_visible(new_pos)
        self.card_selected.emit(result)
        return True

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Down:
            if self.navigate_by(1):
                event.accept()
                return
            event.accept()
            return
        if key == Qt.Key.Key_Up:
            if self.navigate_by(-1):
                event.accept()
                return
            event.accept()
            return
        super().keyPressEvent(event)

    def _rebuild_geometry(self) -> None:
        heights: list[int] = []
        offsets: list[int] = []
        y = 0
        for entry in self._entries:
            kind = entry[0]
            if kind == "header":
                h = HEADER_HEIGHT
            else:
                view = RANKED_VIEW if self._ranked and entry[2] else self._view_mode
                h = _ROW_HEIGHTS.get(view, 108)
            offsets.append(y)
            heights.append(h)
            y += h + 6
        self._row_heights = heights
        self._row_offsets = offsets
        self._total_height = max(y, 1)
        self._content.setFixedHeight(self._total_height)
        self._content.setMinimumWidth(self._content_width())
        self._content.resize(self._content_width(), self._total_height)

    def _compute_visible_window(self) -> tuple[int, int, list[int], list[int], tuple[int, int]]:
        scroll_y = self.verticalScrollBar().value()
        view_h = self.viewport().height()
        top = scroll_y
        bottom = scroll_y + view_h

        first = 0
        for i, off in enumerate(self._row_offsets):
            if off + self._row_heights[i] >= top:
                first = max(0, i - 1)
                break
        last = len(self._entries) - 1
        for i, off in enumerate(self._row_offsets):
            if off > bottom:
                last = min(len(self._entries) - 1, i + 1)
                break

        visible_indices = list(range(first, last + 1))
        card_indices = [i for i in visible_indices if self._entries[i][0] == "card"]
        header_indices = [i for i in visible_indices if self._entries[i][0] == "header"]
        signature = (
            card_indices[0] if card_indices else -1,
            card_indices[min(len(card_indices), VISIBLE_POOL_SIZE) - 1]
            if card_indices
            else -1,
        )
        return first, last, card_indices, header_indices, signature

    def _on_scroll(self, value: int) -> None:
        if not self._entries:
            return
        _first, _last, card_indices, header_indices, signature = self._compute_visible_window()
        changed = False
        if signature != self._visible_signature:
            changed = self._layout_visible(
                precomputed=(card_indices, header_indices, signature)
            )
        if changed:
            self.scrolled.emit()
        bar = self.verticalScrollBar()
        if value >= max(0, bar.maximum() - 120):
            self.near_bottom.emit()

    def _layout_visible(
        self,
        *,
        precomputed: tuple[list[int], list[int], tuple[int, int]] | None = None,
    ) -> bool:
        if not self._entries:
            return False
        view_w = self._content_width()
        view_h = self.viewport().height()
        if view_h < 8 or view_w < 50:
            self._schedule_layout()
            return False
        if precomputed is None:
            _first, _last, card_indices, header_indices, new_signature = (
                self._compute_visible_window()
            )
        else:
            card_indices, header_indices, new_signature = precomputed

        self._content.setUpdatesEnabled(False)
        try:
            visible_header_count = min(len(header_indices), len(self._header_pool))
            for slot, idx in enumerate(header_indices[:visible_header_count]):
                hdr = self._header_pool[slot]
                _, text, style = self._entries[idx]
                hdr.setText(str(text))
                if style == "near":
                    hdr.setStyleSheet(
                        "font-weight:bold;color:#fbbf24;padding:6px;margin-top:4px;"
                    )
                else:
                    hdr.setStyleSheet(
                        "font-weight:bold;color:#e2e8f0;background:#2d3544;"
                        "padding:6px 8px;border-radius:4px;margin-top:4px;"
                    )
                hdr.setGeometry(
                    4, self._row_offsets[idx], view_w, HEADER_HEIGHT
                )
                hdr.show()
                self._slot_header_index[slot] = idx
            for slot in range(visible_header_count, len(self._header_pool)):
                self._header_pool[slot].hide()
                self._slot_header_index[slot] = -1

            visible_card_count = min(len(card_indices), VISIBLE_POOL_SIZE)
            for slot, idx in enumerate(card_indices[:visible_card_count]):
                card = self._pool[slot]
                _, result, rank = self._entries[idx]
                view = RANKED_VIEW if self._ranked and rank else self._view_mode
                same_fid = int(card.result.file_id) == int(result.file_id)
                if self._slot_index[slot] != idx or not same_fid:
                    card.rebind(
                        result,
                        view_mode=view,
                        query_ctx=self._query_ctx,
                        rank=int(rank or 0),
                        simple_mode=self._simple_mode,
                    )
                    self._slot_index[slot] = idx
                else:
                    card.result = result
                    card._rank = int(rank or 0)
                    card._update_rank_badge()
                    card._update_labels()
                    if card._thumbnail_pixmap.isNull():
                        card._try_cached_thumbnail()
                card.set_selected(int(result.file_id) == self._selected_file_id)
                card.set_preview_checked(
                    self._preview_is_selected(int(result.file_id))
                )
                h = self._row_heights[idx]
                y_off = self._row_offsets[idx]
                card.setGeometry(4, y_off, view_w, h)
                card.raise_()
                card.show()
            for slot in range(visible_card_count, len(self._pool)):
                self._pool[slot].hide()
                self._slot_index[slot] = -1
        finally:
            self._content.setUpdatesEnabled(True)
        changed = new_signature != self._visible_signature
        self._visible_signature = new_signature
        return changed

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        w = self._content_width()
        if self._total_height > 0:
            self._content.resize(w, self._total_height)
        self._layout_visible()


def _dummy_result() -> SearchResult:
    return SearchResult(
        file_id=0,
        path="",
        filename="",
        customer="",
        thumbnail_path="",
        score=0.0,
        score_percent=0.0,
    )
