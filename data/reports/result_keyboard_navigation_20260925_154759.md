# Result List Keyboard Navigation

**Date:** 2026-09-25 15:47:59  
**Commit:** `feat: add keyboard navigation for result list`

## Structure

- `VirtualResultsList` — virtual scroll + recycled `ResultCard`s
- Active inspect selection: `_selected_file_id` + `card_selected` → `ResultsPanel._select_result` → `result_selected` → `MainWindow._show_result_detail`
- Checkbox multi-select: separate `PreviewSelectionStore` (untouched by arrows)

## Keyboard link

| Key | Action |
|-----|--------|
| ↓ | `navigate_by(+1)` next card |
| ↑ | `navigate_by(-1)` previous card |
| First/last | no-op (selection unchanged) |

- `VirtualResultsList.keyPressEvent` when list focused
- `ResultsPanel.keyPressEvent` forwards ↑/↓ when focus is inside panel and **not** in `QLineEdit` / `QTextEdit` / `QPlainTextEdit` / spin / editable combo
- `_select_result` calls `scroll.setFocus()` so post-click arrows work

## Preview sync

`navigate_by` → `card_selected` → existing `_select_result` / `_show_result_detail` → `FixedHoverPreviewPanel.set_selection_pixmap` (normal + large overlay source). List-hover path unchanged.

## Auto-scroll

`ensure_card_pos_visible` scrolls so the active row intersects the viewport.

## Tests

12 passed — `tests/test_result_keyboard_navigation.py`

## Real Windows UI

Click result → ↓/↑ cycles selection + detail preview; ends clamp; search box ignores arrows; checkboxes unchanged.
