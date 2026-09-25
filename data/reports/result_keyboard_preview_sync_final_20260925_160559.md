# Keyboard + Detail Preview + Canvas Hover — Final Sync

**Date:** 2026-09-25 16:05:59  
**Commit:** `fix: sync keyboard results with detail preview and canvas hover`

## Flow

```
↑/↓ → navigate_by → card_selected → _select_result → result_selected
    → _show_result_detail → set_selection_* (ends list-hover)
    → right canvas = ACTIVE RESULT

List hover → temporary override (show_thumbnail)
Selection/keyboard → end_list_hover_override (active wins)

Canvas hover → DetailPreviewHoverOverlay parent=panel, geo=contentsRect (red zone)
    → hi-res FeaturePreview/detail/selection cache, maximize contain-fit
Canvas leave → hide overlay → normal active preview
```

## Root fixes

1. **Keyboard stuck on old preview:** `set_selection_pixmap` skipped paint while `_list_hovering`. Now selection clears list-hover and paints immediately.
2. **Canvas hover flaky:** `lbl_image` stole mouse → panel `leaveEvent` hid overlay. Children now `WA_TransparentForMouseEvents`.
3. **Red zone:** overlay = full panel `contentsRect`; hi-res (≥512) fills canvas.

## Tests

42 passed (sync final + position + keyboard + second-level).

## Real UI

Hover right preview → hi-res fills red Result Detail preview area; no center popup. ↑/↓ updates right preview 1→2→3…
