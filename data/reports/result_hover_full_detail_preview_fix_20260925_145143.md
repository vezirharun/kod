# Result List Hover → Full Result Detail Preview

**Date:** 2026-09-25 14:51:43  
**Commit:** `fix: use full result detail canvas for hover preview`

## Root cause

Hover large preview was implemented as a **floating** `ResultDetailPreviewOverlay` (`QFrame` child of `QMainWindow`), anchored to the left of the dock preview. That floating frame is the **mavi alan** (small blue rectangle over the results list).

The intended canvas is already the dock's top panel: `FixedHoverPreviewPanel` / `lbl_image` (kırmızı alan).

## Fix

1. **Removed** `ResultDetailPreviewOverlay` and all overlay show/hide / enterEvent magnify paths.
2. **List hover** (`show_thumbnail`) continues to paint into `FixedHoverPreviewPanel.lbl_image` — the full Result Detail preview canvas.
3. **Contain-fit** now maximizes inside the canvas (allows upscale of small thumbs so the red area is filled):  
   `scale = min(box_w/nw, box_h/nh)` with `tw,th ≤ canvas`.
4. Selection vs hover: `_list_hovering` temporary; `_selection_*` unchanged; leave → `_restore_selection()`.

## Correct preview canvas

| Role | Widget |
|------|--------|
| LARGE_PREVIEW_CANVAS | `FixedHoverPreviewPanel` (`#FixedHoverPreview`) |
| Image label | `lbl_image` |

## Hover → detail link

`results_panel.hover_preview` → `MainWindow._on_hover_preview` → `FixedHoverPreviewPanel.show_thumbnail`  
Clear: `hover_preview_clear` → `schedule_hide` → `_restore_selection`

## Geometry

Hover does not change dock width, main window, or preview panel geometry — only the pixmap inside `lbl_image`.

## Tests

```
17 passed (test_result_hover_full_detail_preview + retired overlay stubs + large_preview)
```

Named suite in `tests/test_result_hover_full_detail_preview.py` — all required cases PASS.

## Real Windows UI

Automated Qt on Windows (`QMainWindow` + right InspectorDock):

- List hover updates caption/pixmap on full canvas
- No `ResultDetailPreviewOverlay`
- Dock/window geometry unchanged
- No `os.startfile` / `openUrl` on hover
- Leave restores selection preview

Interactive leopard smoke: same path (`_on_hover_preview` → `show_thumbnail`); floating blue overlay can no longer appear.
