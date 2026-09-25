# Result Detail — Dual Normal + Hover Large Preview (Final)

**Date:** 2026-09-25 14:36:31  
**Commit message:** `fix: finalize dual normal and hover preview`

## Root cause

Earlier hover-magnify paths risked mutating the docked normal preview (in-canvas magnify / refit on enter). The correct model is two widgets:

1. **NormalPreview** (`FixedHoverPreviewPanel`) — sticky top of Result Detail; geometry frozen on hover.
2. **LargeHoverPreviewOverlay** (`ResultDetailPreviewOverlay`) — child of `QMainWindow`; shown only while the hover zone (normal OR large) is active.

## Architecture

| Role | Class | Lifetime |
|------|--------|----------|
| Normal preview | `FixedHoverPreviewPanel` | Always visible in Result Detail dock |
| Large overlay | `ResultDetailPreviewOverlay` | Shown on normal enter; hidden after leave of both |

### Hover event chain

1. Mouse **enters** normal → `_overlay_hide_timer.stop()` → `_show_overlay()` (no `_apply_fit_image`).
2. Mouse **leaves** normal → delayed hide timer starts (gap for crossing to overlay).
3. Mouse **enters** large overlay → `_overlay_zone_enter()` stops timer; overlay stays open.
4. Mouse **leaves** large overlay → `_overlay_zone_leave()` starts timer → hide.
5. Leave both zones → `LargeHoverPreviewOverlay.hide()`.

### Overlay geometry

- Anchor: `large.right ≈ normal.left - gap` (`gap = 8`).
- Prefer left of normal (over results list); never expand Result Detail dock / main window.
- Contain-fit scale of existing `_selection_pix` only.
- Clamp to main-window client rect.

### Source policy

Uses existing selection pixmap (`_selection_pix` / FeaturePreview path). No DINO/CLIP/NAS/TIFF re-decode/AI open on hover. Overlay does not open external files.

## Geometry assertion (critical)

`normal_rect_before == normal_rect_after` on hover — covered by  
`test_normal_preview_geometry_unchanged_on_hover`.

## Code touched (scope)

- `ui/fixed_hover_preview.py` — enter shows overlay only; clamp geometry; comments locking normal fit.
- `tests/test_result_detail_dual_preview.py` — dual-preview regression suite.

**Not touched:** Search, Ranking, DINO, CLIP, FAISS, Indexing, Learning, Pattern DNA, Thumbnail pipeline, JobStore, Customer Memory, Category Filter, Görseli Aç, Klasörde Aç.

## Test results

```
pytest tests/test_result_detail_dual_preview.py tests/test_result_detail_hover_magnify.py -q
28 passed in ~118s
```

| Test | Result |
|------|--------|
| test_normal_preview_geometry_unchanged_on_hover | PASS |
| test_large_preview_overlay_opens_on_hover | PASS |
| test_large_preview_overlay_closes_after_hover_zone_leave | PASS |
| test_large_preview_stays_open_between_normal_and_overlay | PASS |
| test_large_preview_does_not_resize_result_detail | PASS |
| test_large_preview_does_not_resize_main_window | PASS |
| test_large_preview_does_not_open_external_file | PASS |
| test_large_preview_contain_fit | PASS |
| test_large_preview_inside_main_window | PASS |
| (+ hover_magnify suite) | PASS |

## Real Windows UI

Automated Qt UI (QMainWindow + right InspectorDock + real enter/leave events) verified:

- Normal geometry unchanged before/after hover.
- Overlay opens left of normal, inside main window, contain-fit.
- Overlay stays across normal→large transition; hides after zone leave.
- No `os.startfile` / `QDesktopServices.openUrl` on hover.

Interactive leopard-search smoke (manual): same dual behavior expected — normal dock preview unchanged; large overlay only.

## Acceptance

- **NORMAL:** right panel normal preview unchanged.
- **HOVER:** same normal preview + separate large overlay to its left.
- Normal never grows; leave both → overlay hides.
