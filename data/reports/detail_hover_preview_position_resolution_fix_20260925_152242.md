# Detail Hover Preview — Position + High Resolution Fix

**Date:** 2026-09-25 15:22:42  
**Commit:** `fix: align detail hover preview and use high resolution source`

## Root cause (audited)

### A–D Position

| Question | Answer |
|----------|--------|
| A Widget | `DetailPreviewHoverOverlay` |
| B Parent | **was** `QMainWindow` |
| C Geometry | main-window / central-widget coords; placed LEFT of dock then clamped |
| D Why center | `max_w ≈ 55% central` + clamp → overlay floated over results/center |

### E–H Resolution

| Question | Answer |
|----------|--------|
| E Source | `_source_pix` (whatever was last fitted) |
| F Size | often list-thumb / small cache when `_source_pix` dominated |
| G FeaturePreview | yes — `request_detail_preview(..., size=1024)` + `peek_detail_image` |
| H Why unused | overlay ignored detail cache / preferred current `_source_pix`; then **upscaled into huge central box** → pixelation |

## Fix (minimal)

1. **Parent** = `FixedHoverPreviewPanel` (preview viewport), not main window.
2. **Geometry** = `lbl_image.geometry()` (kırmızı canvas).
3. **Source** = max pixel-area among `_selection_pix`, `_source_pix`, `peek_detail_image`, `peek_image`.
4. **Scale** = contain-fit with **upscale cap 1.0** (no blow-up of low-res thumbs).

## Tests

28 passed (`test_detail_hover_position_resolution` + second-level + list-hover).

## Real Windows UI

Overlay maps to canvas top-left in window coords; not central center. Hi-res selection preferred over 256 thumb.
