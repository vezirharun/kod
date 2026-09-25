# Detail Preview — Second-Level Hover Magnify

**Date:** 2026-09-25 15:07:53  
**Commit:** `fix: add detail preview hover magnify`

## Hover chain (3 states)

| State | Trigger | Behavior |
|-------|---------|----------|
| 1 Normal | no list hover | selection pixmap in dock canvas |
| 2 List hover | mouse on result card | same canvas shows hovered result (unchanged) |
| 3 Detail hover | mouse on dock preview | **new** `DetailPreviewHoverOverlay` left of canvas |

## Gap that was fixed

STATE 1–2 worked after prior commit. STATE 3 (hover on right preview → larger overlay) was missing after overlay removal.

## Widgets

- Canvas: `FixedHoverPreviewPanel` / `lbl_image` (geometry frozen on STATE 3)
- Magnify: `DetailPreviewHoverOverlay` (QMainWindow child)
- Zone: normal OR overlay; delayed hide timer (120ms)

## Overlay geometry

- Prefer left of normal preview (`large.right ≈ normal.left - 8`)
- Max box: `max(space_left, 55% client width, 240)` × `70% client height`
- Contain-fit from `_source_pix` (no AI/NAS/file open)
- Clamp inside main window client

## Tests

```
20 passed — test_detail_preview_second_level_hover + test_result_hover_full_detail_preview
```

## Real Windows UI

Qt QMainWindow + central + right InspectorDock:

- Overlay opens on detail enter; larger than canvas
- Normal preview / dock / window geometry unchanged
- Overlay stays across canvas→overlay move; hides after both leave
- List hover still canvas-only (overlay closed)
- No external file open
