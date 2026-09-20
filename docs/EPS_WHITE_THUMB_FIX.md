# EPS/AI white list-thumbnail fix

## Problem
List UI served `cache/thumbnails/{hash}.webp` for `.eps`/`.ai` without the blank/white gate.
Detail UI already used `cache/feature_previews/{hash}_fp.webp` (gated). Result: ~150-byte white stubs in the list vs 80KB+ valid feature previews in detail.

## Scope (minimal)
Touches only list/detail path resolution and thumbnail HIT reuse:

| File | Change |
|------|--------|
| `core/thumb_resolve.py` | `_is_eps_ai`, `_thumb_usable_for_eps` / `is_eps_ai_thumb_usable`; EPS/AI prefer valid FP, reject tiny/white thumbs |
| `core/preview_renderer.py` | `resolve_display_image_path`: for `.eps`/`.ai` check FP/`get_existing` before gated thumb (no size>0 early return) |
| `core/thumbnailer.py` | `create` / `create_from_existing_preview`: on HIT for EPS/AI run gate; unlink invalid stubs and fall through / overwrite from FP |

**Unchanged:** Ghostscript `-dEPSCrop` / blank rejection inside `render_preview_for_index` / `_try_ghostscript` (reuses `_eps_ai_preview_gate` only). JPG/PNG/TIFF thumb-first behavior. Indexer, JobStore, DINO/CLIP, ranking, Archive Health, Customer Memory.

## Behavior (.eps / .ai only)
1. Prefer valid feature preview (`lookup_valid_preview` / explicit FP path).
2. Serve thumbnail only if usable (exists, size ≥ 512 bytes, and `_eps_ai_preview_gate` / blank check when importable).
3. If thumb is white/tiny but FP is valid → return FP so the list can show it.
4. Thumbnailer HIT on a bad stub deletes it and regenerates via Preview Pool SSOT / existing gated render path.

## Tests
`tests/test_eps_list_thumb.py` — tempfile + gate stubs; no PySide/Ghostscript.

```bash
cd /workspace/vezir_audit/eps_fix && PYTHONPATH=. python3 -m unittest tests.test_eps_list_thumb -v
```
