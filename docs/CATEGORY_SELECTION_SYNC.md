# Category selection auto-sync (Teach / Edit)

## Problem

`ResultMetadataDialog` parent combo (`cmb_parent`) uses a semantic catalog from
`flatten_children_catalog`, whose displays look like `insan / erkek`.

If the user picks that display:

1. Ana kategori text becomes the **full path**
2. Alt stays empty
3. Tags stay empty
4. `values()` did `path = f"{parent}/{child}"` → identity corrupted
   (e.g. `"insan / erkek"` as parent alone, or `"insan / erkek/"` junk)

## Fix (generic — no label hardcodes)

Pure helpers in `core/category_selection_sync.py`:

| Helper | Role |
|--------|------|
| `split_category_selection` | `"a / b/c"` → `("a", "b/c")` |
| `canonical_path_from_parts` | stable `parent/child` path |
| `path_segment_tags` | ordered unique segment tags |
| `merge_tags_preserve_manual` | append path tags; never wipe |
| `resolve_parts_against_options` | prefer known casing from options |
| `sync_selection_state` | one-shot `{parent, child, category_path, tags}` |

`ResultMetadataDialog`:

- `_syncing_category` re-entrancy guard
- `_apply_category_selection_sync(source=…)` splits path-qualified Ana text into Ana/Alt, reloads children, merges tags
- Parent change → sync + reload; child change → tag merge only
- `values()` always splits path-qualified parent text and builds path via `canonical_path_from_parts`

## Out of scope

- No Indexer / JobStore / search_engine / ranking / concept_registry / teach_search_wire changes
- No CATEGORY_TREE redesign
- No invented synonym tags

## Tests

```bash
cd /workspace/vezir_audit/cat_sync
PYTHONPATH=. QT_QPA_PLATFORM=offscreen python3 -m unittest tests.test_category_selection_sync -v
```

Qt dialog cases are optional (`CAT_SYNC_QT_DIALOG=1`) when PySide6 is available.
