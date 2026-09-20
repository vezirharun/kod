# Teach → Text Search Wire

**Date:** 2026-09-20 (Europe/Istanbul)  
**Scope:** Minimal generic bridge so taught category paths participate in text search.

## Problem

Teaching a category (e.g. path `parent/leaf`) wrote `manual_category_path` for the inspector, but:

1. `text_search_blob` often omitted slash-split tokens, so FTS/blob recall missed leaf queries.
2. `_label_text_search` ignored category path columns when family hints were empty.
3. Gender-only human-semantic demotion zeroed scores for non-face hits even when the row was user-taught / learned / path-matched.

## Wire (generic — no gender hardcode)

| Layer | Change |
|-------|--------|
| `core/teach_search_wire.py` | Helpers: path tokens, blob token list, gender-keep evidence, SQLite path search |
| `core/text_index.py` | `build_text_search_blob` falls back to `texture_map` path/aliases; adds slash-split segments |
| `core/category_learning.py` | Passes `category_path` / `category_aliases` / `pattern_type` into blob build |
| `core/teach_me.py` | `_write_file_classification` rebuilds `text_search_blob` when path set |
| `core/db.py` | `_category_path_label_search` always merged from `_label_text_search` |
| `core/search_engine.py` | Skip gender demotion when `keep_taught_evidence_on_gender`; stamp path on `debug` |

## Non-goals

- No ranking rewrite; no Indexer / JobStore / DINO / CLIP / Preview schema changes.
- `collect_learned_hits` / `apply_learned_to_results` untouched.
- Face-gender preferential path kept.

## Test

```bash
cd /workspace/vezir_audit/erkek_wire && PYTHONPATH=. python3 -m unittest tests.test_teach_text_search_wire -v
```
