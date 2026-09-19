# Archive Health — Minimal Projection Implementation

**Date:** 2026-09-19 (Europe/Istanbul)  
**Tree:** `/workspace/vezir_audit/ah_impl/` (copied from `ah_audit`)  
**Product:** Vezir Pattern Search

## Goal

Add a **thin read-only projection** that maps existing V3 SSOT / JobStore / preview
signals into a unified Archive Health view for UI. No parallel repair engine, no
schema changes, no indexer / ranking / queue logic changes.

## Files touched

| Path | Change |
|------|--------|
| `core/archive_health.py` | **NEW** — projection layer |
| `ui/health_panel.py` | Arşiv Sağlığı group + refresh hooks |
| `tests/__init__.py` | package marker |
| `tests/test_archive_health.py` | **NEW** — unit tests (no PySide) |
| `ARCHIVE_HEALTH_IMPL.md` | this report |
| `core/__init__.py`, `core/index_v3/__init__.py`, `ui/__init__.py` | package markers |
| `core/index_analysis_guards.py`, `core/manual_label_guard.py` | minimal stubs so `artifact_state` imports in this truncated audit tree |

**Not modified:** `core/index_v3/` logic (queues, ui_bridge, artifact_state, workers),
indexer, preview_self_heal, physical_reconcile, cache_reconciliation, search ranking.
Existing button **“Eksikleri Onar”** unchanged (`repair_index` path).

## API (`core/archive_health.py`)

### `HealthState`

`HEALTHY | MISSING | STALE | BROKEN | REPAIR_PENDING | FAILED | UNKNOWN`

### `PRIORITY` (worst first)

`FAILED > BROKEN > REPAIR_PENDING > STALE > MISSING > UNKNOWN > HEALTHY`

### `merge_states(*states) -> HealthState`

Deterministic worst-of merge by `PRIORITY`.

### `assess_file_health(db, file_id, *, job_store=None, require_disk=False) -> dict`

Returns:

```text
{state, reasons: list[str], preview, source, ai, physical}
```

Signals used (existing only):

| Signal | Mapping |
|--------|---------|
| `files.status` in `{missing, excluded_internal, excluded}` | source **MISSING** |
| `artifact_state.assess_file` → PREVIEW READY/MISSING/INVALID | preview HEALTHY/MISSING/BROKEN |
| `preview_status` starts with `preview_invalid` | preview **BROKEN** |
| `preview_status` `preview_heal_failed` | preview **FAILED** |
| `preview_status` `preview_missing_physical` / `preview_missing*` | preview **STALE** |
| `preview_suspicious` alone | **not** BROKEN (mono may be OK) |
| path set + `physical_preview_ready=0` | physical **STALE** |
| JobStore `pending`/`claimed` for file | **REPAIR_PENDING** |
| JobStore `failed_permanent` for file | **FAILED** |
| AI_FINAL artifacts missing/invalid | ai MISSING/BROKEN |
| All light READY + no bad flags | HEALTHY |

Reasons cite real signal strings only (no invented codes). Unset UNKNOWN
components are omitted from the overall merge so they do not drag HEALTHY → UNKNOWN.

JobStore duck-hooks for tests: `file_job_states(file_id) -> iterable[str]`.

### `summarize_archive_health(db, *, job_store=None, source_ids=None) -> dict`

Fast aggregate — **no per-file full scan**. Uses `count_v3_ssot` + light SQL +
JobStore counts.

| Bucket | Approximation |
|--------|----------------|
| `healthy` | `ssot["preview"]` (physical-ready preview baseline) |
| `missing` | `ssot["waiting_preview"]` |
| `stale` | `ssot["legacy_preview"] + ssot["legacy_thumbnail"]` |
| `repair_pending` | JobStore pending files on light\|preview\|repair (0 if no store) |
| `broken` | `COUNT files WHERE preview_status LIKE 'preview_invalid%'` |
| `failed` | `ssot["failed"] + JobStore failed_permanent_files` |
| `repair_needed` | `stale + repair_pending` (main 5-bucket UI) |
| `unknown` | 0 (reserved) |
| `total` | `ssot["total"]` |
| `ssot` / `jobstore` | passthrough detail dicts |

**Double-counting:** buckets are independent projections and **may overlap / not
sum to `total`**. Documented intentionally for UI speed.

JobStore duck-hook for tests: `archive_health_counts(source_ids=...) -> dict`.

### `format_archive_health_labels(summary) -> dict[str,str]`

Turkish labels: Sağlıklı / Eksik / Onarım gerekli / Bozuk / Başarısız.

### Helpers

- `job_store_path_for_db(db_path)` → `<stem>.v3jobs.db` (same as `app_status`)
- `open_job_store_for_settings(settings)` → lazy `JobStore` or `None`

## UI (`ui/health_panel.py`)

- New `QGroupBox("Arşiv Sağlığı")` with 5 labels.
- `set_archive_health(summary)` updates labels via `format_archive_health_labels`.
- `refresh_archive_health()` opens DB + optional JobStore, calls `summarize_*`;
  on error shows `—`. Counts only — no heavy scan.
- Called at end of `apply_snapshot` when `settings.db_path` is set.
- **“Eksikleri Onar”** unchanged.

## Tests

```bash
cd /workspace/vezir_audit/ah_impl && PYTHONPATH=. python3 -m unittest tests.test_archive_health -v
```

Result (2026-09-19): **18 tests OK** (merge priority, healthy/missing/stale/broken/
repair/failed/unknown, physical INVALID, suspicious≠BROKEN, multi-issue worst,
legacy empty preview_status, summarize sqlite buckets, labels).

PySide not required for this suite; UI not imported.

## Hard rules compliance

- No new repair engine / JobStore / schema
- No indexer / queues / dino/faiss/preview_pool / ranking changes
- No rewrite of preview_self_heal / physical_reconcile / cache_reconciliation
- No new repair button
- index_v3 modules read-only (imports only)

## Notes / follow-ups

1. Bucket overlaps are expected; drill-down can use per-file `assess_file_health`.
2. Full product tree already has `index_analysis_guards` / `manual_label_guard`;
   stubs here exist only for the truncated audit sandbox.
3. Optional later: wire “Eksikleri Onar” to V3 `Mode.REPAIR` (out of scope).
