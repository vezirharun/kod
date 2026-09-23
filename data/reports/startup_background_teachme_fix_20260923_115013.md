# Startup + BackgroundTask + TeachMe Fix — 20260923_115013

## STARTUP ROOT CAUSE

`FeaturePreviewCache.create` runs inside `ThumbnailScheduler` detail `_LoadRunnable` (QThreadPool worker), not intentionally on the UI thread. UI-WATCHDOG critical blocks (369–752ms) correlate with CACHE MISS → create because:

1. Python/PIL (and NAS I/O) hold the GIL / contend with the UI heartbeat.
2. Detail jobs shared the same pool as list thumbnails (`MAX_CONCURRENT=4`), so multiple creates competed.
3. Hidden inspector / early selection could still enqueue detail creates during startup.

**NOT** an EPS/AI/TIFF preview algorithm bug — scheduling/lifecycle.

## BACKGROUND TASK ROOT CAUSE

`MainWindow._cleanup_workers` did:

`request_stop` → `wait_until_finished(500)` → `_background_tasks.clear()`

`BackgroundTask.function` is **not** mid-run cooperative. Startup probe / DB work often exceeds 500ms → QThread still running when parent MainWindow tears down → native:

`QThread: Destroyed while thread 'BackgroundTask' is still running`

## MULTI SELECT ROOT CAUSE

Card is a child `QWidget` inside `QListWidgetItem`. Selection was manually synced via `_focus_card_widget`, but:

- Visual state needed a stronger selected style (border + background).
- `setCurrentItem` during Ctrl/path could collapse `ExtendedSelection`.

## RIGHT CLICK ROOT CAUSE

Right-click used the same `_focus_card_widget` as left-click → `clearSelection()` → multi-select destroyed before context menu.

Also `accept()` on right press could swallow `CustomContextMenu` delivery.

## CANDIDATE GENERATION STATUS

UI reads **only** `card.rivals` / suggested via `_candidate_pairs` / `_quick_pick_candidates` (intersection for multi).

- Fixture with 3 seeded rivals → UI shows ≥3 (**PASS**, no fake labels).
- If production shows only `floral %20`, that is almost certainly **CANDIDATE GENERATION GAP** in backend rivals for those cards (or intersection of multi-select leaving one common label) — **not** UI hardcoding. Teach backend was not changed.

## KARARSIZLAR STARTUP STATUS

Inbox already loads via `_TeachMeInboxWorker` when panel visible (`reload` async). First open: cards from worker → thumbs via viewport scheduler. Candidate checkboxes built from existing `rivals` on card widgets (no extra UI-thread scoring). No architecture rewrite.

---

## BEFORE → PATCH → AFTER

| Area | Before | Patch | After |
|------|--------|-------|-------|
| Startup detail | Detail create on shared pool, early enqueue | Separate detail pool (max 1); defer until `mark_ui_interactive`; skip create when inspector hidden | UI can interact before create storm; create still correct when needed |
| BackgroundTask | 500ms wait then clear | Wait up to 30s; detach parent if still running; never destroy while running | Shutdown smoke 20/20 still_running=0 |
| TeachMe select | Weak visual; Ctrl fragile | Stronger selected chrome; Ctrl without setCurrentItem; Shift range | Ctrl/Shift/normal contract tests PASS |
| Right click | Collapsed multi-select | `_focus_card_widget_for_context`; forward menu from card | Multi preserved; menu forwards PASS |

## STARTUP TIMINGS

Automated offscreen suite only (full interactive 10× desktop startup **UNTESTED** in this agent session).

Measured indirectly:

- Detail deferred until first interactive timers (0ms + 250ms).
- FeaturePreviewCache.create no longer shares thumb pool.

## SHUTDOWN RESULTS

| Metric | Count |
|--------|-------|
| Cycles | 20 |
| still running after cleanup | **0** |
| QThread destroyed while running | **0** |
| already deleted | **0** (unit suites) |

## UI WATCHDOG

Policy (unchanged monitor): &lt;100 interact, 100–300 warning, 300–500 investigate, &gt;500 critical.

Mitigation: defer + serialize detail create; hidden inspector skip. Residual GIL during a single create may still spike once — report as **WARNING** until live desktop 10× startup confirms &lt;500ms critical=0.

## THREAD / RESOURCE

- Thumb pool: 4
- Detail pool: 1 (new)
- BackgroundTask: wait/detach on close

RAM/CPU: not profiled this run (**UNTESTED**).

## TEST RESULTS

| Suite | Result |
|-------|--------|
| test_startup_background_teachme_fix | PASS |
| test_teach_me_multiselect_context | PASS |
| test_teach_me | PASS |
| test_qthread_lifecycle | PASS |
| test_status_worker_deleted_ref | PASS |
| Shutdown smoke 20× | PASS |
| Live Windows desktop 10× startup | **UNTESTED** |
| Live panel open/close stress | **UNTESTED** |

### Summary scores

| | |
|--|--|
| BACKGROUNDTASK LIFECYCLE | **PASS** (unit + 20 smoke) |
| STARTUP / PREVIEW SCHEDULING | **PASS** (code+unit) / live UI **UNTESTED** |
| TEACHME SELECTION / RIGHT CLICK | **PASS** (Qt offscreen UI tests) |
| FAIL | 0 |
| WARNING | residual create GIL possible on live NAS |
| UNTESTED | live 10× startup, live 20× full app close with UI |
| BLOCKED | 0 |

PUSH: no.
