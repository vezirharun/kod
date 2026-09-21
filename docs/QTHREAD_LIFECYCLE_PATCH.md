# QThread / SearchWorker lifecycle stability patch

Minimal changes under `life_patch/` to stop QThread use-after-free and duplicate AI model loads.

## Goals

1. **Single-flight SearchWorker** — never start a second search thread while one is running; queue at most one pending `(query, instant)` and launch it on `finished`.
2. **Safe teardown** — `arm_delete_later_on_finished()` connects `QThread.finished → deleteLater` once; UI never long-waits; running workers are not orphan-destroyed.
3. **Shared FeatureExtractor** — `try_load_ai_extractor` / `get_shared_ai_extractor` cache one AI extractor per `(use_gpu,)` key under a lock; `SearchEngine.ensure_ai_loaded` reuses it (no second ctor).
4. **Debuggability** — each worker subclass sets `objectName` to its class name; health `_ProductionWorker` and metadata `_OptionsWorker` likewise.

## Files

| File | Change |
|------|--------|
| `ui/worker_threads.py` | `arm_delete_later_on_finished` on `_WorkerBase`; `setObjectName` per subclass |
| `ui/main_window.py` | pending search / quick-index / source-count; stop→finished→deleteLater; StatusWorker armed |
| `core/capability_check.py` | module lock + shared AI extractor cache |
| `core/search_engine.py` | `ensure_ai_loaded` assigns shared extractor |
| `core/qthread_lifecycle.py` | `should_defer_new_worker(is_running)` |
| `ui/health_panel.py` / `ui/result_metadata_dialog.py` | objectName on workers |
| `tests/test_qthread_lifecycle.py` | unit tests (offscreen Qt + stubs) |

Worker `objectName` values: IndexWorker, QuickIndexWorker, SearchWorker, SourceFileCountWorker, BackgroundTask, PurgeMissingWorker, StatusWorker, CacheReconciliationWorker, PurgeSourceWorker.

## SearchWorker flow

```
_start_search_worker(query, instant)
  UI prep…
  if current.isRunning():
      _pending_search = (query, instant)   # overwrite = latest wins
      request_stop(); hook finished once
      return
  _launch_search_worker(...)

_on_search_qthread_finished:
  deleteLater sender; clear ref if current
  if pending: launch it
```

Cancel/clear clears `_pending_search` so a cancelled search does not relaunch.

## Other workers

- **QuickIndexWorker / SourceFileCountWorker / CacheReconciliationWorker**: if old running → `request_stop` + arm deleteLater; pending or ≤100ms wait before starting replacement; never force-destroy a running C++ thread.
- **StatusWorker / lane StatusWorker**: `arm_delete_later_on_finished` on create; optional Python-ref clear when idle after finished_ok.

## Out of scope

index_v3, Indexer, JobStore, ranking, safe_decode, DINO/CLIP algorithms, preview, DB schema, category/customer/learning.

## Test

```bash
cd /workspace/vezir_audit/life_patch
PYTHONPATH=. QT_QPA_PLATFORM=offscreen \
  .venv/bin/python -m unittest tests.test_qthread_lifecycle -v
```
