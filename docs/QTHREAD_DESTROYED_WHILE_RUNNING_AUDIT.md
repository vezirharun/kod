# QTHREAD DESTROYED WHILE RUNNING — ROOT CAUSE AUDIT

**Date:** 2026-09-21 ~09:44 (Europe/Istanbul)  
**Scope:** Audit only — no code patch.  
**Doc:** confirms WER + app.log + source lifecycle.

---

## Timeline (app.log + WER)

| Time | Event |
|------|--------|
| 09:44:14 | RESOURCE threads=96, INDEX_ACTIVE=0, workers=0 |
| 09:44:15 | OpenCLIP yüklendi (cpu) |
| 09:44:18 | DINOv2 yüklendi (cpu) |
| 09:44:21 | RESOURCE threads=103 (+7); OpenCLIP yüklendi (cpu) again |
| 09:44:22 | RESULT_DETAIL file_id=3530 |
| 09:44:25 | UI-WATCHDOG blocked=328ms; RESULT_DETAIL 3530 again |
| 09:44:26 | **WER Application Error: Qt6Core.dll 0xc0000409** (PID 0x49b0=18864) |
| 09:44:31 | WER BEX64 same signature |

`QThread: Destroyed while thread ''' is still running` is **not** in `app.log` (stderr/Qt). It coincides with this native abort window.

Note: `setup_logger()` returns a **process-wide singleton**; first importer wins the logger name — so FeatureExtractor lines appear as `core.network_index_throttle`.

---

## A) KESİN BULGU

1. Process death = **native Qt fail-fast**: `Qt6Core.dll` / **`0xc0000409`** / BEX64 — **same signature** as 20.09 15:22 (and earlier). Not a Python traceback exit.
2. `QThread: Destroyed while running` = Qt warning when a **QThread QObject destructor** runs while the OS thread is still alive. Empty name `''` = no `setObjectName` on workers → **instance not named in the message**.
3. **StatusWorker isRunning guard exists** (`main_window.py` ~3949–3950); with INDEX_ACTIVE=0 this crash path is **not** index lane refresh spam.
4. **SearchWorker cancel does not wait** (`_cancel_search_worker` ~2204–2208 explicitly: request_stop only, then `_search_worker = SearchWorker(...)`). Same stop-without-wait pattern: QuickIndexWorker, SourceFileCountWorker, CacheReconciliationWorker replace.
5. **Double model load is kesin:** `ensure_ai_loaded` → `try_load_ai_extractor` constructs `FeatureExtractor(use_ai=True)` (full DINO+CLIP init + logs), then constructs **another** `FeatureExtractor(use_ai=True)` for `self.extractor` (`search_engine.py` ~315–329; `capability_check.py` ~146–150). No process-wide model singleton. Matches OpenCLIP → DINO → OpenCLIP and threads 96→103.
6. **RESULT_DETAIL does not create QThread / does not load OpenCLIP.** Inspector `set_result` → `_set_result_heavy` is UI text; relations use **QRunnable + QThreadPool**; detail image via **thumbnail_scheduler** (also QRunnable/QThreadPool). RESULT_DETAIL log is paint/size after decode.
7. **safe_decode** uses Python `ThreadPoolExecutor`, not QThread — cannot emit this Qt QThread warning.

---

## B) QTHREAD'I YARATAN SINIF (inventory)

| Class | File | parent | deleteLater | stop/wait |
|-------|------|--------|-------------|-----------|
| `_WorkerBase` / IndexWorker / SearchWorker / StatusWorker / … | `ui/worker_threads.py` | usually MainWindow | StatusWorker: **no** | base `wait_until_finished` on cleanup only |
| `_ProductionWorker` | `ui/health_panel.py` | | | |
| `_OptionsWorker` | `ui/result_metadata_dialog.py` | dialog | | dialog destroy risk |
| `_TeachMeInboxWorker` | `ui/teach_me_panel.py` | | finished→deleteLater | better |
| BackgroundTask | worker_threads | MainWindow | after finished_ok/error | held in `_background_tasks` |

Inspector/thumb: **QRunnable**, not QThread.

Lines ~113 / ~1413 = `_WorkerBase.__init__` → `QThread.__init__` / `StatusWorker.__init__` (construction, not the destroy message itself).

---

## C) QTHREAD'I YOK EDEN SINIF / SCOPE

- **Cannot pin exact instance** from the empty-name stderr line alone.
- Destroy happens when C++ QThread is deleted: MainWindow/`QApplication` teardown, `deleteLater`, or parentless GC.
- Python reassignment `self._search_worker = …` with `parent=self` does **not** immediately destroy the old C++ object (stays MainWindow child) — orphans accumulate until teardown/abort.
- `closeEvent` → `_cleanup_workers` → `request_stop` + `wait_until_finished`; on **timeout** wait returns False and teardown still continues → classic Destroyed-while-running during quit.
- This incident looks like **in-session abort** (WER at 09:44:26), so warning likely from Qt tearing down / aborting while a child QThread still runs (orphaned SearchWorker/StatusWorker/etc.), not proven as a single named worker.

---

## D) THREAD NEDEN HÂLÂ RUNNING

Likely: `request_stop()` set but `run()` still inside long work (`ensure_ai_loaded` / search / decode) when QObject destruction begins (abort or parent teardown). AI load at 09:44:15–21 is exactly such a long `SearchWorker.run()` section.

---

## E) DETAIL / AI / MODEL İLİŞKİSİ

```
SearchWorker.run
  → SearchEngine(load_ai=False)
  → ensure_ai_loaded()
       → try_load_ai_extractor → FeatureExtractor()  # load #1 logs
       → FeatureExtractor() again                    # load #2 logs
  → (results on UI)
Inspector.set_result / RESULT_DETAIL                 # paint only; no model load
```

RESULT_DETAIL is **correlation**, not the model-loader. OpenCLIP/DINO are **not** Qt QThreads; torch may add OS threads (+7).

---

## F) STATUSWORKER İLİŞKİSİ

- Index path: isRunning guard → max 1 concurrent lane worker; **no deleteLater** after finish → QObject churn, not proven concurrent QThread growth.
- This crash: INDEX_ACTIVE=0 → lane refresh not the active story.
- StatusWorker **not** identified as the destroyed instance.

---

## G) SAFE_DECODE İLİŞKİSİ

- Timeout: `fut.result(timeout)` raises; worker may keep running in shared pool; after 8 abandons pool `shutdown(wait=False)`.
- **Yes**, timed-out decode can continue in background (native/pyvips).
- **Not** a QThread; unrelated to this Qt warning text.

---

## H) THREAD COUNT KAYNAĞI (96→103)

- Not classified per-thread in logs (only process `threads=`).
- Timing matches **double FeatureExtractor / torch** init (+~7), not QThread create (workers=0, INDEX_ACTIVE=0).
- Mix: Python threads, torch/OpenMP, Qt internal, QThreadPool workers, QThreads — instrumentation needed to split.

---

## I) ÖNCEKİ CRASHLERLE İLİŞKİ

| Event | Signature |
|-------|-----------|
| 20.09 15:22 | Qt6Core **0xc0000409** |
| 20.09 18:00 | pyside6 **0xc0000005** |
| **21.09 09:44:26** | Qt6Core **0xc0000409** (same offset family) |

**Same native Qt abort family** as 20.09 15:22. QThread stderr is consistent with Qt dying while threads still run — **related symptom**, not a separate root proof. Do not claim identical C++ stack without WinDbg on the dump.

---

## J) EN KÜÇÜK GÜVENLİ PATCH ÖNERİSİ (not applied)

1. `_cancel_search_worker` / QuickIndex / SourceFileCount / CacheRecon: **wait** (or queue) before replacing; `finished` → `deleteLater`; `setObjectName` for diagnosis.
2. `ensure_ai_loaded`: reuse extractor from `try_load_ai_extractor` — **one** FeatureExtractor (fixes double OpenCLIP/DINO).
3. WinDbg dump for PID 18864 / latest LocalDump before broader refactors.

**No code was changed in this audit.**
