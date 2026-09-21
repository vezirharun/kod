# INDEX KAPANMA / THREAD DUMP ROOT CAUSE AUDIT

**Date:** 2026-09-21 (Europe/Istanbul)  
**Scope:** Audit only — no code patch.  
**Trigger:** Index running; process exited; user saw faulthandler-style stack dump.

---

## A) Kesin bulgu

1. **Stack dump kill etmez.** `core/ui_perf.py` UI-freeze watchdog (`gap >= 0.50s`) `faulthandler.dump_traceback(all_threads=True)` yazar (`data/logs/ui_freeze_tracebacks.log`, ~118 MB). Bu diagnostik; process terminate etmez.

2. **Process olumu native crash (WER).** Application Error / WER:
   - **2026-09-20 15:22** — `python.exe` / **Qt6Core.dll** / **0xc0000409** (BEX64) → LocalDump `data/crash_dumps/python.exe.20580.dmp` (PID 0x5064=20580)
   - **2026-09-20 18:00** — `python.exe` / **pyside6.abi3.dll** / **0xc0000005** (ACCESS_VIOLATION) → `python.exe.10348.dmp` (PID 0x286c=10348)
   Ayni imza (Qt6Core 0xc0000409) 18–20 Eylul araliginda tekrarli.

3. **`_refresh_status_during_index` paralel StatusWorker spam etmez.**
   - Timer: `QTimer` interval **1000 ms** (`main_window.py` ~167–169).
   - Guard: `if self._lane_status_worker and self._lane_status_worker.isRunning(): return` (~3949–3950).
   - Sonra `StatusWorker(..., lanes_only=True)` + `start()` (~3951–3956).
   Ayni anda en fazla **1** lane StatusWorker.

4. **Dump’taki satirlar:**
   - `worker_threads.py:113` = `_WorkerBase.__init__` → `QThread.__init__`
   - `worker_threads.py:1413` = `StatusWorker.__init__`
   Tip: **QThread** (ThreadPool/QRunnable degil).

5. **Bitmis StatusWorker icin `deleteLater` yok** (`_on_status_updated`). Bitince referans degisir; QObject MainWindow child olarak kalabilir → **QObject churn** (olcum yok; concurrent thread sayisi buyume iddiasi degil).

6. **`safe_decode.run_with_timeout`:** paylasimli `ThreadPoolExecutor(max_workers=2)`. Timeout sonrasi Future terk edilir; native decode thread aninda olmez; 8 abandon sonrasi pool `shutdown(wait=False)` + yeni pool. Eski native thread leak/pressure riski kod yorumunda da belirtilmis.

7. **Bu olayda WER’de 0xc0000374 (heap corruption) yok** (onceki ayri olay olabilir).

---

## B) Guclu aday

- **Olum nedeni:** Qt/PySide **native crash**, Python traceback exception degil; freeze dump yalnizca UI stall aninin fotografidir.
- Index sirasinda ayni anda: preview (`safe_decode` + **pyvips**), StatusWorker lane COUNT, Archive Intelligence, background_scan, health/resource monitors, UI watchdog — **yuksek concurrent native/UI baskisi** crash pencereleriyle zamansal olarak ortusuyor.
- Full `_on_status_updated` yolunda UI thread’de `category_tree_panel.set_db_path` → DB/migrate (`~4324`) freeze log’da goruldu (index-disi path); UI stall uretebilir.

---

## C) Sadece ihtimal

- StatusWorker her ~1s yeniden construct → Qt QThread lifecycle / signal churn → Qt6Core fail-fast’e katki.
- Timeout’ta terk edilen pyvips/decode thread → bellek/heap baskisi (bu dump’ta 0xc0000374 yok).
- Archive Intelligence dogrudan crash fail module degil; resource pressure.

---

## D) Kanitlanamayanlar

- WinDbg ile dump icinden Qt callstack (dosyalar 3–10 GB; bu audit’te acilmadi).
- Crash aninda current thread’in StatusWorker `__init__` oldugu.
- "Thread sayisi sinirsiz buyuyor" — **runtime olcum yok**; kod concurrent StatusWorker’i 1 ile sinirliyor.
- Preview worker ile StatusWorker arasinda dogrudan paylasilan mutable Qt nesnesi baglantisi.

---

## E) Dosya / satir

| Konu | Yer |
|------|-----|
| Index status timer | `ui/main_window.py` ~167–169, ~3824 |
| Lane refresh + guard | `ui/main_window.py` ~3943–3956 |
| Full status refresh | `ui/main_window.py` ~3958–3979 |
| Status apply / tree UI | `ui/main_window.py` ~4285–4324 |
| QThread base / StatusWorker | `ui/worker_threads.py` ~109–130, ~1409–1426 |
| Decode timeout pool | `core/index_v3/safe_decode.py` ~21–126 |
| Freeze dump (kill degil) | `core/ui_perf.py` ~111–171 |
| LocalDumps | `HKLM\...\LocalDumps` → `data/crash_dumps` |

---

## F) Program neden kapanmis olabilir?

1. UI freeze watchdog stack yazdi (kill degil).
2. Ayni yuk penceresinde **Qt6Core 0xc0000409** veya **pyside6 0xc0000005** native crash → WER + LocalDump → process olumu.
3. Eldeki faulthandler cikti "neden oldu" degil; "o anda kim kosuyordu" snapshot’i.

---

## G) Duzeltme gerekiyorsa en kucuk guvenli adimlar (PATCH YOK — oneri)

1. Once kucuk dump’u WinDbg ile ac: faulting stack Qt signal/QThread mi?
2. StatusWorker: tek kalici instance **veya** `finished` → `deleteLater()` (churn azalt).
3. Status callback’te UI thread DB/tree populate’i ertelenmis/worker’a al.
4. `ui_freeze_tracebacks.log` rotate/cap (118 MB I/O baskisi).
5. Index sirasinda AI scanner’i gecici dusurmek A/B — yalniz olcumle.

**Yapilmadi:** hicbir kod degisikligi.
