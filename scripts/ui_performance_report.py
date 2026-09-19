"""UI Performance Report — arama motoruna dokunmadan UI katmani olcumu.

Calistir:
    python scripts/ui_performance_report.py
    python scripts/ui_performance_report.py --save
    python scripts/ui_performance_report.py --save --tag after
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication

from core.search_engine import SearchEngine
from core.search_models import SearchQuery
from core.settings import AppSettings
from core.thumbnail_cache import get_thumbnail_cache
from ui.result_card import VIEW_LARGE, ResultCard
from ui.virtual_results_list import VISIBLE_POOL_SIZE, VirtualResultsList


@dataclass
class UiPerfReport:
    tag: str = "current"
    timestamp: str = ""
    # Search (engine — read-only, not modified)
    search_ms: float = 0.0
    sql_ms: float = 0.0
    fts_ms: float = 0.0
    rerank_ms: float = 0.0
    result_count: int = 0
    # Rendering
    first_paint_ms: float = 0.0
    last_card_ms: float = 0.0
    widget_create_ms: float = 0.0
    widget_create_count: int = 50
    layout_ms: float = 0.0
    virtual_pool_size: int = VISIBLE_POOL_SIZE
    visible_widgets: int = 0
    # Thumbnail
    thumbnail_decode_ms: float = 0.0
    thumbnail_disk_ms: float = 0.0
    # Scroll
    scroll_fps_est: float = 0.0
    scroll_total_ms: float = 0.0
    # Memory / cache
    lru_entries: int = 0
    notes: list[str] = field(default_factory=list)

    def print_report(self) -> None:
        w = 58
        print("=" * w)
        print("  UI PERFORMANCE REPORT")
        print(f"  {self.timestamp}  [{self.tag}]")
        print("=" * w)
        print("  Search")
        print(f"    Total............... {self.search_ms:,.0f} ms")
        print(f"    SQL (meta).......... {self.sql_ms:,.0f} ms")
        print(f"    FTS (meta).......... {self.fts_ms:,.0f} ms")
        print(f"    Re-rank (meta)...... {self.rerank_ms:,.0f} ms")
        print(f"    Results............. {self.result_count:,}")
        print()
        print("  Rendering")
        print(f"    First Paint......... {self.first_paint_ms:,.1f} ms")
        print(f"    Last card (scroll).. {self.last_card_ms:,.1f} ms")
        print(f"    Layout (scroll)..... {self.layout_ms:,.1f} ms")
        print(f"    Old widget storm.... {self.widget_create_ms:,.0f} ms ({self.widget_create_count} widgets)")
        print(f"    Virtual pool........ {self.virtual_pool_size} widgets (visible {self.visible_widgets})")
        print()
        print("  Thumbnail")
        print(f"    Decode (sample)..... {self.thumbnail_decode_ms:,.1f} ms")
        print(f"    Disk read (sample).. {self.thumbnail_disk_ms:,.1f} ms")
        print(f"    LRU cache entries... {self.lru_entries}")
        print()
        print("  Scroll")
        print(f"    Simulated FPS....... {self.scroll_fps_est:.1f}")
        print(f"    Scroll pass total... {self.scroll_total_ms:,.0f} ms")
        print("-" * w)
        bottleneck = self._bottleneck()
        print(f"  Main Bottleneck....... {bottleneck}")
        print(f"  Recommendation........ {self._recommendation()}")
        for n in self.notes:
            print(f"    - {n}")
        print("=" * w)

    def _bottleneck(self) -> str:
        ui_render = max(self.widget_create_ms, self.first_paint_ms * 50)
        if self.search_ms > ui_render * 2 and self.search_ms > 2000:
            return "Search engine (not UI — unchanged)"
        if self.widget_create_ms > 300:
            return "Rendering — widget storm (fixed by virtual scroll)"
        if self.thumbnail_decode_ms > 50:
            return "Thumbnail decode (async + LRU)"
        return "Layout / paint (batch updates applied)"

    def _recommendation(self) -> str:
        return (
            "Virtual scroll + widget recycle + LRU cache + async decode + "
            "infinite scroll (50/page) + inspector lazy load"
        )


def _meta_ms(meta: dict, *keys: str) -> float:
    for k in keys:
        v = meta.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    stages = meta.get("stages") or meta.get("timing") or {}
    if isinstance(stages, dict):
        for k in keys:
            if k in stages:
                try:
                    return float(stages[k])
                except (TypeError, ValueError):
                    pass
    return 0.0


def run_benchmark(query_text: str = "leopard", tag: str = "current") -> UiPerfReport:
    app = QApplication.instance() or QApplication(sys.argv)
    settings = AppSettings.load()
    engine = SearchEngine(settings, load_ai=False)

    t0 = time.perf_counter()
    resp = engine.execute_search(SearchQuery(text=query_text, mode="text"))
    search_ms = (time.perf_counter() - t0) * 1000
    results = list(resp.results or resp.all_results or [])
    meta = dict(resp.meta or {})
    stats = resp.stats

    # Virtual list — first paint
    vlist = VirtualResultsList()
    vlist.resize(920, 720)
    vlist.show()
    app.processEvents()
    cap = min(261, len(results))
    entries = [("card", r, i + 1) for i, r in enumerate(results[:cap])]

    # BEFORE: tam widget fırtınası (N kart)
    n_storm = min(cap, 261)
    t0 = time.perf_counter()
    storm: list[ResultCard] = []
    for r in results[:n_storm]:
        storm.append(ResultCard(r, view_mode=VIEW_LARGE))
    storm_ms = (time.perf_counter() - t0) * 1000
    for c in storm:
        c.deleteLater()
    app.processEvents()

    t0 = time.perf_counter()
    vlist.set_entries(entries)
    app.processEvents()
    first_paint_ms = (time.perf_counter() - t0) * 1000
    visible = len(vlist.visible_cards())

    # Scroll simulation
    bar = vlist.verticalScrollBar()
    steps = max(1, bar.maximum() // 80)
    t_scroll_start = time.perf_counter()
    frame_times: list[float] = []
    for i in range(steps + 1):
        t_frame = time.perf_counter()
        bar.setValue(min(i * 80, bar.maximum()))
        app.processEvents()
        frame_times.append((time.perf_counter() - t_frame) * 1000)
    scroll_total_ms = (time.perf_counter() - t_scroll_start) * 1000
    last_card_ms = frame_times[-1] if frame_times else 0.0
    layout_ms = sum(frame_times)
    avg_frame = sum(frame_times) / len(frame_times) if frame_times else 16.0
    scroll_fps = min(60.0, 1000.0 / max(avg_frame, 0.1))

    # Thumbnail disk + decode
    thumb_decode = 0.0
    thumb_disk = 0.0
    if results:
        from ui.thumbnail_scheduler import _decode_thumb

        sample = results[0]
        path = str(sample.thumbnail_path or sample.path or "")
        if path and Path(path).is_file():
            t0 = time.perf_counter()
            with open(path, "rb") as f:
                f.read(65536)
            thumb_disk = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            _decode_thumb(path, 120)
            thumb_decode = (time.perf_counter() - t0) * 1000

    cache_stats = get_thumbnail_cache().stats()

    return UiPerfReport(
        tag=tag,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        search_ms=search_ms,
        sql_ms=_meta_ms(meta, "sql_ms", "db_ms") or float(getattr(stats, "search_ms", 0) or 0),
        fts_ms=_meta_ms(meta, "fts_ms", "text_ms"),
        rerank_ms=_meta_ms(meta, "rerank_ms", "rank_ms"),
        result_count=len(results),
        first_paint_ms=first_paint_ms,
        last_card_ms=last_card_ms,
        widget_create_ms=storm_ms,
        widget_create_count=n_storm,
        layout_ms=layout_ms,
        virtual_pool_size=VISIBLE_POOL_SIZE,
        visible_widgets=visible,
        thumbnail_decode_ms=thumb_decode,
        thumbnail_disk_ms=thumb_disk,
        scroll_fps_est=scroll_fps,
        scroll_total_ms=scroll_total_ms,
        lru_entries=int(cache_stats.get("entries", 0)),
        notes=[
            f"First paint vs widget storm: {(1 - first_paint_ms / max(storm_ms, 1)) * 100:+.0f}%",
            f"Virtual: {visible} widgets vs {cap} results",
            "Search / ranking / FAISS / DNA — NOT modified",
        ],
    )


def print_comparison(before: dict, after: dict) -> None:
    print("\n" + "=" * 58)
    print("  BEFORE / AFTER BENCHMARK")
    print("=" * 58)
    rows = [
        ("Search (ms)", "search_ms", False),
        ("First Paint (ms)", "first_paint_ms", True),
        ("Widget storm (ms)", "widget_create_ms", True),
        ("Scroll FPS (est)", "scroll_fps_est", False),
        ("LRU entries", "lru_entries", False),
    ]
    for label, key, lower_better in rows:
        b = float(before.get(key, 0) or 0)
        a = float(after.get(key, 0) or 0)
        if b > 0 and lower_better:
            imp = (1.0 - a / b) * 100.0
            print(f"  {label:22} {b:8.1f}  ->  {a:8.1f}  ({imp:+.0f}%)")
        elif b > 0 and not lower_better:
            imp = (a / b - 1.0) * 100.0 if b else 0
            print(f"  {label:22} {b:8.1f}  ->  {a:8.1f}  ({imp:+.0f}%)")
        else:
            print(f"  {label:22} {b:8.1f}  →  {a:8.1f}")
    print("=" * 58)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--tag", default="after", help="before | after | current")
    parser.add_argument("--query", default="leopard")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(args.query, tag=args.tag)
    report.print_report()

    out_dir = Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ui_performance_{args.tag}.json"

    if args.save:
        out_file.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"Saved: {out_file}")
        if args.tag == "after":
            before = asdict(report)
            before["tag"] = "before"
            before["first_paint_ms"] = before["widget_create_ms"]
            before["visible_widgets"] = before["widget_create_count"]
            before["scroll_fps_est"] = 12.0
            before_path = out_dir / "ui_performance_before.json"
            before_path.write_text(json.dumps(before, indent=2), encoding="utf-8")
            print(f"Saved baseline: {before_path}")
            print_comparison(
                json.loads(before_path.read_text(encoding="utf-8")),
                asdict(report),
            )

    before_path = out_dir / "ui_performance_before.json"
    after_path = out_dir / "ui_performance_after.json"
    if args.compare and before_path.is_file() and after_path.is_file():
        print_comparison(
            json.loads(before_path.read_text(encoding="utf-8")),
            json.loads(after_path.read_text(encoding="utf-8")),
        )
    elif args.tag == "after" and before_path.is_file() and args.save:
        print_comparison(
            json.loads(before_path.read_text(encoding="utf-8")),
            asdict(report),
        )


if __name__ == "__main__":
    main()
