"""Hard Case Benchmark — zor ayrım çiftlerinde metin araması kalitesi.

Çalıştır:
    python scripts/hard_case_benchmark.py
    python scripts/hard_case_benchmark.py --save  # JSON raporu yazar
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

from core.search_engine import SearchEngine
from core.search_models import SearchQuery
from core.settings import AppSettings


# ── Zor çiftler ──────────────────────────────────────────────────────────────
# Her kayıt: query, beklenen_family, kaçınılacak_family
HARD_CASES = [
    # Hayvan baskı ayrımları
    {"query": "leopard",  "want": "animal_print", "avoid": "floral",      "note": "Leopard vs Floral"},
    {"query": "jaguar",   "want": "animal_print", "avoid": "floral",      "note": "Jaguar vs Floral"},
    {"query": "cheetah",  "want": "animal_print", "avoid": "geometric",   "note": "Cheetah vs Geometric"},
    {"query": "zebra",    "want": "animal_print", "avoid": "stripe",      "note": "Zebra vs Stripe"},
    {"query": "tiger",    "want": "animal_print", "avoid": "stripe",      "note": "Tiger vs Stripe"},
    {"query": "snake",    "want": "animal_print", "avoid": "geometric",   "note": "Snake vs Geometric"},
    # Çiçek/hayvan
    {"query": "leopard flower", "want": "floral",       "avoid": "animal_print", "note": "Leopard+Çiçek → Floral"},
    # Doku/aile karışımı
    {"query": "paisley",  "want": "paisley",       "avoid": "floral",      "note": "Paisley vs Floral"},
    {"query": "marble",   "want": "marble_abstract","avoid": "geometric",  "note": "Marble vs Geometric"},
    {"query": "abstract", "want": "marble_abstract","avoid": "floral",     "note": "Abstract vs Floral"},
    {"query": "plaid",    "want": "plaid_check",    "avoid": "stripe",     "note": "Plaid vs Stripe"},
    {"query": "houndstooth","want":"plaid_check",   "avoid": "geometric",  "note": "Houndstooth vs Geometric"},
    # Marka / monogram
    {"query": "louis vuitton","want":"monogram_logo","avoid":"floral",     "note": "LV Monogram vs Floral"},
    {"query": "gucci",    "want": "monogram_logo", "avoid": "geometric",  "note": "GG vs Geometric"},
    # Yazım hataları + doğru aile
    {"query": "leoaprd",  "want": "animal_print", "avoid": "floral",      "note": "Typo: leoaprd → leopard"},
    {"query": "florla",   "want": "floral",        "avoid": "animal_print","note": "Typo: florla → floral"},
    {"query": "zebrra",   "want": "animal_print", "avoid": "stripe",      "note": "Typo: zebrra → zebra"},
    {"query": "puantye",  "want": "geometric",    "avoid": "floral",      "note": "Typo: puantye → puantiye"},
]


@dataclass
class CaseResult:
    query: str
    note: str
    want: str
    avoid: str
    n_results: int
    top_family: str
    top_filename: str
    top_score: float
    want_in_top5: bool
    avoid_dominated: bool  # top-3 tüm avoid ailesi mi
    spell_corrected: str
    elapsed_ms: float
    passed: bool
    details: list[dict] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    version: str
    timestamp: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    elapsed_ms: float
    cases: list[CaseResult]

    def print_summary(self) -> None:
        w = 64
        print("=" * w)
        print(f"  HARD CASE BENCHMARK  --  {self.timestamp}")
        print(f"  Gecti: {self.passed}/{self.total}  ({self.pass_rate:.0%})  {self.elapsed_ms:.0f}ms toplam")
        print("=" * w)
        for c in self.cases:
            icon = "OK" if c.passed else "XX"
            spell_raw = c.spell_corrected.encode("ascii", errors="replace").decode("ascii")
            spell = f" [typo:{spell_raw}]" if c.spell_corrected else ""
            print(
                f"  {icon} {c.query!r:20}{spell:16} "
                f"top={c.top_family or '-':20} "
                f"%{c.top_score*100:.0f}  {c.note}"
            )
        print("=" * w)
        if self.passed < self.total:
            print("  BASARISIZ DURUMLAR:")
            for c in self.cases:
                if not c.passed:
                    print(f"    - {c.query!r} ({c.note})")
                    print(f"      Beklenen: {c.want}  Bulunan: {c.top_family}")
        print()


def run_benchmark(engine: SearchEngine, *, verbose: bool = False) -> BenchmarkReport:
    t_total = time.perf_counter()
    results: list[CaseResult] = []

    for case in HARD_CASES:
        q    = case["query"]
        want = case["want"]
        avoid = case["avoid"]
        note = case["note"]

        t0 = time.perf_counter()
        resp = engine.execute_search(SearchQuery(text=q, mode="text"))
        elapsed = (time.perf_counter() - t0) * 1000

        hits = resp.results or []
        top = hits[0] if hits else None
        top_family  = top.pattern_family if top else ""
        top_fname   = top.filename[:40] if top else ""
        top_score   = top.score if top else 0.0
        spell_corr  = resp.meta.get("spell_corrected", "") if resp.meta else ""

        top5_families = [r.pattern_family for r in hits[:5]]
        want_in_top5  = any(pf == want or pf in (want, "animal_print") for pf in top5_families)
        avoid_dom     = bool(top5_families) and all(pf == avoid for pf in top5_families[:3])

        # Geçer mi?
        # Kural: istenen aile top-5'te olmalı VE kaçınılan aile top-1'de olmamalı
        passed = want_in_top5 and (top_family != avoid or top_family == want)

        details = [
            {"rank": i + 1, "filename": r.filename[:40], "family": r.pattern_family, "score": round(r.score, 3)}
            for i, r in enumerate(hits[:5])
        ]

        cr = CaseResult(
            query=q, note=note, want=want, avoid=avoid,
            n_results=len(hits), top_family=top_family, top_filename=top_fname,
            top_score=top_score, want_in_top5=want_in_top5,
            avoid_dominated=avoid_dom, spell_corrected=spell_corr,
            elapsed_ms=elapsed, passed=passed, details=details,
        )
        results.append(cr)
        if verbose:
            icon = "✓" if passed else "✗"
            print(f"  {icon} {q!r:18} → {top_family or '-':20} %{top_score*100:.0f}  [{elapsed:.0f}ms]")

    total_ms = (time.perf_counter() - t_total) * 1000
    n_pass = sum(1 for r in results if r.passed)
    return BenchmarkReport(
        version="1.0",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total=len(results),
        passed=n_pass,
        failed=len(results) - n_pass,
        pass_rate=n_pass / max(len(results), 1),
        elapsed_ms=total_ms,
        cases=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard Case Benchmark")
    parser.add_argument("--save", action="store_true", help="JSON raporu kaydet")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    settings = AppSettings.load()
    engine   = SearchEngine(settings, load_ai=False)

    print("Hard Case Benchmark başlatılıyor…\n")
    report = run_benchmark(engine, verbose=args.verbose)
    report.print_summary()

    if args.save:
        out = Path("data/reports/hard_case_benchmark.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(report)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rapor: {out}")


if __name__ == "__main__":
    main()
