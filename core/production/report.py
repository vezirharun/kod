"""Üretim raporu — HTML (PDF tarayıcıdan yazdırılabilir)."""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

from core.db import Database
from core.logger import setup_logger
from core.production.benchmark import run_performance_benchmark
from core.production.log_analyzer import analyze_logs_24h

logger = setup_logger(__name__)


def _brand_distribution(db: Database, limit: int = 20) -> list[tuple[str, int]]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(NULLIF(TRIM(f.customer), ''), 'Bilinmiyor') AS brand, COUNT(*) AS cnt
            FROM files f
            WHERE f.status='indexed'
            GROUP BY brand
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [(str(r["brand"]), int(r["cnt"])) for r in rows]


def _pattern_families(db: Database, limit: int = 15) -> list[tuple[str, int]]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT g.name, COUNT(m.file_id) AS cnt
            FROM pattern_groups g
            JOIN pattern_group_members m ON m.group_id=g.id
            GROUP BY g.id
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [(str(r["name"]), int(r["cnt"])) for r in rows]


def generate_production_report(settings, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    db = Database(settings.db_path)
    dash = db.count_index_pipeline_dashboard()
    bench = run_performance_benchmark(settings)
    logs = analyze_logs_24h(settings)
    brands = _brand_distribution(db)
    families = _pattern_families(db)

    total = int(dash.get("total_searchable", 0) or 0)
    ocr_pct = bench["metrics"].get("ocr_coverage_pct", 0)
    sem_pct = bench["metrics"].get("semantic_coverage_pct", 0)
    dup_groups = len(families)

    out_dir = Path(output_dir or Path(settings.db_path).parent / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    html_path = out_dir / f"production_report_{ts}.html"

    brand_rows = "".join(
        f"<tr><td>{html.escape(b)}</td><td>{c:,}</td></tr>" for b, c in brands
    )
    family_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{c:,}</td></tr>" for n, c in families
    )

    body = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>Vezir Production Report</title>
<style>
body{{font-family:Segoe UI,sans-serif;margin:24px;color:#1a1a1a}}
h1{{color:#2c5282}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ccc;padding:8px;text-align:left}}
th{{background:#edf2f7}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{border:1px solid #e2e8f0;border-radius:8px;padding:16px}}
</style></head><body>
<h1>Vezir Pattern Search — Üretim Raporu</h1>
<p>Oluşturulma: {html.escape(time.strftime("%Y-%m-%d %H:%M:%S"))}</p>
<div class="grid">
<div class="card"><h3>Arşiv</h3>
<ul>
<li>Toplam desen: <b>{total:,}</b></li>
<li>OCR oranı: <b>{ocr_pct}%</b></li>
<li>Semantic oranı: <b>{sem_pct}%</b></li>
<li>Duplicate grupları: <b>{dup_groups}</b></li>
</ul></div>
<div class="card"><h3>Son 24 Saat Log</h3>
<ul>
<li>OCR hataları: <b>{logs.get('ocr_errors', 0)}</b></li>
<li>SQLite lock: <b>{logs.get('sqlite_locks', 0)}</b></li>
<li>Handle leak uyarısı: <b>{logs.get('handle_leaks', 0)}</b></li>
<li>Crash: <b>{logs.get('crashes', 0)}</b></li>
<li>Ort. CPU: <b>{logs.get('avg_cpu_percent', '—')}%</b></li>
<li>Ort. RAM: <b>{logs.get('avg_ram_mb', '—')} MB</b></li>
</ul></div>
</div>
<h2>Marka Dağılımı</h2>
<table><tr><th>Marka</th><th>Adet</th></tr>{brand_rows}</table>
<h2>Pattern Family</h2>
<table><tr><th>Aile</th><th>Üye</th></tr>{family_rows}</table>
<h2>Eksik Analizler</h2>
<pre>{html.escape(json.dumps({{
    "pending_preview": dash.get("pending_preview"),
    "pending_ocr": dash.get("pending_ocr"),
    "pending_embedding": dash.get("pending_embedding"),
    "pending_pattern_dna": dash.get("pending_pattern_dna"),
    "pending_semantic_tag": dash.get("pending_semantic_tag"),
}}, indent=2, ensure_ascii=False))}</pre>
<h2>Performans</h2>
<pre>{html.escape(json.dumps(bench.get("metrics", {{}}), indent=2, ensure_ascii=False))}</pre>
</body></html>"""

    html_path.write_text(body, encoding="utf-8")
    result = {
        "html_path": str(html_path),
        "total_patterns": total,
        "duplicate_groups": dup_groups,
        "logs_24h": logs,
        "benchmark": bench.get("metrics"),
    }
    logger.info("Production report → %s", html_path)
    return result
