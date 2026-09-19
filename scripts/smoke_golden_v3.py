"""Smoke: few golden files through IndexEngineV3 + real extractors."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from core.db import Database
from core.index_v3 import IndexEngineV3, Mode, count_progress
from core.settings import AppSettings

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden_v3" / "files"


def main() -> None:
    samples = list(GOLDEN.glob("*.jpg"))[:1] + list(GOLDEN.glob("*.png"))[:1] + list(
        GOLDEN.glob("*.tif*")
    )[:1]
    if len(samples) < 2:
        raise SystemExit(f"need samples, got {len(samples)}")
    work = Path(tempfile.mkdtemp(prefix="v3_smoke_"))
    files = work / "files"
    files.mkdir()
    for p in samples:
        shutil.copy2(p, files / p.name)
    settings = AppSettings(
        cache_dir=str(work / "cache"),
        db_path=str(work / "smoke.db"),
        ai_embedding_enabled=True,
        ocr_enabled=False,
        network_timeout_sec=60,
        use_gpu=False,
    )
    settings.ensure_dirs()
    db = Database(settings.db_path)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(id, name, root_path, is_active) VALUES (1,?,?,1)",
            ("smoke", str(files)),
        )
    engine = IndexEngineV3(
        db,
        job_db_path=work / "jobs.db",
        settings=settings,
        use_real_extractors=True,
    )
    sources = [{"id": 1, "root_path": str(files)}]
    r1 = engine.run(mode=Mode.FAST, sources=sources)
    c1 = count_progress(db, source_ids=[1])
    print("FAST", c1.light_complete, "/", c1.total, r1.worker)
    r2 = engine.run(mode=Mode.GENERAL_AI, sources=sources, walk_disk=False)
    c2 = count_progress(db, source_ids=[1])
    print("AI", c2.ai_final, "/", c2.total, r2.worker)
    ok = c1.light_complete == c1.total and c2.ai_final == c2.total
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
