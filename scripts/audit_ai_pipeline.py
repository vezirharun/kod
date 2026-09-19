#!/usr/bin/env python
"""P0 AI Pipeline Audit — sistem sağlık raporu.

  python scripts/audit_ai_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from core.ai_pipeline_audit import format_health_report, run_system_health_report
    from core.settings import AppSettings

    settings = AppSettings.load()
    report = run_system_health_report(settings)
    print(format_health_report(report))
    print()
    print("--- JSON summary ---")
    import json

    print(json.dumps(
        {
            "ok": report.get("ok"),
            "device": report.get("device"),
            "components": {
                k: {"ok": v.get("ok"), "detail": v.get("detail")}
                for k, v in (report.get("components") or {}).items()
            },
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
