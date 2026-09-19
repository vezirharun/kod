"""Oturum pipeline adım sayaçları — attempted / saved / skipped / failed."""

from __future__ import annotations

from typing import Any

STEPS = ("embedding", "ai", "semantic", "dna", "ocr", "texture")


def init_pipeline_steps(stats: dict) -> None:
    if "pipeline_steps" not in stats:
        stats["pipeline_steps"] = {
            step: {"attempted": 0, "saved": 0, "skipped": 0, "failed": 0}
            for step in STEPS
        }


def bump_pipeline_step(
    stats: dict, step: str, outcome: str, amount: int = 1
) -> None:
    init_pipeline_steps(stats)
    bucket = stats["pipeline_steps"].setdefault(
        step, {"attempted": 0, "saved": 0, "skipped": 0, "failed": 0}
    )
    bucket[outcome] = int(bucket.get(outcome, 0)) + amount


def format_pipeline_step_summary(stats: dict) -> str:
    init_pipeline_steps(stats)
    lines: list[str] = []
    for step in STEPS:
        b = stats["pipeline_steps"].get(step, {})
        lines.append(
            f"{step}: attempted={int(b.get('attempted',0))} "
            f"saved={int(b.get('saved',0))} "
            f"skipped={int(b.get('skipped',0))} "
            f"failed={int(b.get('failed',0))}"
        )
    return "Pipeline save summary:\n  " + "\n  ".join(lines)
