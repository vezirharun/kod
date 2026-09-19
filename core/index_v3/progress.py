"""Index Engine V3 — progress SSOT from artifact reality (no event counters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from core.index_v3.artifact_state import assess_file
from core.index_v3.types import (
    HEAVY_ARTIFACTS,
    LIGHT_ARTIFACTS,
    OBJECT_CONCEPT_ARTIFACTS,
    OWLV2_ARTIFACTS,
    POST_GA_ARTIFACTS,
    Artifact,
)


@dataclass
class StageProgress:
    name: str
    completed: int
    total: int

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    @property
    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return 100.0 * self.completed / self.total


@dataclass
class ProgressSnapshot:
    total: int
    stages: dict[str, StageProgress] = field(default_factory=dict)
    light_complete: int = 0
    ai_final: int = 0
    preview_waiting: int = 0

    def stage(self, key: str) -> StageProgress:
        return self.stages.get(key) or StageProgress(key, 0, self.total)


def count_progress(
    db: Any,
    file_ids: Iterable[int] | None = None,
    *,
    source_ids: list[int] | None = None,
) -> ProgressSnapshot:
    """completed = COUNT distinct active files where artifact READY.

    Bulk (file_ids is None): fast SQL SSOT — 10k+ arşivde assess_file döngüsü YASAK.
    Small id list: assess_file (unit tests / sparse checks).
    """
    if file_ids is not None:
        ids = list(file_ids)
        total = len(ids)
        counters = {a.value: 0 for a in list(LIGHT_ARTIFACTS) + list(HEAVY_ARTIFACTS) + list(OBJECT_CONCEPT_ARTIFACTS) + list(OWLV2_ARTIFACTS) + list(POST_GA_ARTIFACTS)}
        light_ok = 0
        ai_ok = 0
        preview_wait = 0
        for fid in ids:
            report = assess_file(db, fid, require_disk=False)
            for art in list(LIGHT_ARTIFACTS) + list(HEAVY_ARTIFACTS) + list(OBJECT_CONCEPT_ARTIFACTS) + list(OWLV2_ARTIFACTS) + list(POST_GA_ARTIFACTS):
                if report.ready(art):
                    counters[art.value] += 1
            if report.light_complete:
                light_ok += 1
            if report.ai_final:
                ai_ok += 1
            if not report.preview_ready and report.ready(Artifact.HASH):
                preview_wait += 1
            elif not report.preview_ready:
                if any(not report.ready(a) for a in (Artifact.THUMBNAIL, Artifact.PREVIEW)):
                    preview_wait += 1
        stages = {name: StageProgress(name, n, total) for name, n in counters.items()}
        stages["light"] = StageProgress("light", light_ok, total)
        stages["ai_final"] = StageProgress("ai_final", ai_ok, total)
        return ProgressSnapshot(
            total=total,
            stages=stages,
            light_complete=light_ok,
            ai_final=ai_ok,
            preview_waiting=preview_wait,
        )

    from core.index_v3.ui_bridge import count_v3_ssot

    c = count_v3_ssot(db, source_ids)
    total = int(c.get("total") or 0)
    counters = {
        a.value: int(c.get(a.value) or 0)
        for a in list(LIGHT_ARTIFACTS) + list(HEAVY_ARTIFACTS) + list(OBJECT_CONCEPT_ARTIFACTS) + list(OWLV2_ARTIFACTS) + list(POST_GA_ARTIFACTS)
    }
    light_ok = int(c.get("light_complete") or 0)
    ai_ok = int(c.get("ai_final") or 0)
    preview_wait = int(c.get("waiting_preview") or 0)
    stages = {name: StageProgress(name, n, total) for name, n in counters.items()}
    stages["light"] = StageProgress("light", light_ok, total)
    stages["ai_final"] = StageProgress("ai_final", ai_ok, total)
    return ProgressSnapshot(
        total=total,
        stages=stages,
        light_complete=light_ok,
        ai_final=ai_ok,
        preview_waiting=preview_wait,
    )


def session_delta(
    current: ProgressSnapshot, baseline: ProgressSnapshot
) -> dict[str, int]:
    """session_completed = current_completed - session_start_completed."""
    out: dict[str, int] = {}
    keys = set(current.stages) | set(baseline.stages)
    for key in keys:
        out[key] = int(current.stage(key).completed) - int(
            baseline.stage(key).completed
        )
    out["light"] = current.light_complete - baseline.light_complete
    out["ai_final"] = current.ai_final - baseline.ai_final
    return out
