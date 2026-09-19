"""Preview / thumb self-healing — light validation + targeted repair.

Does not rewrite Preview Pool or Index. Solid monochrome is VALID when the
source matches; only misrepresenting / corrupt / missing artifacts are repaired.
"""

from __future__ import annotations

from core.preview_self_heal.hooks import (
    gate_new_preview,
    heal_candidates_for_sources,
    invalidate_if_bad_preview,
)
from core.preview_self_heal.repair import RepairOutcome, repair_preview_artifact
from core.preview_self_heal.validate import (
    Verdict,
    ValidationResult,
    compare_source_vs_preview,
    validate_artifact_light,
    validate_preview,
)

__all__ = [
    "Verdict",
    "ValidationResult",
    "validate_artifact_light",
    "compare_source_vs_preview",
    "validate_preview",
    "RepairOutcome",
    "repair_preview_artifact",
    "gate_new_preview",
    "invalidate_if_bad_preview",
    "heal_candidates_for_sources",
]
