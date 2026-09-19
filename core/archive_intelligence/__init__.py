"""Archive Intelligence — soft self-audit of classifications.

Suggests REVIEW / ÖĞRET only. Never auto-deletes, never auto-changes
category/concept, never wipes learned data. Runs at low priority after
search / index / preview work.
"""

from __future__ import annotations

from core.archive_intelligence.consistency import (
    ConsistencyReport,
    score_file_consistency,
)
from core.archive_intelligence.family_graph import (
    family_snapshot,
    may_merge_as_same_family,
)
from core.archive_intelligence.hooks import (
    merge_archive_candidates_into_pools,
    resolve_candidates_for_files,
)
from core.archive_intelligence.scanner import ArchiveIntelligenceScanner
from core.archive_intelligence.store import ArchiveIntelligenceStore
from core.archive_intelligence.time_fields import labeled_time_fields

__all__ = [
    "ArchiveIntelligenceScanner",
    "ArchiveIntelligenceStore",
    "ConsistencyReport",
    "family_snapshot",
    "labeled_time_fields",
    "may_merge_as_same_family",
    "merge_archive_candidates_into_pools",
    "resolve_candidates_for_files",
    "score_file_consistency",
]
