"""Conservative cross-format Project Search candidate grouping."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.db import Database
from core.pattern_groups import RELATION_FORMAT, PatternGroupManager


@dataclass(frozen=True)
class ProjectCandidate:
    group_id: str
    file_ids: tuple[int, ...]
    confidence: float
    reason: str
    directory: str
    base_filename: str


def _normalized_stem(filename: str) -> str:
    return Path(filename or "").stem.strip().casefold()


class ProjectSearchManager:
    """Creates review candidates; it never merges pattern groups automatically."""

    def __init__(self, db: Database):
        self.db = db

    def discover_candidates(self, records: Iterable[dict[str, Any]]) -> list[ProjectCandidate]:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in records:
            path = str(record.get("path") or "")
            filename = str(record.get("filename") or os.path.basename(path))
            stem = _normalized_stem(filename)
            if not path or not stem or not record.get("id"):
                continue
            directory = os.path.normcase(os.path.normpath(os.path.dirname(path)))
            buckets.setdefault((directory, stem), []).append(record)

        candidates: list[ProjectCandidate] = []
        for (directory, stem), members in buckets.items():
            if len(members) < 2:
                continue
            extensions = {Path(str(row.get("filename") or row.get("path") or "")).suffix.casefold() for row in members}
            confidence = 0.80
            reasons = ["same_base_filename", "same_folder"]
            customers = {str(row.get("customer") or "").strip().casefold() for row in members}
            customers.discard("")
            if len(customers) == 1:
                confidence += 0.05
                reasons.append("same_customer")
            mtimes = [float(row.get("mtime") or 0) for row in members if float(row.get("mtime") or 0) > 0]
            if len(mtimes) >= 2 and max(mtimes) - min(mtimes) <= 7 * 24 * 3600:
                confidence += 0.05
                reasons.append("near_modified_time")
            if len(extensions) > 1:
                confidence += 0.05
                reasons.append("cross_format")
            phashes = [str(row.get("phash") or "") for row in members]
            phashes = [value for value in phashes if value]
            if len(phashes) >= 2 and self._hashes_are_similar(phashes):
                confidence += 0.05
                reasons.append("similar_thumbnail_hash")
            digest = hashlib.sha1(f"{directory}|{stem}".encode("utf-8")).hexdigest()[:20]
            candidates.append(
                ProjectCandidate(
                    group_id=f"candidate:{digest}",
                    file_ids=tuple(sorted(int(row["id"]) for row in members)),
                    confidence=min(confidence, 0.95),
                    reason=",".join(reasons),
                    directory=directory,
                    base_filename=stem,
                )
            )
        return candidates

    @staticmethod
    def _hashes_are_similar(values: list[str], max_distance: int = 8) -> bool:
        try:
            numbers = [int(value, 16) for value in values]
        except (TypeError, ValueError):
            return False
        widths = {len(value) for value in values}
        if len(widths) != 1:
            return False
        return max((left ^ right).bit_count() for left in numbers for right in numbers) <= max_distance

    def refresh_directory(self, directory: str) -> list[ProjectCandidate]:
        candidates = self.discover_candidates(self.db.list_files_in_directory(directory))
        for candidate in candidates:
            self.db.set_project_candidate(
                list(candidate.file_ids),
                candidate.group_id,
                candidate.confidence,
                candidate.reason,
                status="candidate",
            )
        return candidates

    def refresh_for_file(self, file_id: int) -> list[ProjectCandidate]:
        record = self.db.get_file_by_id(file_id)
        if not record:
            return []
        path = str(record.get("path") or "")
        directory = os.path.dirname(path)
        stem = _normalized_stem(str(record.get("filename") or os.path.basename(path)))
        candidates = self.discover_candidates(
            self.db.list_project_siblings(directory, stem)
        )
        for candidate in candidates:
            self.db.set_project_candidate(
                list(candidate.file_ids),
                candidate.group_id,
                candidate.confidence,
                candidate.reason,
                status="candidate",
            )
        return candidates

    def approve_candidate(
        self,
        group_id: str,
        *,
        admin_approved: bool = False,
        actor: str = "",
    ) -> int:
        if not admin_approved:
            raise PermissionError("Kalıcı proje birleştirmesi admin onayı gerektirir")
        members = self.db.get_project_candidate_members(group_id)
        if len(members) < 2:
            raise ValueError("Proje adayı bulunamadı veya yetersiz üye var")
        manager = PatternGroupManager(self.db)
        representative = int(members[0]["id"])
        persistent_group_id = 0
        for member in members[1:]:
            persistent_group_id = manager.add_or_strengthen(
                representative,
                int(member["id"]),
                RELATION_FORMAT,
                float(member.get("project_group_confidence") or 0),
                group_type="admin_project",
                label=Path(str(members[0].get("filename") or "")).stem,
            )
        self.db.set_project_candidate(
            [int(row["id"]) for row in members],
            group_id,
            float(members[0].get("project_group_confidence") or 0),
            f"{members[0].get('project_group_reason') or ''},approved_by:{actor or 'admin'}",
            status="approved",
        )
        return persistent_group_id
