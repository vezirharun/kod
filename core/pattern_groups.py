"""Kalıcı desen grupları — aynı desen ailesi tekrar keşfedilmesin."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.db import Database
from core.logger import setup_logger

logger = setup_logger(__name__)

RELATION_EXACT = "exact"
RELATION_FORMAT = "format_variant"
RELATION_RESOLUTION = "resolution_variant"
RELATION_CROP = "crop_variant"
RELATION_COLOR = "color_variant"
RELATION_SIMILAR = "similar_texture"
RELATION_RELATED = "related_family"


class PatternGroupManager:
    def __init__(self, db: Database):
        self.db = db

    def find_group_for_file(self, file_id: int) -> dict[str, Any] | None:
        return self.db.get_pattern_group_for_file(file_id)

    def add_or_strengthen(
        self,
        representative_file_id: int,
        member_file_id: int,
        relation: str,
        score: float = 0.0,
        *,
        group_type: str = "auto",
        label: str = "",
        pattern_family: str = "",
        animal_print_type: str = "",
        color_family: str = "",
    ) -> int:
        """İki dosyayı aynı gruba ekle veya mevcut grubu güçlendir."""
        rep_group = self.db.get_pattern_group_for_file(representative_file_id)
        mem_group = self.db.get_pattern_group_for_file(member_file_id)

        if rep_group and mem_group and rep_group["id"] != mem_group["id"]:
            # Basit birleştirme: küçük grubu büyüğe taşı
            target = rep_group if rep_group["id"] < mem_group["id"] else mem_group
            source = mem_group if target["id"] == rep_group["id"] else rep_group
            self.db.merge_pattern_groups(source["id"], target["id"])
            group_id = target["id"]
        elif rep_group:
            group_id = rep_group["id"]
        elif mem_group:
            group_id = mem_group["id"]
        else:
            group_id = self.db.create_pattern_group(
                {
                    "representative_file_id": representative_file_id,
                    "group_type": group_type,
                    "label": label,
                    "pattern_family": pattern_family,
                    "animal_print_type": animal_print_type,
                    "color_family": color_family,
                }
            )

        now = datetime.now(timezone.utc).isoformat()
        self.db.add_pattern_group_member(
            group_id, representative_file_id, relation, score, now
        )
        if member_file_id != representative_file_id:
            self.db.add_pattern_group_member(
                group_id, member_file_id, relation, score, now
            )
        return group_id

    def members_of(self, group_id: int) -> list[dict[str, Any]]:
        return self.db.list_pattern_group_members(group_id)

    def groups_for_search_results(
        self,
        file_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        return self.db.get_pattern_groups_for_files(file_ids)
