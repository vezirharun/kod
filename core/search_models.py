"""Arama sorgusu, istatistik ve yanıt modelleri."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchQuery:
    """Tek arama isteği."""

    mode: str = "image"  # image | text | hybrid
    image_path: str = ""
    text: str = ""
    crop_rect: tuple[int, int, int, int] | None = None  # x, y, w, h piksel
    use_crop: bool = False
    customer: str = ""
    threshold: float = 0.50
    fast_only: bool = False
    category_path_filter: str = ""
    image_paths: list[str] = field(default_factory=list)


@dataclass
class SearchStats:
    total_indexed: int = 0
    sources_searched: int = 0
    selected_source_files: int = 0
    supported_image_files: int = 0
    thumbnail_files: int = 0
    feature_files: int = 0
    candidates_evaluated: int = 0
    above_threshold: int = 0
    displayed: int = 0
    remaining: int = 0
    near_below_threshold: int = 0
    search_ms: float = 0.0
    used_crop: bool = False
    text_query: str = ""
    query_path: str = ""
    prefilter_candidates: int = 0
    prefilter_used: bool = False

    def summary_lines(self) -> list[str]:
        lines = [
            f"Toplam index: {self.total_indexed:,} dosya",
            f"Aranan kaynak: {self.sources_searched}",
            f"Seçili kaynak dosyası: {self.selected_source_files:,}",
            f"Desteklenen görsel: {self.supported_image_files:,}",
            f"Thumbnail olan: {self.thumbnail_files:,}",
            f"Feature olan: {self.feature_files:,}",
        ]
        if self.prefilter_used:
            lines.append(f"Prefilter aday: {self.prefilter_candidates:,}")
        lines.extend(
            [
                f"Skorlanan aday: {self.candidates_evaluated:,}",
                f"Eşik üstü: {self.above_threshold:,}",
                f"Gösterilen: {self.displayed:,}",
                f"Yakın ama eşik altı: {self.near_below_threshold:,}",
            ]
        )
        if self.remaining > 0:
            lines.append(f"Daha fazla sonuç: {self.remaining:,}")
        return lines


@dataclass
class SearchResponse:
    """Tam arama yanıtı — UI filtreleme ve sayfalama için."""

    results: list[Any] = field(default_factory=list)
    all_results: list[Any] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)
    below_threshold_preview: list[Any] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _passes_display_threshold(result: Any, threshold: float) -> bool:
        """Apply the same acceptance domain used by SearchEngine.

        SearchEngine already separates human/gender retrieval from the normal
        textile threshold.  The UI must not re-apply the generic 0.60 threshold
        to ``all_results`` or a valid ``kadın/erkek`` search is turned back into
        zero results after the worker has completed successfully.
        """
        dbg = getattr(result, "debug", {}) or {}
        if dbg.get("face_gender_match"):
            return True
        if dbg.get("human_semantic_mode") or dbg.get("human_semantic_only"):
            try:
                human_score = float(
                    dbg.get(
                        "human_semantic_score",
                        dbg.get("gender_visual_score", 0.0),
                    )
                    or 0.0
                )
            except (TypeError, ValueError):
                return False
            return human_score >= 0.18
        if dbg.get("protected_exact"):
            return True
        if dbg.get("learned_concept_exact") or dbg.get("learned_concept"):
            return True
        if dbg.get("user_taught_positive"):
            return True
        try:
            return float(getattr(result, "score", 0.0) or 0.0) >= float(threshold)
        except (TypeError, ValueError):
            return False

    def filter_by_threshold(self, threshold: float) -> list[Any]:
        return [
            r for r in self.all_results
            if self._passes_display_threshold(r, threshold)
        ]

    def count_above(self, threshold: float) -> int:
        return sum(
            1 for r in self.all_results
            if self._passes_display_threshold(r, threshold)
        )


def results_for_display(response: SearchResponse, threshold: float, *, ranked: bool = False) -> list[Any]:
    """Count and list must share this source. Never advertise hits that are not shown."""
    if ranked:
        shown = list(response.results or [])
    else:
        shown = response.filter_by_threshold(threshold)
        if not shown:
            shown = list(response.results or [])
    return shown
