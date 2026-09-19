"""Büyük arşivler için aday daraltma katmanı."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.logger import setup_logger
from core.utils import phash_similarity, text_search_variants

logger = setup_logger(__name__)


@dataclass
class PrefilterStats:
    total_index: int = 0
    hash_bucket_candidates: int = 0
    texture_family_candidates: int = 0
    color_family_candidates: int = 0
    faiss_dino_candidates: int = 0
    faiss_clip_candidates: int = 0
    text_candidates: int = 0
    feedback_candidates: int = 0
    pattern_group_candidates: int = 0
    exact_hash_candidates: int = 0
    protected_exact_candidates: int = 0
    final_candidate_count: int = 0
    used_prefilter: bool = False
    prefilter_layers: list[str] = field(default_factory=list)


def _phash_bucket(phash: str, bits: int = 12) -> str:
    if not phash:
        return ""
    return phash[: max(1, bits // 4)]


def _hamming_bucket_match(q: str, c: str, max_dist: int = 8) -> bool:
    if not q or not c:
        return False
    if q[:4] == c[:4]:
        return True
    return phash_similarity(q, c) >= (1.0 - max_dist / 64.0)


class CandidatePrefilter:
    def __init__(self, settings):
        self.settings = settings

    def should_prefilter(self, pool_size: int) -> bool:
        if not self.settings.candidate_prefilter_enabled:
            return False
        return pool_size >= self.settings.prefilter_min_index_size

    def narrow_candidates(
        self,
        indexed: list[dict[str, Any]],
        query_features,
        query_profile,
        faiss_store,
        text_query: str = "",
        query_partial_hash: str = "",
        feedback_candidate_ids: set[int] | None = None,
        pattern_group_candidate_ids: set[int] | None = None,
        protected_exact_ids: set[int] | None = None,
    ) -> tuple[list[dict[str, Any]], PrefilterStats]:
        stats = PrefilterStats(total_index=len(indexed))
        if not self.should_prefilter(len(indexed)):
            stats.final_candidate_count = len(indexed)
            return indexed, stats

        stats.used_prefilter = True
        layers: list[str] = []
        candidate_ids: set[int] = set(protected_exact_ids or ())
        stats.protected_exact_candidates = len(protected_exact_ids or ())
        if protected_exact_ids:
            layers.append("protected_exact")

        # 1. Exact/partial hash
        if query_partial_hash:
            exact_ids: set[int] = set()
            for rec in indexed:
                if rec.get("partial_hash") == query_partial_hash:
                    exact_ids.add(rec["id"])
            candidate_ids.update(exact_ids)
            stats.exact_hash_candidates = len(exact_ids)
            if exact_ids:
                layers.append("exact_hash")

        # 2. pHash bucket — single O(n) prefix pass (no second full scan).
        max_candidates = max(
            200,
            int(getattr(self.settings, "prefilter_faiss_top_k", 800) or 800),
        )
        q_phash = getattr(query_features, "phash", "") or ""
        primary_ids: set[int] = set()
        secondary_ids: set[int] = set()
        if q_phash:
            bucket = _phash_bucket(q_phash, self.settings.prefilter_hash_bucket_bits)
            bucket2 = _phash_bucket(
                q_phash, max(8, self.settings.prefilter_hash_bucket_bits - 4)
            )
            for rec in indexed:
                c_phash = rec.get("phash", "") or ""
                if not c_phash:
                    continue
                if c_phash.startswith(bucket):
                    primary_ids.add(rec["id"])
                elif c_phash.startswith(bucket2):
                    secondary_ids.add(rec["id"])
            candidate_ids.update(primary_ids)
            stats.hash_bucket_candidates = len(primary_ids)
            layers.append("phash_bucket")
            if len(candidate_ids) < max(40, max_candidates // 4) and secondary_ids:
                for sid in secondary_ids:
                    candidate_ids.add(sid)
                    if len(candidate_ids) >= max_candidates * 2:
                        break
                layers.append("fallback_hash_expand")

        # Texture/color full-index scans are O(n) over 100k+ rows and alone can
        # exceed the progressive 500ms budget. Family preference is applied in
        # quick_rank among hash-bucket candidates instead.
        q_family = getattr(query_profile, "pattern_family", "") or ""
        q_animal = getattr(query_profile, "animal_print_type", "") or ""
        q_color = getattr(query_profile, "color_family", "") or ""

        # 5. FAISS
        if faiss_store and faiss_store.available and self.settings.ai_embedding_enabled:
            k = self.settings.prefilter_faiss_top_k
            if query_features.dino_embedding:
                try:
                    if len(query_features.dino_embedding) != 384 * 4:
                        raise ValueError("invalid_dino_query_embedding")
                    vec = np.frombuffer(query_features.dino_embedding, dtype=np.float32)
                except (TypeError, ValueError):
                    logger.warning("Ön filtre: geçersiz DINO sorgu embedding'i; atlandı")
                else:
                    dino_ids = {fid for fid, _ in faiss_store.search_dino(vec, k=k)}
                    candidate_ids.update(dino_ids)
                    stats.faiss_dino_candidates = len(dino_ids)
                    layers.append("faiss_dino")
            if query_features.clip_embedding:
                try:
                    if len(query_features.clip_embedding) != 512 * 4:
                        raise ValueError("invalid_clip_query_embedding")
                    vec = np.frombuffer(query_features.clip_embedding, dtype=np.float32)
                except (TypeError, ValueError):
                    logger.warning("Ön filtre: geçersiz CLIP sorgu embedding'i; atlandı")
                else:
                    clip_ids = {fid for fid, _ in faiss_store.search_clip(vec, k=k)}
                    candidate_ids.update(clip_ids)
                    stats.faiss_clip_candidates = len(clip_ids)
                    layers.append("faiss_clip")

        # 6. Text filename/OCR
        if text_query:
            text_ids: set[int] = set()
            tq = text_query.lower()
            for variant in text_search_variants(text_query):
                v = variant.lower()
                for rec in indexed:
                    fname = (rec.get("filename") or "").lower()
                    ocr = (rec.get("ocr_text") or "").lower()
                    if v in fname or v in ocr:
                        text_ids.add(rec["id"])
            candidate_ids.update(text_ids)
            stats.text_candidates = len(text_ids)
            layers.append("text")

        feedback_ids = set(feedback_candidate_ids or ())
        group_ids = set(pattern_group_candidate_ids or ())
        candidate_ids.update(feedback_ids)
        candidate_ids.update(group_ids)
        stats.feedback_candidates = len(feedback_ids)
        stats.pattern_group_candidates = len(group_ids)
        if feedback_ids:
            layers.append("feedback")
        if group_ids:
            layers.append("pattern_group")

        if not candidate_ids:
            stats.final_candidate_count = len(indexed)
            stats.prefilter_layers = layers
            logger.warning("Prefilter boş kaldı, full scan")
            return indexed, stats

        # Hard mandatory only: exact/feedback/group. Same-family stays in the
        # texture_family bucket but MUST remain subject to quality_rank_cap —
        # otherwise common families (floral/animal) explode past the budget and
        # first progressive results miss the 500ms acceptance gate.
        mandatory_ids = set(protected_exact_ids or ()) | feedback_ids | group_ids
        q_dhash = getattr(query_features, "dhash", "") or ""
        q_whash = getattr(query_features, "whash", "") or ""

        def quick_rank(rec: dict[str, Any]) -> float:
            tm = rec.get("texture_map") or {}
            if isinstance(tm, str):
                import json

                try:
                    tm = json.loads(tm)
                except Exception:
                    tm = {}
            # Single hash compare on hot path — dhash/whash added only for ties.
            rank = phash_similarity(q_phash, rec.get("phash", "")) * 0.8
            if q_family and tm.get("pattern_family") == q_family:
                rank += 0.08
            if q_animal and tm.get("animal_print_type") == q_animal:
                rank += 0.08
            if q_color and tm.get("color_family") == q_color:
                rank += 0.02
            return rank

        id_map = {rec["id"]: rec for rec in indexed}
        if len(candidate_ids) > max_candidates:
            import heapq

            room = max(0, max_candidates - len(mandatory_ids))
            pool_iter = (
                rec
                for rec in indexed
                if rec["id"] in candidate_ids and rec["id"] not in mandatory_ids
            )
            ranked = heapq.nlargest(room, pool_iter, key=quick_rank) if room else []
            narrowed = [id_map[fid] for fid in mandatory_ids if fid in id_map] + ranked
            layers.append("quality_rank_cap")
        else:
            narrowed = sorted(
                (id_map[fid] for fid in candidate_ids if fid in id_map),
                key=quick_rank,
                reverse=True,
            )
        stats.final_candidate_count = len(narrowed)
        stats.prefilter_layers = layers
        logger.info(
            "Prefilter: %d → %d aday (%s)",
            len(indexed),
            len(narrowed),
            ",".join(layers),
        )
        return narrowed, stats
