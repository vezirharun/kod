"""Benzer desen arama motoru — iki kademeli skor (hash önce, doku/AI sonra)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.candidate_prefilter import CandidatePrefilter, PrefilterStats
from core.db import Database
from core.dynamic_groups import QueryContext
from core.exact_guard import (
    collect_protected_exact_ids,
    is_protected_exact_match,
    merge_candidate_records,
    protected_score_floor,
)
from core.faiss_store import (
    CLIP_MAP_UNAVAILABLE,
    VISUAL_SIM_FALLBACK,
    FaissStore,
)
from core.feature_extractor import ExtractedFeatures, FeatureExtractor
from core.group_gates import candidate_family_display_label, enforce_cluster_gate
from core.hash_verify import apply_verified_hashes, verified_hashes_for_record
from core.logger import setup_logger
from core.multiscale_patch import multiscale_patch_similarity

def _visual_candidate_ids(existing_rows: list[dict[str, Any]], visual_ids: Any, *, limit: int = 800) -> list[int]:
    """Return visual/AI hit ids that are not already in the SQL candidate set.

    Retrieval-only helper: it does not score, classify, or accept a result.
    The normal search gates run after these records are loaded.
    """
    existing = {int(r.get("id", 0) or 0) for r in (existing_rows or [])}
    out: list[int] = []
    for raw in (visual_ids or {}):
        try:
            fid = int(raw)
        except (TypeError, ValueError):
            continue
        if fid > 0 and fid not in existing:
            out.append(fid)
        if len(out) >= max(1, int(limit)):
            break
    return out

from core.pattern_classifier import (
    ANIMAL_PRINT_MIN_CONFIDENCE,
    effective_query_family,
    is_confident_animal_print,
)
from core.adaptive_retrieve import (
    is_expand_worthy,
    is_relevant_hit,
    next_retrieve_k,
    should_stop_expansion,
)
from core.search_latency import NullLatency, SearchLatencyProfiler, classify_storage
from core.search_models import SearchQuery, SearchResponse, SearchStats
from core.settings import SUPPORTED_EXTENSIONS, AppSettings
from core.sources import SearchFilter, SearchScope
from core.text_score import filename_text_boost_only, text_match_score
from core.texture_profile import (
    CLUSTER_DISTANT,
    CLUSTER_LABELS,
    CLUSTER_LEOPARD_OTHER_COLOR,
    CLUSTER_LEOPARD_SAME_COLOR,
    CLUSTER_RELATED_ANIMAL,
    CLUSTER_UNRELATED,
    LEOPARD_FAMILY,
    TextureProfile,
    animal_family_similarity,
    assign_cluster_group,
    compute_hierarchy_score,
    hierarchy_sort_key,
    palette_similarity,
    texture_family_score,
)
from core.thumbnailer import Thumbnailer
from core.user_feedback import UserFeedbackStore
from core.utils import (
    format_file_size,
    normalize_path,
    partial_file_hash,
    phash_similarity,
)

CLUSTER_GROUP_LABELS = CLUSTER_LABELS
CLUSTER_GROUP_LABELS.update(
    {
        "exact_same": "Aynı Dosyalar",
        "format_variant": "Aynı Dosyalar",
        "resolution_variant": "Aynı Dosyalar",
        "crop_variant": "Aynı Dosyalar",
        "color_variant": "Aynı Desen Ailesi",
        "same_family_close": "Aynı Desen Ailesi",
        "same_family_style": "Benzer Stil",
        "related_family": "Benzer Stil",
        "far_texture": "İlham Verebilir",
        "unrelated": "İlham Verebilir",
    }
)

logger = setup_logger(__name__)

NON_TEXTILE_FOR_ANIMAL_QUERY = frozenset(
    {
        "abstract",
        "plain",
        "document",
        "icon_logo_non_textile",
        "floral",
        "geometric",
        "paisley",
        "plaid_check",
        "monogram_logo",
        "texture_ground",
        "unknown",
        "marble_abstract",
        "stripe",
        "baroque",
        "chain",
        "scarf_border",
    }
)

# Sonuç kategorileri
CATEGORY_EXACT = "exact"  # %95-100 Aynı / neredeyse aynı
CATEGORY_NEAR = "near_variant"  # %85-95 Çok yakın varyant
CATEGORY_SIMILAR = "similar"  # %70-85 Benzer desen
CATEGORY_STYLE = "style"  # %50-70 Benzer tarz / doku
CATEGORY_COLOR_VARIANT = "color_variant"
CATEGORY_FORMAT_VARIANT = "format_variant"
CATEGORY_ANIMAL_PRINT = "animal_print"
CATEGORY_TEXTILE_TEXTURE = "textile_texture"
CATEGORY_WEAK = "weak"

CATEGORY_LABELS = {
    CATEGORY_EXACT: "Aynı / Neredeyse Aynı",
    CATEGORY_NEAR: "Çok Yakın Varyant",
    CATEGORY_SIMILAR: "Benzer Desen",
    CATEGORY_STYLE: "Benzer Doku / Tarz",
    CATEGORY_COLOR_VARIANT: "Renk Varyantı",
    CATEGORY_WEAK: "Zayıf ama İlişkili",
}
CATEGORY_LABELS.update(
    {
        CATEGORY_FORMAT_VARIANT: "Ayni Desen - Farkli Format/Cozunurluk",
        CATEGORY_ANIMAL_PRINT: "Benzer Leopard / Animal Print",
        CATEGORY_TEXTILE_TEXTURE: "Benzer Doku / Metraj Desen",
    }
)


@dataclass
class SearchResult:
    file_id: int
    path: str
    filename: str
    customer: str
    thumbnail_path: str
    score: float
    score_percent: float
    feature_preview_path: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    mtime: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    source_name: str = ""
    source_type: str = ""
    category: str = CATEGORY_STYLE
    cluster_group: str = ""
    color_family: str = ""
    pattern_family: str = ""
    animal_print_type: str = ""
    hierarchy_score: float = 0.0
    palette_similarity: float = 0.0
    same_color_family: bool = False
    same_pattern_family: bool = False
    same_animal_family: bool = False
    texture_family_score: float = 0.0
    cluster_reason: str = ""
    text_match_reason: str = ""
    match_explanations: list[str] = field(default_factory=list)
    text_score_breakdown: dict[str, float] = field(default_factory=dict)
    is_self_match: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def file_size_human(self) -> str:
        return format_file_size(self.file_size)

    @property
    def cluster_label(self) -> str:
        learned = str(self.debug.get("learned_tier_label") or "").strip()
        if learned:
            return learned
        return CLUSTER_GROUP_LABELS.get(
            self.cluster_group, self.cluster_group or self.category_label
        )

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.category, self.category)


_QUERY_FEAT_CACHE: dict[tuple, tuple[Any, str, str]] = {}
_QUERY_FEAT_HITS = 0
_QUERY_FEAT_MISSES = 0
_QUERY_FEAT_MAX = 24


def query_features_cache_stats() -> dict[str, int]:
    return {
        "hits": int(_QUERY_FEAT_HITS),
        "misses": int(_QUERY_FEAT_MISSES),
        "size": len(_QUERY_FEAT_CACHE),
    }


def clear_query_features_cache() -> None:
    global _QUERY_FEAT_HITS, _QUERY_FEAT_MISSES
    _QUERY_FEAT_CACHE.clear()
    _QUERY_FEAT_HITS = 0
    _QUERY_FEAT_MISSES = 0


def _query_features_cache_key(
    image_path: str,
    crop_rect: tuple[int, int, int, int] | None,
    fast_only: bool,
) -> tuple:
    norm = normalize_path(image_path)
    try:
        st = os.stat(norm)
        sig = (int(st.st_size), int(st.st_mtime))
    except OSError:
        sig = (0, 0)
    crop = tuple(int(x) for x in crop_rect) if crop_rect else None
    return (norm, crop, bool(fast_only), sig)


class SearchEngine:
    def __init__(self, settings: AppSettings, *, load_ai: bool | None = None):
        self.settings = settings
        # SearchWorker ayrı read-only SQLite connection — Health/Status ile paylaşılmaz
        self.db = Database(settings.db_path, read_only=True)
        self.faiss = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
        self._prefilter = CandidatePrefilter(settings)
        self._feedback = UserFeedbackStore(self.db)
        self._indexed_pool_cache: dict[tuple, list[dict[str, Any]]] = {}
        self._last_scope_fallback: str = ""
        self._last_text_intel_meta: dict[str, Any] = {}
        self._last_query_intent = None
        self._last_clip_rivals: dict[int, dict[str, float]] = {}
        self._last_v2_concepts: dict[int, dict[str, float]] = {}
        self._last_v2_rep: dict[int, dict[str, float]] = {}
        # Aynı metin sorgusunu tekrar tekrar CLIP tokenizer/modelinden geçirme.
        # Özellikle "marka", "çiçek", "amiri" gibi sık kullanılan sorgular
        # ikinci aramada yalnızca FAISS/DB yolunu kullanır.
        self._text_embedding_cache: dict[str, np.ndarray] = {}
        self._text_embedding_cache_max = 64
        # Müşteri kapsamı: path keşfi (files.customer kolonuna bağımlı değil)
        self._customer_registry = None
        from core.index_freeze import INDEX_FROZEN

        if not INDEX_FROZEN:
            self._maybe_rebuild_faiss()
        use_ai = settings.ai_embedding_enabled
        if load_ai is not None:
            use_ai = use_ai and load_ai
        self.extractor = FeatureExtractor(
            use_ai=use_ai,
            use_gpu=settings.use_gpu,
            fast_hash_only=False,
        )
        self._ai_load_error = ""
        self._last_pipeline_audit: dict[str, Any] = {}
        self._last_retrieve_meta: dict[str, Any] = {}
        self._latency: NullLatency | SearchLatencyProfiler = NullLatency()
        self._thumbnailer = Thumbnailer(
            settings.cache_dir,
            settings.thumbnail_max_edge,
            settings.thumbnail_format,
        )

    def ensure_ai_loaded(self) -> bool:
        """İkinci aşama aramadan önce DINO/CLIP modellerini yükle."""
        if not self.settings.ai_embedding_enabled:
            self._ai_load_error = "AI embedding ayarlarda kapalı"
            return False
        if not self.ai_index_ready():
            self._ai_load_error = (
                f"FAISS indeksi yetersiz "
                f"(dino={self.faiss.dino_count}, clip={self.faiss.clip_count})"
            )
            return False
        if (
            self.extractor.use_ai
            and (
                getattr(self.extractor, "_dino_model", None) is not None
                or getattr(self.extractor, "_clip_model", None) is not None
            )
        ):
            self._ai_load_error = ""
            return True
        from core.capability_check import get_shared_ai_extractor, try_load_ai_extractor

        ok, msg = try_load_ai_extractor(self.settings)
        if not ok:
            self._ai_load_error = msg or "AI modeli yüklenemedi"
            logger.warning(self._ai_load_error)
            self.extractor = FeatureExtractor(
                use_ai=False,
                use_gpu=self.settings.use_gpu,
                fast_hash_only=False,
            )
            return False
        ext = get_shared_ai_extractor(self.settings)
        if ext is not None:
            self.extractor = ext
        else:
            # Cache miss edge case — should not construct a second AI extractor
            self._ai_load_error = "Shared AI extractor missing after successful load"
            logger.warning(self._ai_load_error)
            self.extractor = FeatureExtractor(
                use_ai=False,
                use_gpu=self.settings.use_gpu,
                fast_hash_only=False,
            )
            return False
        if not self.extractor.ai_available:
            self._ai_load_error = "FeatureExtractor yüklendi ama model bellekte yok"
            return False
        self._ai_load_error = ""
        return True

    def start_latency_profile(self) -> SearchLatencyProfiler:
        """Benchmark only — ranking path unchanged."""
        self._latency = SearchLatencyProfiler()
        return self._latency

    def stop_latency_profile(self) -> dict[str, Any]:
        snap = self._latency.snapshot() if getattr(self._latency, "enabled", False) else {}
        self._latency = NullLatency()
        return snap

    def _lat_add(self, name: str, t0: float) -> float:
        lat = self._latency
        if getattr(lat, "enabled", False):
            lat.add(name, (time.perf_counter() - t0) * 1000.0)
            return time.perf_counter()
        return t0

    @staticmethod
    def _engine_sort_key(result: SearchResult) -> tuple:
        """Exact engine her zaman family engine'in önünde kalır."""
        dbg = result.debug or {}
        layer = str(dbg.get("result_layer") or "")
        exact = float(
            dbg.get("exact_score", dbg.get("exact_search_score", 0)) or 0
        )
        family = float(
            dbg.get("family_variant_score", dbg.get("pattern_family_score", 0)) or 0
        )
        if result.is_self_match:
            return (0, -1.0, -1.0, -float(result.score), int(result.file_id))
        if dbg.get("face_match"):
            face_score = float(dbg.get("face_similarity", 0.0) or 0.0)
            return (1, -face_score, -float(result.score), -float(result.hierarchy_score or 0), int(result.file_id))
        if dbg.get("learned_concept_exact") or dbg.get("user_taught_positive"):
            return (2, -2.0, -float(result.score), -float(result.hierarchy_score or 0), int(result.file_id))
        if dbg.get("learned_concept"):
            return (2, -1.5, -float(result.score), -float(result.hierarchy_score or 0), int(result.file_id))
        if dbg.get("protected_exact") or layer == "same_files":
            return (2, -exact, -float(result.score), -float(result.hierarchy_score or 0), int(result.file_id))
        if layer == "same_pattern_family":
            return (3, -family, -float(result.score), -float(result.hierarchy_score or 0), int(result.file_id))
        if layer == "similar_patterns":
            return (4, -family, -float(result.score), -float(result.hierarchy_score or 0), int(result.file_id))
        return (5, -float(result.score), -family, -float(result.hierarchy_score or 0), int(result.file_id))

    def ai_index_ready(self) -> bool:
        minimum = max(
            1, int(getattr(self.settings, "ai_search_min_vectors", 500) or 500)
        )
        if getattr(self.faiss, "clip_map_unavailable", False):
            return self.faiss.dino_count >= minimum
        if getattr(self.faiss, "dino_map_unavailable", False):
            return self.faiss.clip_count >= minimum
        return self.faiss.dino_count >= minimum and self.faiss.clip_count >= minimum

    def _maybe_rebuild_faiss(self) -> None:
        """DB embedding SSOT; bellek/disk FAISS gerideyse mevcut store ile rebuild."""
        from core.index_freeze import INDEX_FROZEN, guard_index_write

        if INDEX_FROZEN:
            return
        guard_index_write("faiss.rebuild_from_db", "core.search_engine")
        if not self.faiss.available or not self.settings.ai_embedding_enabled:
            return
        if getattr(self.faiss, "clip_map_unavailable", False) or getattr(
            self.faiss, "dino_map_unavailable", False
        ):
            logger.warning(
                "FAISS rebuild skipped: %s",
                getattr(self.faiss, "clip_map_error", "")
                or getattr(self.faiss, "dino_map_error", "")
                or "id map unavailable",
            )
            return
        clip_pos = [int(x) for x in (self.faiss.clip_id_map or []) if int(x) > 0]
        clip_unmapped = sum(1 for x in (self.faiss.clip_id_map or []) if int(x) <= 0)
        if clip_pos and clip_unmapped:
            logger.warning(
                "FAISS rebuild skipped: partial recovered CLIP map (%s mapped, %s unmapped)",
                len(clip_pos),
                clip_unmapped,
            )
            return
        try:
            with self.db.connect() as conn:
                dino_db = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM features fe
                        JOIN files f ON f.id=fe.file_id
                        WHERE f.status NOT IN ('excluded_internal','missing')
                          AND fe.dino_embedding IS NOT NULL
                          AND length(fe.dino_embedding)>0
                        """
                    ).fetchone()[0]
                )
                clip_db = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM features fe
                        JOIN files f ON f.id=fe.file_id
                        WHERE f.status NOT IN ('excluded_internal','missing')
                          AND fe.clip_embedding IS NOT NULL
                          AND length(fe.clip_embedding)>0
                        """
                    ).fetchone()[0]
                )
        except Exception:
            dino_db = clip_db = -1
        flag = ""
        try:
            flag = str(self.db.get_meta("faiss_needs_rebuild") or "")
        except Exception:
            flag = ""
        need = bool(self.faiss.needs_rebuild) or flag in ("1", "true", "True")
        if (
            self.faiss.dino_count == 0
            and self.faiss.clip_count == 0
            and (dino_db > 0 or clip_db > 0)
        ):
            need = True
        # Count mismatch is not a rebuild trigger — partial recovered CLIP maps
        # must not rewrite FAISS from DB.
        if not need:
            return
        try:
            with self.db.connect() as conn:
                with_emb = [
                    dict(r)
                    for r in conn.execute(
                        """
                        SELECT f.id AS id, fe.dino_embedding, fe.clip_embedding
                        FROM files f
                        JOIN features fe ON fe.file_id = f.id
                        WHERE f.status NOT IN ('excluded_internal','missing')
                          AND (
                            (fe.dino_embedding IS NOT NULL AND length(fe.dino_embedding)>0)
                            OR (fe.clip_embedding IS NOT NULL AND length(fe.clip_embedding)>0)
                          )
                        """
                    ).fetchall()
                ]
        except Exception:
            with_emb = []
        if with_emb or need:
            self.faiss.rebuild_from_db(with_emb)
            try:
                self.db.set_meta("faiss_needs_rebuild", "0")
            except Exception:
                pass

    def _feature_image_path(self, image_path: str) -> str:
        """DB'de feature preview varsa onu kullan."""
        rec = self.db.get_file_by_path(image_path)
        if rec:
            fp = rec.get("feature_preview_path", "")
            if fp and os.path.isfile(fp):
                return fp
            thumb = rec.get("thumbnail_path", "")
            if thumb and os.path.isfile(thumb):
                return thumb
        return image_path

    def _search_filter(self, customer: str = "") -> SearchFilter:
        scope = self.settings.search_scope
        cust = customer or self.settings.customer_filter
        if scope == SearchScope.CUSTOMER.value:
            return SearchFilter(scope=scope, customer=cust)

        folder_path = ""
        if scope == SearchScope.QUICK_FOLDER.value:
            folder_path = self._folder_as_directory(self.settings.quick_search_folder)
        elif scope == SearchScope.SELECTED_FOLDER.value:
            folder_path = self._folder_as_directory(self.settings.selected_folder_path)

        use_source_id = (
            scope
            not in (
                SearchScope.ALL.value,
                SearchScope.SELECTED_SOURCES.value,
                SearchScope.QUICK_FOLDER.value,
                SearchScope.SELECTED_FOLDER.value,
            )
            and self.settings.selected_source_id
        )
        return SearchFilter(
            scope=scope,
            customer="",
            source_id=self.settings.selected_source_id if use_source_id else 0,
            source_ids=(
                list(self.settings.selected_source_ids or [])
                if scope == SearchScope.SELECTED_SOURCES.value
                else []
            ),
            folder_path=folder_path,
        )

    @staticmethod
    def _folder_as_directory(path: str) -> str:
        """Dosya yolu verilmişse üst klasöre çevir — tek dosya aday hatasını önler."""
        if not path:
            return ""
        p = Path(normalize_path(path))
        if p.suffix and p.is_file():
            return str(p.parent)
        return str(p)

    def _customer_registry_cached(self):
        """Path tabanlı müşteri kataloğu (lazy)."""
        if self._customer_registry is not None:
            return self._customer_registry
        try:
            from core.customer_discovery import get_customer_registry

            self._customer_registry = get_customer_registry(db=self.db)
        except Exception:
            from core.customer_discovery import CustomerRegistry

            self._customer_registry = CustomerRegistry()
        return self._customer_registry

    def refresh_customer_registry(self) -> int:
        """İndeks güncellenince müşteri listesini yenile."""
        from core.customer_discovery import get_customer_registry

        self._customer_registry = get_customer_registry(refresh=True, db=self.db)
        self._indexed_pool_cache.clear()
        return int(getattr(self._customer_registry, "discovered_count", 0) or 0)

    def _path_in_customer_scope(self, path: str, customer: str) -> bool:
        """Müşteri kapsamı: klasör kökü / path prefix (files.customer değil)."""
        cust = (customer or "").strip()
        if not cust:
            return True
        reg = self._customer_registry_cached()
        if reg.get(cust.split(" / ", 1)[0].strip()) or reg.scope_prefixes_for(cust):
            return reg.path_matches(path, cust)
        # Keşifte yoksa: kayıt customer kolonu veya path içinde klasör adı
        rec_fallback = normalize_path(path).casefold()
        needle = cust.casefold()
        if f"\\{needle}\\" in f"\\{rec_fallback}\\" or f"/{needle}/" in f"/{rec_fallback}/":
            return True
        return False

    def _indexed_files(
        self, customer: str = "", *, lightweight: bool = False
    ) -> list[dict[str, Any]]:
        self._last_scope_fallback = ""
        sf = self._search_filter(customer)
        cust = (customer or sf.customer or self.settings.customer_filter or "").strip()
        kwargs: dict[str, Any] = {}
        # Path-tabanlı müşteri filtresi: SQL customer kolonuna yazma (çoğu kayıt boş).
        # SQL'e customer= verme → 0 sonuç. Post-filter ile path prefix kullan.
        types = sf.source_types()
        if types:
            kwargs["source_types"] = types

        if sf.scope == SearchScope.SELECTED_FOLDER.value and sf.folder_path:
            kwargs["folder_prefix"] = normalize_path(sf.folder_path)
        elif sf.scope == SearchScope.QUICK_FOLDER.value and sf.folder_path:
            kwargs["folder_prefix"] = normalize_path(sf.folder_path)
        elif sf.scope == SearchScope.SELECTED_SOURCES.value:
            ids, roots = self._resolve_selected_sources(sf.source_ids)
            kwargs["source_ids"] = ids
            kwargs["source_roots"] = roots
        elif sf.scope == SearchScope.ALL.value:
            pass
        elif sf.scope == SearchScope.CUSTOMER.value:
            pass
        elif sf.source_id:
            kwargs["source_id"] = sf.source_id

        cache_key = (
            cust,
            lightweight,
            sf.scope,
            tuple(sorted(kwargs.get("source_ids", []))),
            tuple(kwargs.get("source_roots", [])),
            kwargs.get("source_id", 0),
            kwargs.get("folder_prefix", ""),
            tuple(kwargs.get("source_types", [])),
        )
        if cache_key in self._indexed_pool_cache:
            return self._indexed_pool_cache[cache_key]

        files = self.db.get_indexed_files(
            **kwargs,
            lightweight=lightweight,
            include_processing_ready=True,
        )
        if cust:
            files = [
                f
                for f in files
                if self._path_in_customer_scope(str(f.get("path") or ""), cust)
            ]
        if (
            not files
            and getattr(self.settings, "search_scope_fallback_on_empty", True)
            and sf.scope
            in (
                SearchScope.SELECTED_FOLDER.value,
                SearchScope.QUICK_FOLDER.value,
            )
        ):
            logger.warning(
                "Arama kapsamı boş (scope=%s klasör=%s) — aktif kaynaklara genişletiliyor",
                sf.scope,
                kwargs.get("folder_prefix", ""),
            )
            self._last_scope_fallback = sf.scope
            files = self.db.get_indexed_files(
                lightweight=lightweight,
                include_processing_ready=True,
            )
        self._indexed_pool_cache[cache_key] = files
        logger.info(
            "Arama aday havuzu: scope=%s kaynak_ids=%s klasör=%s → %d dosya",
            sf.scope,
            kwargs.get("source_ids", []),
            kwargs.get("folder_prefix", ""),
            len(files),
        )
        return files

    def _resolve_selected_sources(
        self,
        source_ids: list[int] | None,
    ) -> tuple[list[int], list[str]]:
        """Seçili kaynak ID + kök yolları; boşsa tüm aktif kaynaklar."""
        ids = list(source_ids or [])
        if not ids:
            ids = [s["id"] for s in self.db.list_sources(active_only=True)]
        roots: list[str] = []
        for sid in ids:
            src = self.db.get_source(sid)
            if src and src.get("root_path"):
                roots.append(normalize_path(src["root_path"]))
        return ids, roots

    def _record_in_scope(self, rec: dict[str, Any], customer: str = "") -> bool:
        sf = self._search_filter(customer)
        cust = (customer or sf.customer or "").strip()
        if sf.scope == SearchScope.CUSTOMER.value and (sf.customer or cust):
            return self._path_in_customer_scope(
                str(rec.get("path") or ""), sf.customer or cust
            )
        if (
            sf.scope
            in (
                SearchScope.SELECTED_FOLDER.value,
                SearchScope.QUICK_FOLDER.value,
            )
            and sf.folder_path
        ):
            prefix = normalize_path(sf.folder_path)
            if not prefix.endswith(("\\", "/")):
                prefix += os.sep
            return normalize_path(rec.get("path", "")).startswith(prefix)
        if sf.scope == SearchScope.SELECTED_SOURCES.value:
            ids, roots = self._resolve_selected_sources(sf.source_ids)
            if rec.get("source_id") in ids:
                if cust:
                    return self._path_in_customer_scope(str(rec.get("path") or ""), cust)
                return True
            rec_path = normalize_path(rec.get("path", ""))
            for root in roots:
                r = root.rstrip("\\/")
                if rec_path.startswith(r + os.sep) or rec_path == r:
                    if cust:
                        return self._path_in_customer_scope(rec_path, cust)
                    return True
            return False
        types = sf.source_types()
        if types:
            src = self.db.get_source(rec.get("source_id", 0) or 0)
            if src and src.get("source_type") not in types:
                return False
        if sf.source_id and rec.get("source_id") != sf.source_id:
            return False
        if cust:
            # Path tabanlı SEARCH SCOPE — kolon eşleşmesi değil
            return self._path_in_customer_scope(str(rec.get("path") or ""), cust)
        return True

    def _query_features_from_image(
        self,
        image_path: str,
        crop_rect: tuple[int, int, int, int] | None = None,
        fast_only: bool = False,
    ) -> tuple[ExtractedFeatures, str, str]:
        """Index ile aynı pipeline: thumbnail → feature (+ opsiyonel crop)."""
        lat = self._latency
        lat_on = getattr(lat, "enabled", False)
        norm_path = normalize_path(image_path)
        if lat_on:
            lat.set("query_path", norm_path)
            lat.set("query_storage", classify_storage(norm_path))
        cache_key = _query_features_cache_key(norm_path, crop_rect, fast_only)
        cached_feat = _QUERY_FEAT_CACHE.get(cache_key)
        if cached_feat is not None:
            global _QUERY_FEAT_HITS
            _QUERY_FEAT_HITS += 1
            if lat_on:
                lat.set("query_embedding_source", "query_cache")
            return cached_feat
        global _QUERY_FEAT_MISSES
        _QUERY_FEAT_MISSES += 1

        def _remember(payload: tuple[Any, str, str]) -> tuple[Any, str, str]:
            if len(_QUERY_FEAT_CACHE) >= _QUERY_FEAT_MAX:
                oldest = next(iter(_QUERY_FEAT_CACHE))
                _QUERY_FEAT_CACHE.pop(oldest, None)
            _QUERY_FEAT_CACHE[cache_key] = payload
            return payload

        if not crop_rect:
            rec = self.db.get_file_by_path(norm_path)
            t_stat = time.perf_counter() if lat_on else 0.0
            stat = os.stat(norm_path) if os.path.isfile(norm_path) else None
            if lat_on:
                lat.add("query_stat", (time.perf_counter() - t_stat) * 1000.0)
                lat.set("query_stat_hit", bool(stat))
            record_is_current = bool(
                rec
                and stat
                and int(rec.get("file_size", 0) or 0) == int(stat.st_size)
                and abs(float(rec.get("mtime", 0) or 0) - float(stat.st_mtime)) < 1.0
            )
            if record_is_current:
                ready = self.db.get_indexed_files_by_ids(
                    [int(rec["id"])],
                    include_processing_ready=True,
                    include_inactive_sources=True,
                    lightweight=bool(fast_only),
                )
                if ready:
                    cached = ready[0]
                    features = ExtractedFeatures(
                        phash=cached.get("phash", "") or "",
                        dhash=cached.get("dhash", "") or "",
                        whash=cached.get("whash", "") or "",
                        color_hist=(b"" if fast_only else (cached.get("color_hist") or b"")),
                        dominant_colors=([] if fast_only else (cached.get("dominant_colors") or [])),
                        texture_features=([] if fast_only else (cached.get("texture_features") or [])),
                        dino_embedding=(b"" if fast_only else (cached.get("dino_embedding") or b"")),
                        clip_embedding=(b"" if fast_only else (cached.get("clip_embedding") or b"")),
                        patch_embeddings_meta=([] if fast_only else (cached.get("patch_embeddings_meta") or [])),
                        texture_map=cached.get("texture_map") or {},
                    )
                    partial = rec.get("partial_hash", "") or partial_file_hash(
                        norm_path
                    )
                    if lat_on:
                        lat.set("query_embedding_source", "db_cache")
                        lat.set("query_nas_reread", classify_storage(norm_path) == "nas")
                    return _remember((features, norm_path, partial))
        feature_path = self._feature_image_path(norm_path)
        from core.index_freeze import INDEX_FROZEN

        if INDEX_FROZEN:
            thumb_path = feature_path
        else:
            with lat.span("query_thumbnail"):
                thumb = self._thumbnailer.create(image_path)
            thumb_path = thumb.thumbnail_path if thumb.success else feature_path
        _ = thumb_path

        if crop_rect:
            import numpy as np
            from PIL import Image

            x, y, w, h = crop_rect
            with Image.open(image_path) as img:
                img = Thumbnailer.normalize_pillow_image(img)
                x = max(0, min(int(x), img.width - 1))
                y = max(0, min(int(y), img.height - 1))
                w = max(1, min(int(w), img.width - x))
                h = max(1, min(int(h), img.height - y))
                cropped = img.crop((x, y, x + w, y + h))
                arr = np.array(cropped.convert("RGB"))
            ext = (
                FeatureExtractor(use_ai=False, use_gpu=False, fast_hash_only=True)
                if fast_only
                else self.extractor
            )
            features = ext.extract_from_array(
                arr,
                include_patches=not fast_only,
                filename=Path(norm_path).name,
                path=norm_path,
            )
            if fast_only:
                if lat_on:
                    lat.set("query_embedding_source", "extract_crop")
                return _remember((features, norm_path, ""))
            partial = partial_file_hash(norm_path) if os.path.isfile(norm_path) else ""
            if lat_on:
                lat.set("query_embedding_source", "extract_crop")
            return _remember((features, norm_path, partial))

        ext = (
            FeatureExtractor(use_ai=False, use_gpu=False, fast_hash_only=True)
            if fast_only
            else self.extractor
        )
        with lat.span("query_embed_extract"):
            features = ext.extract_from_path(feature_path, source_path=norm_path)
        partial = partial_file_hash(norm_path) if os.path.isfile(norm_path) else ""
        if lat_on:
            lat.set("query_embedding_source", "extract")
            lat.set("query_nas_reread", classify_storage(norm_path) == "nas")
        return _remember((features, norm_path, partial))

    def execute_search(
        self,
        query: SearchQuery,
        *,
        result_callback: Callable[[list[SearchResult], int, int], None] | None = None,
    ) -> SearchResponse:
        """Ana arama girişi — tüm sonuçları ve istatistikleri döndürür."""
        from core.index_freeze import search_session

        with search_session():
            return self._execute_search_inner(
                query, result_callback=result_callback
            )

    def _execute_search_inner(
        self,
        query: SearchQuery,
        *,
        result_callback: Callable[[list[SearchResult], int, int], None] | None = None,
    ) -> SearchResponse:
        """Ana arama girişi — tüm sonuçları ve istatistikleri döndürür."""
        t0 = time.perf_counter()
        lat = self._latency
        if getattr(lat, "enabled", False):
            lat.reset()
            lat.mark_start()
        self._last_retrieve_meta = {}
        threshold = (
            query.threshold
            if query.threshold is not None
            else self.settings.similarity_threshold
        )
        if self.settings.search_mode == "comprehensive":
            threshold = min(threshold, self.settings.comprehensive_threshold)

        customer = query.customer or self.settings.customer_filter

        if query.mode == "text" and not query.image_path:
            try:
                from core.search_memory import apply_query_memory

                rewritten, mem_meta = apply_query_memory(
                    self.settings.db_path, query.text, customer=customer
                )
                if rewritten:
                    query.text = rewritten
                self._last_memory_meta = dict(mem_meta or {})
            except Exception:
                self._last_memory_meta = {}
            threshold = (
                query.threshold
                if query.threshold is not None
                else self.settings.similarity_threshold
            )
            # NL 2.0 bridge (submit-time): customer scope + concept search_text
            text_for_search = query.text
            try:
                reg = self._customer_registry_cached()
                from core.search_intelligence_chain import analyze_query_intelligence

                nl_analysis = analyze_query_intelligence(
                    query.text, customer_registry=reg
                )
                self._last_nl_analysis = dict(nl_analysis or {})
                if not (customer or "").strip() and nl_analysis.get(
                    "customer_high_confidence"
                ):
                    customer = str(nl_analysis.get("customer") or "").strip()
                st = str(nl_analysis.get("search_text") or "").strip()
                if nl_analysis.get("customer_high_confidence") and st:
                    text_for_search = st
            except Exception:
                self._last_nl_analysis = {}
            all_results = self._search_text_internal(
                text_for_search, customer, threshold, result_callback=result_callback
            )
            cat_path = (query.category_path_filter or "").strip()
            if cat_path:
                allowed_ids = set(self.db.list_file_ids_by_category_path(cat_path))
                all_results = (
                    [r for r in all_results if int(r.file_id) in allowed_ids]
                    if allowed_ids
                    else []
                )
            self._mark_threshold_debug(all_results, threshold)
            stats = self._build_stats(all_results, threshold, query, t0)
            filtered = [
                r for r in all_results
                if self._passes_search_threshold(r, threshold)
            ]
            # Sıralama zaten search_by_text'te yapıldı (Name Search v2 tier)
            from core.textile_terms import expand_query_terms, fuzzy_correct_query, query_family_hints

            semantic_enabled = bool(
                getattr(self.settings, "semantic_text_search_enabled", False)
            )
            _corrected, _was_corrected = fuzzy_correct_query(query.text)
            effective = _corrected if _was_corrected else query.text
            hints = query_family_hints(effective, include_semantic=semantic_enabled)
            method = "metin"
            if hints.get("pattern_family"):
                method = "metin + doku etiketi"
            if _was_corrected:
                method += " (yazım düzeltildi)"
            response = SearchResponse(
                all_results=all_results,
                results=filtered,
                stats=stats,
                meta={
                    "mode": "text",
                    "text": query.text,
                    "text_query": query.text,
                    "spell_corrected": _corrected if _was_corrected else "",
                    "spell_was_corrected": _was_corrected,
                    "threshold": threshold,
                    "search_method": method,
                    "query_family_hints": hints,
                    "query_pattern_family": hints.get("pattern_family", ""),
                    "query_animal_print_type": hints.get("animal_print_type", ""),
                    "query_pattern_subtype": hints.get("pattern_type", ""),
                    "selected_sources": list(self.settings.selected_source_ids or []),
                    "search_scope": self.settings.search_scope,
                    "category_path_filter": cat_path,
                    "semantic_text_search_enabled": semantic_enabled,
                    "semantic_query_terms": expand_query_terms(
                        query.text, include_semantic=semantic_enabled,
                    ) if semantic_enabled else [],
                    "ai_faiss_used": bool(
                        (getattr(self, "_last_text_intel_meta", {}) or {}).get(
                            "ai_faiss_used"
                        )
                    ),
                    **(getattr(self, "_last_text_intel_meta", {}) or {}),
                    **(getattr(self, "_last_memory_meta", {}) or {}),
                },
            )
            try:
                from core.search_cache import search_cache_key
                from core.search_memory import get_search_memory, remember_query_rewrite

                get_search_memory(self.settings.db_path).put(
                    search_cache_key(self.settings, query), response
                )
                if _was_corrected and _corrected:
                    remember_query_rewrite(
                        self.settings.db_path, query.text, _corrected, kind="typo"
                    )
            except Exception:
                pass
            return response

        if not query.image_path:
            return SearchResponse(meta={"error": "no_image"})

        with lat.span("query_features"):
            features, query_path, query_partial = self._query_features_from_image(
                query.image_path,
                crop_rect=query.crop_rect if query.use_crop else None,
                fast_only=query.fast_only,
            )
        # Global Object AI: sorgu görüntüsündeki nesneleri ve ayrı insan örneklerini
        # çıkarır. Mevcut FAISS/DINO/Pattern DNA skorlarına müdahale etmez.
        # Model/dependency yoksa mevcut arama aynen devam eder.
        global_object_meta: dict[str, Any] = {}
        if bool(getattr(self.settings, "global_object_intelligence_enabled", True)):
            try:
                from core.global_object_intelligence import GlobalObjectIntelligence

                object_ai = GlobalObjectIntelligence(
                    enabled=True,
                    min_confidence=float(getattr(self.settings, "global_object_detection_min_confidence", 0.45) or 0.45),
                    max_detections=int(getattr(self.settings, "global_object_max_detections", 50) or 50),
                )
                with lat.span("global_object_ai"):
                    global_object_meta = object_ai.analyze(query_path or query.image_path)
            except Exception as exc:
                logger.warning("Global Object AI atlandı; mevcut arama devam ediyor: %s", exc)
                global_object_meta = {
                    "enabled": False, "objects": [], "counts": {}, "person_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        with lat.span("indexed_pool"):
            indexed_pool = self._indexed_files(customer, lightweight=True)
        cat_path = (query.category_path_filter or "").strip()
        if cat_path:
            allowed_ids = set(self.db.list_file_ids_by_category_path(cat_path))
            indexed_pool = (
                [r for r in indexed_pool if int(r["id"]) in allowed_ids]
                if allowed_ids
                else []
            )
        with lat.span("identity_protected"):
            query_profile = TextureProfile.from_dict(features.texture_map or {})
            feedback_adj = self._feedback.score_adjustments(query_path or query.image_path)
            wrong_ids = self._feedback.wrong_result_ids(
                query_path or query.image_path or ""
            )
        # Face Intelligence: visual query'de aynı kişiyi taşıyan dosyaları
        # pattern prefilter'dan düşürme. Bu yalnızca yüz kanıtı varsa devreye
        # girer; normal desen araması ve mevcut sıralama korunur.
        face_matches: dict[int, dict[str, Any]] = {}
        if bool(getattr(self.settings, "face_index_enabled", False)) and query.image_path:
            try:
                from core.face_search import FaceSearch

                # execute_search image path has no local `limit` — use query/settings.
                face_limit = int(
                    getattr(query, "limit", 0)
                    or getattr(self.settings, "search_result_limit", 0)
                    or 400
                )
                face_matches = FaceSearch(
                    getattr(self.settings, "face_db_path", ""),
                    threshold=float(getattr(self.settings, "face_identity_threshold", 0.62) or 0.62),
                    min_margin=float(getattr(self.settings, "face_identity_min_margin", 0.05) or 0.05),
                ).image_matches(
                    query.image_path, limit=max(2000, int(face_limit or 400))
                )
            except Exception as exc:
                logger.debug("Face visual retrieval atlandı: %s", exc)
                face_matches = {}
        if face_matches:
            protected_ids.update(int(fid) for fid in face_matches)
        pattern_group_ids: set[int] = set()
        query_record = self.db.get_file_by_path(query_path) if query_path else None
        with lat.span("identity_protected"):
            if query.fast_only:
                # Skip heavy identity blob fetch on progressive path.
                identity_candidates = []
                if query_record:
                    group = self.db.get_pattern_group_for_file(int(query_record["id"]))
                    if group:
                        pattern_group_ids = {
                            int(member["file_id"])
                            for member in self.db.list_pattern_group_members(
                                int(group["id"])
                            )
                        }
            else:
                identity_candidates = [
                    rec
                    for rec in self.db.get_identity_candidates(
                        path=query_path or query.image_path or "",
                        partial_hash=query_partial,
                        full_hash=query_record.get("full_hash", "") if query_record else "",
                    )
                    if self._record_in_scope(rec, customer)
                ]
                if query_record:
                    group = self.db.get_pattern_group_for_file(int(query_record["id"]))
                    if group:
                        pattern_group_ids = {
                            int(member["file_id"])
                            for member in self.db.list_pattern_group_members(
                                int(group["id"])
                            )
                        }
            if query.fast_only:
                protected_ids = set(pattern_group_ids)
                if query_record:
                    protected_ids.add(int(query_record["id"]))
            else:
                protected_ids = collect_protected_exact_ids(
                    indexed_pool,
                    query_phash=features.phash or "",
                    query_dhash=features.dhash or "",
                    query_whash=features.whash or "",
                    query_partial_hash=query_partial,
                    query_full_hash=query_record.get("full_hash", "") if query_record else "",
                    query_path=query_path or query.image_path or "",
                    query_file_id=int(query_record["id"]) if query_record else None,
                    pattern_group_ids=pattern_group_ids,
                )
                # IMPORTANT: keep Face Intelligence candidates protected.
                # The previous code overwrote the set here, silently dropping
                # face matches before the pattern prefilter could see them.
                protected_ids.update(int(fid) for fid in face_matches)
                protected_ids.update(int(rec["id"]) for rec in identity_candidates)
        with lat.span("prefilter"):
            narrowed, pre_stats = self._prefilter.narrow_candidates(
                indexed_pool,
                features,
                query_profile,
                self.faiss,
                text_query=query.text,
                query_partial_hash=query_partial,
                feedback_candidate_ids=set(feedback_adj),
                pattern_group_candidate_ids=pattern_group_ids,
                protected_exact_ids=protected_ids,
            )
        with lat.span("hydrate_candidates"):
            if query.fast_only:
                detailed_candidates = list(narrowed)
            else:
                detailed_candidates = self.db.get_indexed_files_by_ids(
                    [int(rec["id"]) for rec in narrowed],
                    include_processing_ready=True,
                    lightweight=False,
                )
            detailed_candidates = merge_candidate_records(
                detailed_candidates,
                indexed_pool,
                protected_ids,
            )
        with lat.span("hash_verify"):
            # Toplu hash decode TTFR'yi 10s+ kilitliyordu. Deep patch kümesinde yapılır.
            present_ids = {int(rec["id"]) for rec in detailed_candidates}
            if query.fast_only:
                detailed_candidates.extend(
                    rec
                    for rec in identity_candidates
                    if int(rec["id"]) not in present_ids
                )
            else:
                detailed_candidates.extend(
                    rec
                    for rec in identity_candidates
                    if int(rec["id"]) not in present_ids
                )
        with lat.span("family_overrides"):
            if query.fast_only:
                # Feedback overrides already rare; skip batch SQLite on hot path.
                pass
            else:
                detailed_candidates = self._records_with_family_overrides(
                    detailed_candidates
                )
            # Sorgu özelliklerini de doğrula (bozuk DB hash'i yanlış eşleşme üretmesin)
            if query_path and not query.fast_only:
                qrec = self.db.get_file_by_path(query_path)
                if qrec:
                    vq = verified_hashes_for_record(
                        {
                            **qrec,
                            **{
                                "phash": features.phash,
                                "dhash": features.dhash,
                                "whash": features.whash,
                            },
                        }
                    )
                    if vq.phash:
                        features.phash = vq.phash
                        features.dhash = vq.dhash
                        features.whash = vq.whash
        if getattr(lat, "enabled", False):
            lat.set("candidates_scored", len(detailed_candidates))
            lat.set("query_storage", classify_storage(query_path or query.image_path))
        with lat.span("scoring"):
            all_scored = self._search_with_features(
                features,
                customer=customer,
                query_path=query_path,
                query_partial_hash=query_partial,
                text_query=query.text,
                crop_search=query.use_crop,
                indexed=detailed_candidates,
                query_profile=query_profile,
                feedback_adjustments=feedback_adj,
                pattern_group_ids=pattern_group_ids,
                wrong_result_ids=wrong_ids,
                result_callback=result_callback,
                stream_threshold=threshold,
            )
            # Aynı kişi kanıtı, görsel aramada güçlü ama izole bir ek kanıttır.
            # Pattern skorlarını silmez; yalnızca aynı kişiyi taşıyan sonuçların
            # mevcut görsel benzerliklerinden bağımsız olarak görünür kalmasını
            # ve üst sıralara çıkmasını sağlar.
            if face_matches:
                # Face candidates are authoritative for the identity layer.
                # Do not rely on the textile prefilter to keep them: a person's
                # photo can be visually unrelated to the query's textile family.
                scored_by_id = {int(r.file_id): r for r in all_scored}
                for fid, fm in face_matches.items():
                    result = scored_by_id.get(int(fid))
                    if result is None:
                        rec = self.db.get_file_by_id(int(fid))
                        if rec:
                            sim0 = float(fm.get("similarity", 0.0) or 0.0)
                            result = self._to_result(rec, min(0.99, 0.78 + 0.20 * max(0.0, min(1.0, sim0))), {"face": sim0})
                            all_scored.append(result)
                            scored_by_id[int(fid)] = result
                    if result is None:
                        continue
                    sim = float(fm.get("similarity", 0.0) or 0.0)
                    face_score = min(0.99, 0.78 + 0.20 * max(0.0, min(1.0, sim)))
                    result.debug = dict(result.debug or {})
                    result.debug["face_match"] = True
                    result.debug["result_layer"] = "same_person"
                    result.debug["face_similarity"] = round(sim, 5)
                    result.debug["face_person_id"] = str(fm.get("person_id", ""))
                    result.debug["face_score"] = round(face_score, 5)
                    result.debug["face_evidence"] = "Aynı kişi"
                    result.score = max(float(result.score), face_score)
                    result.score_percent = round(result.score * 100, 1)

            if not query.fast_only and bool(
                getattr(self.settings, "adaptive_retrieve_enabled", True)
            ):
                all_scored = self._adaptive_expand_scored(
                    all_scored,
                    features=features,
                    indexed_pool=indexed_pool,
                    customer=customer,
                    query_path=query_path,
                    query_partial_hash=query_partial,
                    text_query=query.text,
                    crop_search=query.use_crop,
                    query_profile=query_profile,
                    feedback_adj=feedback_adj,
                    pattern_group_ids=pattern_group_ids,
                    wrong_ids=wrong_ids,
                    result_callback=result_callback,
                    threshold=threshold,
                    protected_ids=protected_ids,
                )
        with lat.span("sort"):
            text_family_conflict = False
            if query.text:
                all_scored = self._apply_hybrid_text_boost(all_scored, query.text)
                try:
                    from core.intent_evidence_routing import apply_intent_evidence_routing

                    all_scored = apply_intent_evidence_routing(
                        all_scored,
                        query.text,
                        has_image=bool(query.image_path),
                        db_path=str(getattr(self.settings, "db_path", "") or ""),
                    )
                except Exception as exc:
                    logger.debug("intent evidence routing skipped: %s", exc)
                try:
                    from core.natural_language_query import parse_natural_query
                    parsed_query = parse_natural_query(query.text)
                    from core.brand_aliases import resolve_brand_alias, query_brand_needles
                    dynamic_brand = resolve_brand_alias(query.text, self.db.db_path) or ""
                    brand_needles = query_brand_needles(query.text, self.db.db_path)
                    all_scored = self._apply_structured_query_ranking(
                        all_scored,
                        parsed_query,
                        query.text,
                        dynamic_brand=dynamic_brand,
                        brand_needles=brand_needles,
                    )
                except Exception as exc:
                    logger.warning("Yapılandırılmış sorgu sıralaması atlandı: %s", exc)
                text_family_conflict = any(
                    bool(result.debug.get("text_family_conflict")) for result in all_scored
                )
                all_scored.sort(key=self._engine_sort_key)
            elif query.use_crop:
                # Area Select: no fake text — route by crop visual profile evidence
                # through the same soft layer as pattern text search.
                try:
                    from core.intent_evidence_routing import (
                        apply_intent_evidence_routing,
                        plan_from_visual_profile,
                    )

                    crop_plan = plan_from_visual_profile(query_profile)
                    if crop_plan.wanted:
                        all_scored = apply_intent_evidence_routing(
                            all_scored,
                            "",
                            has_image=True,
                            plan=crop_plan,
                        )
                        all_scored.sort(key=self._engine_sort_key)
                except Exception as exc:
                    logger.debug("crop visual evidence routing skipped: %s", exc)

            all_scored.sort(key=self._engine_sort_key)

        textile_rerank_applied = False
        if (
            self.settings.fine_detail_enabled
            and query.mode != "text"
            and not query.fast_only
        ):
            from core.textile_reranker import textile_rerank_v2

            q_rec = {
                "texture_map": dict(query_profile.texture_map or {}),
                "pattern_family": query_profile.pattern_family or "",
                "organic_blob_score": float(query_profile.organic_blob_score or 0),
                "stripe_score": float(query_profile.stripe_score or 0),
                "scale_pattern_score": float(query_profile.scale_pattern_score or 0),
                "edge_density": float(query_profile.edge_density or 0),
                "contrast_score": float(
                    (query_profile.texture_map or {}).get("contrast_score", 0) or 0
                ),
                "repeat_density": float(query_profile.repeat_density or 0),
                "animal_score": float(
                    (query_profile.texture_map or {}).get("animal_score", 0) or 0
                ),
                "floral_score": float(
                    (query_profile.texture_map or {}).get("floral_score", 0) or 0
                ),
            }
            # FAISS/hash sonrası pool → tekstil rerank (Spot Geometry + aile cezası)
            pool_k = max(300, int(self.settings.fine_detail_top_k or 80) * 3)
            with lat.span("textile_rerank"):
                all_scored = textile_rerank_v2(
                    q_rec,
                    all_scored,
                    pool_k=pool_k,
                )
            textile_rerank_applied = True
            with lat.span("sort"):
                all_scored.sort(
                key=lambda r: (
                    (
                        0
                        if r.is_self_match
                        else 1 if r.debug.get("protected_exact") else 2
                    ),
                    hierarchy_sort_key(
                        r.cluster_group or CLUSTER_UNRELATED,
                        r.hierarchy_score,
                        r.score,
                    ),
                ),
            )

        # Face Intelligence is a first-class result layer. Textile reranking and
        # family sorting must never push a same-person match below unrelated
        # pattern results.
        if face_matches:
            all_scored.sort(key=self._engine_sort_key)

        with lat.span("display_payload"):
            self._mark_threshold_debug(all_scored, threshold)
            filtered = [r for r in all_scored if r.score >= threshold]
            stats = self._build_stats(
                all_scored, threshold, query, t0, query_path, indexed_pool, pre_stats
            )
            below = sorted(
                [r for r in all_scored if r.score < threshold],
                key=lambda r: r.score,
                reverse=True,
            )[:50]
        with lat.span("family_display"):
            from core.search_display import (
                filter_ranked_visual_results,
                is_ranked_visual_search,
                visual_search_floor,
            )
            from core.search_reject_report import build_family_reject_report
            from core.dynamic_groups import normalize_cluster_key

            ranked_visual = is_ranked_visual_search(
                {"mode": query.mode, "text": query.text},
                has_text=bool((query.text or "").strip()),
            )
            display_floor = (
                visual_search_floor(self.settings) if ranked_visual else threshold
            )
            hidden = None
            # Görsel aramada "Alakasız sonuçları göster" kapalıysa,
            # moddan bağımsız olarak uzak/alakasız kümeleri gerçekten gizle.
            # Önceden bu kontrol yalnızca "comprehensive" modda çalışıyordu;
            # style/similar modlarında family gate tarafından "unrelated"
            # işaretlenen sonuçlar tekrar UI'ye sızıyordu.
            if ranked_visual and not self.settings.show_unrelated_results:
                from core.dynamic_groups import DEFAULT_HIDDEN_GROUPS

                hidden = DEFAULT_HIDDEN_GROUPS
            display_shown = (
                filter_ranked_visual_results(
                    all_scored,
                    display_floor,
                    hidden_clusters=hidden,
                    normalize_cluster_key=normalize_cluster_key,
                    query_family=str(query_profile.pattern_family or ""),
                    query_animal=str(query_profile.animal_print_type or ""),
                )
                if ranked_visual
                else filtered
            )
            family_reject_report = build_family_reject_report(
                all_scored,
                display_shown,
                query_family=str(query_profile.pattern_family or ""),
                query_animal=str(query_profile.animal_print_type or ""),
                floor=display_floor,
                threshold=threshold,
                hidden_clusters=hidden,
            )
        if getattr(lat, "enabled", False):
            family_n = sum(
                1
                for r in all_scored
                if getattr(r, "same_pattern_family", False)
                or str(getattr(r, "cluster_group", "") or "")
                in ("same_family_close", "same_family_style", "related_family")
            )
            lat.set("family_count", family_n)
            lat.set("results_total", len(all_scored))
            lat.set("results_shown", len(display_shown))
            lat.set("uvi_in_image_path", False)
        pipeline_audit = self._build_pipeline_audit(
            query=query,
            features=features,
            indexed_pool=indexed_pool,
            pre_stats=pre_stats,
            all_scored=all_scored,
            filtered=filtered,
            display_shown=display_shown,
            textile_rerank_applied=textile_rerank_applied,
            t0=t0,
        )
        self._last_pipeline_audit = pipeline_audit
        for rec in all_scored:
            dbg = dict(rec.debug or {})
            dbg["search_pipeline"] = pipeline_audit
            rec.debug = dbg
        # UI'nin kullanması gereken liste, yalnızca skor eşiğinden geçenler
        # değil; görsel arama için family/cluster görünürlük kapısından da
        # geçen sonuçlardır. Önceden `filtered` döndürülüyordu ve yukarıda
        # hesaplanan `display_shown` hiç kullanılmadığı için "unrelated/far"
        # sonuçlar ekranda görünmeye devam ediyordu.
        visible_results = display_shown if ranked_visual else filtered
        stats.displayed = (
            len(visible_results)
            if int(self.settings.search_result_limit or 0) <= 0
            else min(len(visible_results), int(self.settings.search_result_limit))
        )
        stats.remaining = max(
            0, len(visible_results) - int(self.settings.search_result_limit or 0)
        ) if int(self.settings.search_result_limit or 0) > 0 else 0

        # Progressive aramada ara sonuçlar zaten yukarıdan akar; burada bir de
        # son skor/rerank tamamlandıktan sonra nihai sıralamayı yayınla. Böylece
        # yüksek puanlı yeni kayıtlar geldikçe kartlar yukarı taşınır ve son
        # liste ile ara liste arasında kalıcı sıra farkı kalmaz.
        if result_callback and visible_results:
            final_limit = int(self.settings.search_result_limit or 0)
            final_snapshot = (
                visible_results[:final_limit]
                if final_limit > 0
                else list(visible_results)
            )
            result_callback(final_snapshot, len(visible_results), len(indexed_pool))

        return SearchResponse(
            all_results=all_scored,
            results=visible_results,
            stats=stats,
            below_threshold_preview=below,
            meta={
                "mode": query.mode,
                "used_crop": query.use_crop,
                "text": query.text,
                "threshold": threshold,
                "selected_sources": list(self.settings.selected_source_ids or []),
                "search_scope": self.settings.search_scope,
                "scope_fallback_used": bool(self._last_scope_fallback),
                "scope_fallback_from": self._last_scope_fallback,
                "indexed_pool_size": len(indexed_pool),
                "search_mode": self.settings.search_mode,
                "color_weight_mode": self.settings.color_weight_mode,
                "query_pattern_family": query_profile.pattern_family,
                "query_animal_print_type": query_profile.animal_print_type,
                "query_pattern_subtype": query_profile.pattern_subtype,
                "query_color_family": query_profile.color_family,
                "query_confidence": query_profile.classification_confidence,
                "query_effective_family": effective_query_family(query_profile),
                "protected_exact_candidates_count": len(protected_ids),
                "query_texture_summary": {
                    "blob": query_profile.organic_blob_score,
                    "stripe": query_profile.stripe_score,
                    "scale": query_profile.scale_pattern_score,
                },
                "prefilter": pre_stats.__dict__,
                "faiss_dino_count": self.faiss.dino_count,
                "faiss_clip_count": self.faiss.clip_count,
                "ai_embedding_enabled": self.settings.ai_embedding_enabled,
                "fast_hash_only": bool(query.fast_only or self.settings.fast_hash_only),
                "patch_used": bool(
                    not query.fast_only and features.patch_embeddings_meta
                ),
                "texture_map_used": bool(not query.fast_only and features.texture_map),
                "ai_used": bool(
                    self.settings.ai_embedding_enabled
                    and (features.dino_embedding or features.clip_embedding)
                ),
                "user_feedback_applied": bool(feedback_adj),
                "text_family_conflict": text_family_conflict,
                "category_path_filter": cat_path,
                "global_object_intelligence": global_object_meta,
                "family_reject_report": family_reject_report,
                "pipeline_audit": pipeline_audit,
                "ai_load_error": getattr(self, "_ai_load_error", "") or "",
                "latency": (
                    self._latency.snapshot()
                    if getattr(self._latency, "enabled", False)
                    else {}
                ),
                **(getattr(self, "_last_retrieve_meta", None) or {}),
            },
        )

    def _build_pipeline_audit(
        self,
        *,
        query,
        features,
        indexed_pool,
        pre_stats,
        all_scored,
        filtered,
        display_shown,
        textile_rerank_applied: bool,
        t0: float,
    ) -> dict[str, Any]:
        import time as _time

        from core.ai_pipeline_audit import build_search_pipeline_audit

        ai_used = bool(
            self.settings.ai_embedding_enabled
            and (features.dino_embedding or features.clip_embedding)
        )
        faiss_hits = int(getattr(pre_stats, "faiss_hits", 0) or 0)
        if hasattr(pre_stats, "__dict__"):
            faiss_hits = int(
                pre_stats.__dict__.get("faiss_hits")
                or pre_stats.__dict__.get("faiss_candidate_count")
                or faiss_hits
                or 0
            )
        dna_ok = any(
            float((r.debug or {}).get("dna_score") or 0) > 0 for r in (all_scored or [])[:20]
        )
        kb_ok = any(
            float((r.debug or {}).get("knowledge_score") or 0) > 0
            or (r.debug or {}).get("knowledge_explanation")
            for r in (all_scored or [])[:20]
        )
        rerank_ok = textile_rerank_applied or any(
            (r.debug or {}).get("textile_rerank_v2") for r in (all_scored or [])[:5]
        )
        provider = []
        if getattr(self.extractor, "_clip_model", None) is not None:
            provider.append("OpenCLIP")
        if getattr(self.extractor, "_dino_model", None) is not None:
            provider.append("DINOv2")
        dim = 0
        if features.dino_embedding:
            dim = max(dim, len(features.dino_embedding) // 4)
        if features.clip_embedding:
            dim = max(dim, len(features.clip_embedding) // 4)
        audit = build_search_pipeline_audit(
            ai_active=ai_used,
            ai_reason="" if ai_used else (getattr(self, "_ai_load_error", "") or "AI embedding boş"),
            indexed_total=len(indexed_pool or []),
            faiss_hits=faiss_hits or self.faiss.dino_count,
            candidates=len(all_scored or []),
            after_rerank=len(all_scored or []),
            after_gate=len(display_shown or filtered or []),
            final=len(filtered or []),
            embedding_ok=ai_used or bool(query.fast_only),
            dna_ok=dna_ok or bool(features.texture_map),
            knowledge_ok=kb_ok,
            rerank_ok=rerank_ok or bool(query.fast_only),
            gate_ok=True,
            ui_ok=True,
            provider=" + ".join(provider) or "hash+texture",
            device=str(getattr(self.extractor, "device", "cpu")),
            embedding_dim=dim,
            elapsed_sec=round(_time.perf_counter() - t0, 3),
        )
        return audit.to_dict()

    def _persist_repaired_hashes(self, rec: dict[str, Any]) -> None:
        """Bozuk perceptual hash'i DB'ye yaz — search altında yasak."""
        from core.index_freeze import guard_index_write

        guard_index_write("db.update_feature_hashes", "core.search_engine")
        fid = int(rec.get("id") or 0)
        if not fid:
            return
        try:
            self.db.update_feature_hashes(
                fid,
                phash=str(rec.get("phash") or ""),
                dhash=str(rec.get("dhash") or ""),
                whash=str(rec.get("whash") or ""),
            )
        except Exception as exc:
            logger.debug("Hash onarımı kaydedilemedi id=%s: %s", fid, exc)

    def _apply_family_overrides(self, result: SearchResult) -> None:
        """
        Apply per-file family overrides coming from user feedback.
        Uses data attached during candidate prep — no per-row SQLite.
        """
        ov = result.debug.get("feedback_family_override")
        if not isinstance(ov, dict):
            return
        positive = str(ov.get("positive") or "").strip()
        negative = ov.get("negative") or set()
        if positive:
            result.pattern_family = positive
            result.debug["family_override"] = {"positive": positive}
            return
        if isinstance(negative, set) and negative and result.pattern_family in negative:
            result.debug["family_override"] = {"blocked": result.pattern_family}
            result.pattern_family = "unknown"
            result.same_pattern_family = False

    def _records_with_family_overrides(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        overrides = self.db.get_feedback_family_overrides_for_files(
            [int(rec["id"]) for rec in records if rec.get("id")]
        )
        if not overrides:
            return records

        output: list[dict[str, Any]] = []
        for original in records:
            rec = dict(original)
            ov = overrides.get(int(rec.get("id") or 0))
            if not ov:
                output.append(rec)
                continue
            tm = dict(rec.get("texture_map") or {})
            current = rec.get("pattern_family") or tm.get("pattern_family") or "unknown"
            positive = str(ov.get("positive") or "").strip()
            negative = ov.get("negative") or set()
            changed = False
            if positive:
                rec["pattern_family"] = positive
                rec["pattern_confidence"] = 1.0
                tm["pattern_family"] = positive
                tm["classification_confidence"] = 1.0
                tm["user_labeled"] = True
                if positive != "animal_print":
                    rec["pattern_type"] = ""
                    rec["pattern_subtype"] = ""
                    tm["animal_print_type"] = ""
                    tm["pattern_subtype"] = ""
                changed = True
            elif isinstance(negative, set) and current in negative:
                rec["pattern_family"] = "unknown"
                rec["pattern_confidence"] = 1.0
                tm["pattern_family"] = "unknown"
                tm["classification_confidence"] = 1.0
                tm["user_labeled"] = True
                if current == "animal_print":
                    rec["pattern_type"] = ""
                    rec["pattern_subtype"] = ""
                    tm["animal_print_type"] = ""
                    tm["pattern_subtype"] = ""
                changed = True
            if changed:
                rec["texture_map"] = tm
                rec["text_search_blob"] = ""
                rec["feedback_family_override"] = ov
            output.append(rec)
        return output

    def _build_stats(
        self,
        all_results: list[SearchResult],
        threshold: float,
        query: SearchQuery,
        t0: float,
        query_path: str = "",
        indexed: list[dict[str, Any]] | None = None,
        prefilter_stats: PrefilterStats | None = None,
    ) -> SearchStats:
        # Metin aramada tüm indexi RAM'e çekme — sayım sorguları yeterli
        if indexed is None:
            status_counts = self.db.count_by_status() or {}
            total_indexed = int(status_counts.get("indexed", 0) or 0)
            try:
                src_summary = self.db.count_sources_summary()
                sources_n = int(src_summary.get("active", 0) or 1)
            except Exception:
                sources_n = 1
            selected_n = total_indexed
            supported_n = total_indexed
            thumbnail_count = 0
            feature_count = 0
            prefilter_cand = (
                prefilter_stats.final_candidate_count
                if prefilter_stats
                else len(all_results)
            )
        else:
            total_indexed = int(
                (self.db.count_by_status() or {}).get("indexed", 0) or 0
            ) or len(indexed)
            sources = {
                rec.get("source_id")
                for rec in indexed
                if rec.get("source_id")
            }
            sources_n = max(1, len(sources))
            selected_n = len(indexed)
            supported = [
                rec
                for rec in indexed
                if Path(rec.get("path", "")).suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            supported_n = len(supported)
            thumbnail_count = sum(1 for rec in supported if rec.get("thumbnail_path"))
            feature_count = sum(
                1
                for rec in supported
                if rec.get("phash") or rec.get("dhash") or rec.get("patch_embeddings_meta")
            )
            prefilter_cand = (
                prefilter_stats.final_candidate_count
                if prefilter_stats
                else len(indexed)
            )
        above = sum(
            1 for r in all_results
            if self._passes_search_threshold(r, threshold)
        )
        return SearchStats(
            total_indexed=total_indexed,
            sources_searched=sources_n,
            selected_source_files=selected_n,
            supported_image_files=supported_n,
            thumbnail_files=thumbnail_count,
            feature_files=feature_count,
            candidates_evaluated=len(all_results),
            above_threshold=above,
            displayed=(
                above
                if int(self.settings.search_result_limit or 0) <= 0
                else min(above, self.settings.search_result_limit)
            ),
            remaining=(
                0
                if int(self.settings.search_result_limit or 0) <= 0
                else max(0, above - self.settings.search_result_limit)
            ),
            near_below_threshold=sum(
                1 for r in all_results if r.score < threshold and r.score >= 0.30
            ),
            search_ms=(time.perf_counter() - t0) * 1000,
            used_crop=query.use_crop,
            text_query=query.text,
            query_path=query_path or query.image_path,
            prefilter_candidates=prefilter_cand,
            prefilter_used=bool(prefilter_stats and prefilter_stats.used_prefilter),
        )

    @staticmethod
    def _passes_search_threshold(result: SearchResult, threshold: float) -> bool:
        """Return whether a result is displayable at the current search threshold.

        Human/gender semantic retrieval intentionally uses a lower CLIP evidence
        floor (currently 0.18) than the generic textile search threshold
        (typically 0.60).  Before this guard, UVI correctly retrieved women/men
        but execute_search filtered them out again because their raw CLIP score
        was below the global threshold.  This made the UI show a populated
        progressive cache and then finish with 0 results.

        This does not falsify the similarity score: the raw score remains on the
        result.  It only gives the explicitly accepted human-semantic route its
        own acceptance rule.
        """
        dbg = getattr(result, "debug", {}) or {}
        # Human/gender retrieval is deliberately a separate acceptance domain.
        # IMPORTANT: do not use the generic textile threshold for these results.
        # `human_semantic_score` is the canonical marker written by the human
        # ranking stage; older flags remain accepted for backward compatibility.
        if dbg.get("face_gender_match"):
            return True
        # Exact detector evidence from object_index (not parent/hypernym).
        # CLIP-below-textile-threshold must not drop a verified bird/car/cat box.
        if dbg.get("object_index_hit") and not dbg.get("object_index_parent_hit"):
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
        if str(dbg.get("evidence_type") or "") == "ZERO_SHOT_VISUAL" or dbg.get("zero_shot_visual"):
            from core.search_evidence_gate import ZERO_SHOT_FLOOR, zero_shot_score

            return zero_shot_score(result) >= float(ZERO_SHOT_FLOOR)
        if dbg.get("open_vocab_object") and not dbg.get("ovd_rejected"):
            from core.open_vocab_object import OVD_MIN_CONFIDENCE

            return float(dbg.get("ovd_confidence") or 0.0) >= float(OVD_MIN_CONFIDENCE)
        if dbg.get("user_taught_positive"):
            return True
        if dbg.get("learned_concept_exact") or dbg.get("learned_concept"):
            return True
        if str((dbg.get("semantic_intent") or {}).get("motif") or "") == "leaf":
            try:
                clip_s = float(dbg.get("clip_score") or 0.0)
            except (TypeError, ValueError):
                clip_s = 0.0
            try:
                clip_s = max(clip_s, float((getattr(result, "breakdown", {}) or {}).get("clip") or 0.0))
            except (TypeError, ValueError):
                pass
            if clip_s >= 0.20:
                return True
        return float(result.score or 0.0) >= float(threshold)

    @staticmethod
    def _mark_threshold_debug(results: list[SearchResult], threshold: float) -> None:
        for result in results:
            if result.debug.get("protected_exact"):
                result.score = max(
                    result.score, threshold, 0.90 if result.is_self_match else 0.85
                )
                result.score_percent = round(result.score * 100, 1)
                result.debug["threshold_passed"] = True
                result.debug["reject_reason"] = ""
                continue
            passed = SearchEngine._passes_search_threshold(result, threshold)
            result.debug["threshold_passed"] = passed
            if passed:
                result.debug["reject_reason"] = (
                    "human_semantic_floor"
                    if result.debug.get("human_semantic_only") or result.debug.get("face_gender_match")
                    else (
                        "object_index_exact"
                        if result.debug.get("object_index_hit")
                        and not result.debug.get("object_index_parent_hit")
                        else ""
                    )
                )
                continue
            reasons = [f"threshold_below:{result.score:.4f}<{threshold:.4f}"]
            if not result.debug.get("has_thumbnail"):
                reasons.append("thumbnail_missing")
            if not result.debug.get("has_features"):
                reasons.append("feature_missing")
            result.debug["reject_reason"] = ";".join(reasons)

    def search_by_image(
        self,
        image_path: str,
        limit: int | None = None,
        threshold: float | None = None,
        customer: str = "",
    ) -> list[SearchResult]:
        limit = self.settings.search_result_limit if limit is None else int(limit)
        threshold = (
            threshold if threshold is not None else self.settings.similarity_threshold
        )
        q = SearchQuery(
            mode="image",
            image_path=image_path,
            threshold=threshold,
            customer=customer or self.settings.customer_filter,
        )
        resp = self.execute_search(q)
        return resp.results if limit <= 0 else resp.results[:limit]

    def search_in_folder_quick(
        self,
        image_path: str,
        folder_path: str,
        limit: int | None = None,
        threshold: float | None = None,
        index_first: bool = True,
    ) -> list[SearchResult]:
        """Klasörde hızlı ara — gerekirse önce hızlı index."""
        from core.indexer import Indexer

        folder = self._folder_as_directory(folder_path)
        if index_first:
            from core.index_freeze import search_write_protection_active

            # B only — INDEX_FROZEN alone must not skip pre-search folder index.
            if not search_write_protection_active():
                Indexer(self.settings).quick_index_folder(folder)

        prev_scope = self.settings.search_scope
        prev_folder = self.settings.quick_search_folder
        try:
            self.settings.search_scope = SearchScope.QUICK_FOLDER.value
            self.settings.quick_search_folder = folder
            return self.search_by_image(image_path, limit=limit, threshold=threshold)
        finally:
            self.settings.search_scope = prev_scope
            self.settings.quick_search_folder = prev_folder

    def _intel_enabled(self) -> bool:
        return bool(getattr(self.settings, "semantic_pattern_intel_enabled", True))

    def _cached_text_embedding(self, text: str) -> np.ndarray | None:
        key = str(text or "").strip().casefold()
        if not key:
            return None
        cached = self._text_embedding_cache.get(key)
        if cached is not None:
            return cached
        emb = self.extractor.embed_text(text)
        if emb is None:
            return None
        if len(self._text_embedding_cache) >= self._text_embedding_cache_max:
            # En eski eklenen anahtarı çıkar; küçük ve deterministik bir LRU-benzeri
            # havuz yeterli, üçüncü taraf cache gerektirmiyor.
            first = next(iter(self._text_embedding_cache), None)
            if first is not None:
                self._text_embedding_cache.pop(first, None)
        self._text_embedding_cache[key] = emb
        return emb

    def _clip_score_ids(self, text_emb: np.ndarray, file_ids: list[int]) -> dict[int, float]:
        """Cosine of one CLIP text vector vs already-indexed image vectors. Read-only."""
        index = getattr(self.faiss, "clip_index", None)
        id_map = getattr(self.faiss, "clip_id_map", None) or []
        if index is None or not file_ids:
            return {}
        n = len(id_map)
        pos = getattr(self, "_clip_id_pos", None)
        if pos is None or int(getattr(self, "_clip_id_pos_n", 0) or 0) != n:
            pos = {int(fid): i for i, fid in enumerate(id_map)}
            self._clip_id_pos = pos
            self._clip_id_pos_n = n
        q = np.asarray(text_emb, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(q) + 1e-8)
        q = q / denom
        out: dict[int, float] = {}
        for fid in file_ids:
            i = pos.get(int(fid))
            if i is None:
                continue
            try:
                vec = np.asarray(index.reconstruct(int(i)), dtype=np.float32).reshape(-1)
            except Exception:
                continue
            out[int(fid)] = float(np.dot(q, vec) / (float(np.linalg.norm(vec)) + 1e-8))
        return out

    def _zero_shot_clip_ok(self, spec: dict[str, str], clip_here: dict[str, float], rec: dict[str, Any]) -> bool:
        from types import SimpleNamespace

        from core.search_evidence_gate import print_blocks_zero_shot, score_zero_shot_gate

        concept = str((spec or {}).get("concept") or "")
        target = float((clip_here or {}).get("_similarity") or 0.0)
        rivals = {str(k): float(v or 0.0) for k, v in (clip_here or {}).items() if k != "_similarity"}
        tm = rec.get("texture_map") if isinstance(rec.get("texture_map"), dict) else {}
        row = SimpleNamespace(
            pattern_family=str(rec.get("pattern_family") or ""),
            debug={"texture_map": tm, "zs_clip_rivals": rivals},
        )
        if print_blocks_zero_shot(row, concept):
            return False
        ok, _ = score_zero_shot_gate(concept, target, rivals)
        return bool(ok)

    def _clip_text_hits(
        self,
        text: str,
        intent: Any,
        customer: str,
        *,
        intel_on: bool,
        threshold: float,
        uq: Any | None = None,
    ) -> tuple[dict[int, float], dict[str, Any]]:
        from core.semantic_pattern_intel import CLIP_FLOOR, record_is_searchable

        meta: dict[str, Any] = {
            "clip_attempted": False,
            "clip_hits": 0,
            "ai_faiss_used": False,
            "clip_prompt": "",
        }
        visual_on = bool(self.settings.search_text_visual) or intel_on
        self._last_clip_rivals = {}
        self._last_clip_object_prompt = False
        if getattr(self.faiss, "clip_map_unavailable", False):
            meta["clip_map_unavailable"] = True
            meta["clip_fallback"] = CLIP_MAP_UNAVAILABLE
            meta["visual_similarity_fallback"] = VISUAL_SIM_FALLBACK
            return {}, meta
        if not visual_on:
            return {}, meta
        if not (
            self.extractor.ai_available
            and getattr(self.extractor, "_clip_model", None) is not None
            and self.faiss.available
            and self.faiss.clip_count > 0
        ):
            return {}, meta
        from core.semantic_pattern_intel import competing_subtype_prompts

        t0 = time.perf_counter()
        motif = str(getattr(intent, "motif", "") or "") if intent is not None else ""
        prompt = ""
        object_prompt = False
        zs_spec = None
        try:
            from core.search_evidence_gate import zero_shot_spec as _zs_spec

            zs_spec = _zs_spec(text)
        except Exception:
            zs_spec = None
        if motif and intent is not None and getattr(intent, "clip_prompts", None):
            prompt = str(intent.clip_prompts[0] or "")
        elif zs_spec:
            prompt = str(zs_spec.get("prompt") or "")
        elif uq is not None and str(getattr(uq, "node_id", "") or ""):
            from core.universal_visual_intel import node_clip_prompt

            prompt = node_clip_prompt(str(uq.node_id))
            object_prompt = True
        prompt = prompt or text
        meta["clip_attempted"] = True
        meta["clip_prompt"] = prompt
        meta["clip_object_prompt"] = object_prompt
        self._last_clip_object_prompt = object_prompt
        text_emb = self._cached_text_embedding(prompt)
        if text_emb is None:
            return {}, meta
        ntotal = int(getattr(getattr(self.faiss, "clip_index", None), "ntotal", 0) or 0)
        clip_k = min(800, ntotal) if ntotal else 800
        floor = CLIP_FLOOR if intel_on else float(threshold)
        scored: list[tuple[int, float]] = []
        batches: list[dict[str, Any]] = []
        first_clip_ms = 0.0
        while True:
            raw = self.faiss.search_clip(text_emb, k=max(1, clip_k))
            scored = [(int(fid), float(sc)) for fid, sc in raw if float(sc) >= floor]
            if not batches:
                first_clip_ms = (time.perf_counter() - t0) * 1000.0
            batches.append({"k": clip_k, "scored": len(scored), "raw": len(raw)})
            tail = float(raw[-1][1]) if raw else 0.0
            if len(scored) >= 48 or tail < floor:
                break
            nxt = next_retrieve_k(clip_k, ntotal)
            if nxt is None:
                break
            clip_k = nxt
        meta["clip_k"] = clip_k
        meta["adaptive_clip"] = batches
        meta["first_clip_ms"] = round(first_clip_ms, 1)
        rival_by_fid: dict[int, dict[str, float]] = {}
        if intel_on and intent is not None:
            prompts = competing_subtype_prompts(intent)
            rival_k = min(max(clip_k, 800), 3200)
            for motif, rprompt in prompts.items():
                emb = self._cached_text_embedding(rprompt)
                if emb is None:
                    continue
                for fid, sc in self.faiss.search_clip(emb, k=rival_k):
                    if float(sc) < CLIP_FLOOR * 0.9:
                        continue
                    rival_by_fid.setdefault(int(fid), {})[str(motif)] = float(sc)
            meta["rival_clip_files"] = len(rival_by_fid)
            meta["rival_motifs"] = sorted(prompts.keys())
        if zs_spec:
            from core.search_evidence_gate import zero_shot_verify_prompts

            extra = zero_shot_verify_prompts(str(zs_spec.get("concept") or ""))
            zs_fids = [int(fid) for fid, _ in scored[:80]]
            for name, ptxt in extra.items():
                emb = self._cached_text_embedding(ptxt)
                if emb is None:
                    continue
                for fid, sc in self._clip_score_ids(emb, zs_fids).items():
                    rival_by_fid.setdefault(int(fid), {})[str(name)] = float(sc)
            meta["zs_verify_prompts"] = sorted(extra.keys())
        self._last_clip_rivals = rival_by_fid
        if not scored and not rival_by_fid:
            return {}, meta
        ids = [fid for fid, _ in scored]
        for fid in list(rival_by_fid)[:clip_k]:
            if fid not in ids:
                ids.append(fid)
        recs = self.db.get_indexed_files_by_ids(
            ids, include_processing_ready=True, lightweight=True
        )
        by_id = {int(r["id"]): r for r in recs}
        out: dict[int, float] = {}
        for fid, sc in scored:
            rec = by_id.get(fid)
            if rec is None:
                rec = self.db.get_file_by_id(fid)
            if not rec or not record_is_searchable(rec):
                continue
            if not intel_on and rec.get("status") != "indexed":
                continue
            if not self._record_in_scope(rec, customer):
                continue
            out[fid] = sc
        own = str(getattr(intent, "motif", "") or "")
        if intel_on and own:
            for fid, mscore in rival_by_fid.items():
                sc = float(mscore.get(own, 0.0) or 0.0)
                if sc < floor:
                    continue
                rec = by_id.get(fid)
                if rec is None:
                    rec = self.db.get_file_by_id(fid)
                if not rec or not record_is_searchable(rec):
                    continue
                if not self._record_in_scope(rec, customer):
                    continue
                out[fid] = max(out.get(fid, 0.0), sc)
        meta["clip_hits"] = len(out)
        meta["ai_faiss_used"] = bool(out)
        if object_prompt and uq is not None and out:
            nid = str(getattr(uq, "node_id", "") or "")
            if nid:
                concepts = dict(getattr(self, "_last_v2_concepts", {}) or {})
                for fid, sc in out.items():
                    concepts.setdefault(int(fid), {})[nid] = float(sc)
                self._last_v2_concepts = concepts
        return out, meta

    def _clip_score_prompt(
        self, prompt: str, *, k: int, floor: float, customer: str
    ) -> dict[int, float]:
        from core.semantic_pattern_intel import record_is_searchable

        emb = self.extractor.embed_text(prompt)
        if emb is None or not self.faiss.available:
            return {}
        out: dict[int, float] = {}
        hits = [(int(fid), float(sc)) for fid, sc in self.faiss.search_clip(emb, k=k) if float(sc) >= floor]
        if not hits:
            return {}
        recs = self.db.get_indexed_files_by_ids(
            [fid for fid, _ in hits], include_processing_ready=True, lightweight=True
        )
        by_id = {int(r["id"]): r for r in recs}
        for fid, sc in hits:
            rec = by_id.get(fid) or self.db.get_file_by_id(fid)
            if not rec or not record_is_searchable(rec):
                continue
            if not self._record_in_scope(rec, customer):
                continue
            out[fid] = sc
        return out

    def _extend_v2_clip(
        self,
        v2q: Any,
        clip_by_id: dict[int, float],
        customer: str,
    ) -> dict[int, float]:
        from core.pattern_intelligence_v2 import (
            REP_PHOTO,
            REP_TEXTILE,
            concept_prompt,
            representation_prompts,
        )
        from core.semantic_pattern_intel import CLIP_FLOOR

        if not (
            self.extractor.ai_available
            and getattr(self.extractor, "_clip_model", None) is not None
            and self.faiss.available
            and self.faiss.clip_count > 0
        ):
            self._last_v2_concepts = {}
            self._last_v2_rep = {}
            return clip_by_id

        concepts = dict(getattr(self, "_last_clip_rivals", {}) or {})
        floor = CLIP_FLOOR * 0.9
        extra_names = list(v2q.required)
        # Çiçek alt türlerinde genel floral kanıtı da topla. Spesifik rose
        # promptu düşük kalsa bile gerçek çiçek görseli composite sıralamaya
        # yardımcı olur; tek başına exact kabul edilmez.
        if v2q.composition == "composite" and any(
            str(x) in {"rose", "daisy", "tulip", "orchid", "peony"}
            for x in v2q.required
        ):
            extra_names.extend(["floral", "flower"])
        if v2q.composition == "single" and v2q.family == "animal_print":
            extra_names.append("floral")
        if v2q.composition == "single" and v2q.primary == "floral":
            extra_names.extend(["rose", "daisy"])
        seen: set[str] = set()
        for name in extra_names:
            if not name or name in seen:
                continue
            seen.add(name)
            hits = self._clip_score_prompt(concept_prompt(name), k=400, floor=floor, customer=customer)
            own = str(v2q.primary or "")
            for fid, sc in hits.items():
                concepts.setdefault(int(fid), {})[str(name)] = float(sc)
                if name == own or (v2q.composition == "composite" and name in v2q.required):
                    clip_by_id[int(fid)] = max(float(clip_by_id.get(int(fid), 0) or 0), float(sc))

        # Composite görsel sorgusu: tek tek "gül" ve "leopard" skorlarının
        # yanında aynı görselde iki kavramın birlikte bulunmasını hedefleyen
        # tek bir CLIP promptu çalıştır. Bu, "leopard" tek başına güçlü olan
        # dosyaların üstüne gerçekten "gül + leopard" kompozisyonlarını taşır.
        if v2q.composition == "composite" and len(v2q.required) >= 2:
            names = [str(x) for x in v2q.required if str(x).strip()]
            readable = " and ".join(names)
            if v2q.relationship == "overlay" and v2q.base_hint and v2q.overlay_hint:
                composite_prompts = [
                    f"textile fabric print with {v2q.overlay_hint} motif over a {v2q.base_hint} pattern, both clearly visible in the same design",
                    f"fabric pattern, {v2q.base_hint} background with {v2q.overlay_hint} floral motif, same textile design",
                ]
            else:
                composite_prompts = [
                    f"textile fabric print containing both {readable} motifs, both clearly visible in the same design, not separate images",
                    f"single textile pattern combining {readable}, the two motifs visibly coexist in one design",
                    f"textile print with {readable} together in the same composition, not one or the other",
                ]
            composite_hits: dict[int, float] = {}
            for composite_prompt in composite_prompts:
                hits = self._clip_score_prompt(
                    composite_prompt, k=800, floor=max(0.18, floor * 0.9), customer=customer
                )
                for fid, sc in hits.items():
                    composite_hits[int(fid)] = max(
                        float(composite_hits.get(int(fid), 0.0)), float(sc)
                    )
            for fid, sc in composite_hits.items():
                concepts.setdefault(int(fid), {})["_composite"] = float(sc)
                # Composite score is a candidate-retrieval signal. It is NOT
                # allowed to replace the individual concept scores.
                clip_by_id[int(fid)] = max(
                    float(clip_by_id.get(int(fid), 0) or 0), float(sc)
                )

        rep_map: dict[int, dict[str, float]] = {}
        want_rep = {REP_TEXTILE: representation_prompts()[REP_TEXTILE], REP_PHOTO: representation_prompts()[REP_PHOTO]}
        for rname, rprompt in want_rep.items():
            hits = self._clip_score_prompt(rprompt, k=300, floor=floor, customer=customer)
            for fid, sc in hits.items():
                rep_map.setdefault(int(fid), {})[str(rname)] = float(sc)

        self._last_v2_concepts = concepts
        self._last_v2_rep = rep_map
        return clip_by_id

    def _extend_uvi_clip(
        self,
        uq: Any,
        clip_by_id: dict[int, float],
        customer: str,
    ) -> dict[int, float]:
        from core.semantic_pattern_intel import CLIP_FLOOR
        from core.universal_visual_intel import clip_nodes_for_query, node_clip_prompt

        if not uq or not uq.node_id:
            return clip_by_id
        if not (
            self.extractor.ai_available
            and getattr(self.extractor, "_clip_model", None) is not None
            and self.faiss.available
            and self.faiss.clip_count > 0
        ):
            return clip_by_id

        concepts = dict(getattr(self, "_last_v2_concepts", {}) or {})
        already: set[str] = set()
        for mmap in concepts.values():
            already.update(str(k) for k in (mmap or {}))
        for mmap in (getattr(self, "_last_clip_rivals", {}) or {}).values():
            already.update(str(k) for k in (mmap or {}))

        # Person/gender is a general semantic concept just like flower,
        # leopard or car.  It must not depend on a literal text hit or on the
        # face index being complete.  Use a small prompt ensemble and a lower
        # retrieval floor; final ranking still uses the requested gender score.
        gender_query = str(getattr(uq, "node_id", "")) in {"female_person", "male_person"}
        floor = (CLIP_FLOOR * 0.70) if gender_query else (CLIP_FLOOR * 0.9)
        gender_prompts = {
            "female_person": (
                "a photo of a woman, female person, woman's face or body, female fashion portrait",
                "a woman person in a photograph, female human figure, portrait, not fabric",
                "female human subject, woman, model or actress, visible person",
            ),
            "male_person": (
                "a photo of a man, male person, man's face or body, male fashion portrait",
                "a man person in a photograph, male human figure, portrait, not fabric",
                "male human subject, man, model or actor, visible person",
            ),
        }
        for name in clip_nodes_for_query(uq):
            if not name:
                continue
            # Gender/person queries intentionally run their prompt ensemble on
            # every search. _clip_text_hits may already have stored the first
            # prompt in _last_v2_concepts; treating that as "already done"
            # silently disabled the stronger person/gender prompts.
            if not gender_query and name in already:
                continue
            already.add(name)
            prompts = gender_prompts.get(name, (node_clip_prompt(name),))
            if gender_query and name == "person":
                prompts = (
                    "a real photograph of a human person, visible person, portrait or fashion photo, not a textile fabric pattern",
                    "a human figure or person in a photograph or illustration, not fabric",
                )
            for prompt in prompts:
                hits = self._clip_score_prompt(prompt, k=400, floor=floor, customer=customer)
                for fid, sc in hits.items():
                    concepts.setdefault(int(fid), {})[str(name)] = max(
                        float(concepts.setdefault(int(fid), {}).get(str(name), 0.0) or 0.0),
                        float(sc),
                    )
                    if name == uq.node_id or name in (uq.required or []):
                        clip_by_id[int(fid)] = max(float(clip_by_id.get(int(fid), 0) or 0), float(sc))
        raw_n = str(getattr(uq, "raw", "") or "").lower()
        if "suv" in raw_n and "suv" not in already:
            already.add("suv")
            hits = self._clip_score_prompt(
                "SUV sport utility vehicle car, not sedan hatchback textile print",
                k=300,
                floor=floor,
                customer=customer,
            )
            for fid, sc in hits.items():
                concepts.setdefault(int(fid), {})["suv"] = float(sc)
                clip_by_id[int(fid)] = max(float(clip_by_id.get(int(fid), 0) or 0), float(sc))
        self._last_v2_concepts = concepts
        return clip_by_id

    def _candidate_clip_scores(
        self,
        fid: int,
        intent: Any,
        clip_by_id: dict[int, float],
        plan: Any | None = None,
    ) -> dict[str, float]:
        clip_here: dict[str, float] = {}
        sc = float(clip_by_id.get(int(fid), 0.0) or 0.0)
        if sc:
            clip_here["_similarity"] = sc
            motif = str(getattr(intent, "motif", "") or "") if intent is not None else ""
            if motif:
                clip_here[motif] = sc
            elif plan is not None:
                req = [c for c in getattr(plan, "objects", []) if getattr(c, "required", True)]
                if len(req) == 1:
                    clip_here[str(req[0].concept_id)] = sc
        rivals = dict((getattr(self, "_last_clip_rivals", {}) or {}).get(int(fid)) or {})
        clip_here.update(rivals)
        v2c = dict((getattr(self, "_last_v2_concepts", {}) or {}).get(int(fid)) or {})
        clip_here.update(v2c)
        return clip_here

    def _merge_semantic_intel(
        self,
        text: str,
        intent: Any,
        results: list[SearchResult],
        clip_by_id: dict[int, float],
        customer: str,
        plan: Any | None = None,
    ) -> list[SearchResult]:
        from core.semantic_pattern_intel import (
            TIER_TO_CATEGORY,
            collect_evidence,
            feedback_training_hint,
            gate_result,
            hybrid_score,
            cap_confidence_padding,
        )

        seen: dict[int, SearchResult] = {}
        for r in results:
            seen[int(r.file_id)] = r
        clip_only_ids = [fid for fid in clip_by_id if fid not in seen]
        if clip_only_ids:
            recs = self.db.get_indexed_files_by_ids(
                clip_only_ids, include_processing_ready=True, lightweight=True
            )
            recs = self._records_with_family_overrides(recs)
            rec_by = {int(r["id"]): r for r in recs}
            for fid in clip_only_ids:
                rec = rec_by.get(fid)
                if not rec:
                    continue
                if not self._record_in_scope(rec, customer):
                    continue
                clip = clip_by_id.get(fid, 0.0)
                dummy = self._to_result(
                    rec, hybrid_score(0.0, clip), {"clip": clip, "text": 0.0}
                )
                dummy.debug["text_mode"] = True
                dummy.debug["clip_only"] = True
                seen[fid] = dummy

        taught: set[int] = set()
        try:
            from core.search_memory import overlay_positive_ids

            taught = overlay_positive_ids(getattr(self.settings, "db_path", ""), text)
        except Exception:
            taught = set()
        try:
            pack = getattr(self, "_last_learned_concept", None) or {}
            taught |= {int(x) for x in (pack.get("exact_ids") or []) if int(x) > 0}
            taught |= {int(x) for x in (pack.get("neighbor_scores") or {}) if int(x) > 0}
        except Exception:
            pass
        missing_taught = [fid for fid in taught if fid not in seen]
        if missing_taught:
            recs = self.db.get_indexed_files_by_ids(
                missing_taught, include_processing_ready=True, lightweight=True
            )
            recs = self._records_with_family_overrides(recs)
            rec_by = {int(r["id"]): r for r in recs}
            for fid in missing_taught:
                rec = rec_by.get(fid)
                if not rec or not self._record_in_scope(rec, customer):
                    continue
                dummy = self._to_result(rec, 0.72, {"text": 1.0, "user_taught": 1.0})
                dummy.debug["text_mode"] = True
                dummy.debug["user_taught_positive"] = True
                seen[fid] = dummy

        feedback_adj = {}
        wrong_ids: set[int] = set()
        try:
            feedback_adj = self._feedback.score_adjustments(text)
            wrong_ids = self._feedback.wrong_result_ids(text)
        except Exception:
            feedback_adj = {}
            wrong_ids = set()
        try:
            from core.search_memory import overlay_adjustments, overlay_wrong_ids

            dbp = getattr(self.settings, "db_path", "")
            _cust = str(getattr(self, "_active_search_customer", "") or "")
            for fid, delta in overlay_adjustments(dbp, text, customer=_cust).items():
                feedback_adj[fid] = feedback_adj.get(fid, 0.0) + float(delta)
            wrong_ids |= overlay_wrong_ids(dbp, text, customer=_cust)
        except Exception:
            pass

        kept: list[SearchResult] = []
        stats = {
            "clip_contributed": 0,
            "metadata_contributed": 0,
            "dna_contributed": 0,
            "family_contributed": 0,
            "dropped_low_confidence": 0,
            "rejected_competing_subtype": 0,
            "negative_evidence": 0,
            "tiers": {},
        }
        need_ids = [fid for fid in seen if fid not in wrong_ids]
        detailed_map: dict[int, dict[str, Any]] = {}
        if need_ids:
            try:
                rows = self.db.get_indexed_files_by_ids(
                    need_ids, include_processing_ready=True, lightweight=True
                )
                detailed_map = {int(r["id"]): r for r in rows}
            except Exception:
                detailed_map = {}

        for fid, result in seen.items():
            if fid in wrong_ids:
                stats["dropped_low_confidence"] += 1
                continue
            feat = detailed_map.get(fid) or {}
            rec = {
                "id": result.file_id,
                "filename": result.filename,
                "path": result.path,
                "ocr_text": feat.get("ocr_text") or "",
                "status": feat.get("status") or "pending",
                "pattern_family": feat.get("pattern_family") or result.pattern_family,
                "pattern_type": feat.get("pattern_type") or "",
                "pattern_subtype": feat.get("pattern_subtype") or "",
                "texture_map": feat.get("texture_map") or {},
                "feedback_labels": feat.get("feedback_labels") or [],
                "physical_preview_ready": feat.get("physical_preview_ready") or 0,
            }
            clip = float(clip_by_id.get(fid, result.breakdown.get("clip") or 0.0) or 0.0)
            rivals = dict((getattr(self, "_last_clip_rivals", {}) or {}).get(fid) or {})
            if intent and intent.motif and clip and intent.motif not in rivals:
                rivals[str(intent.motif)] = clip
            ev = collect_evidence(intent, rec, clip_score=clip, rival_clip=rivals)
            text_score = float(result.score or 0.0)
            if result.debug.get("clip_only"):
                text_score = 0.0
            ok, tier, score = gate_result(intent, ev, text_score=text_score)
            if fid in taught:
                ok = True
                score = max(float(score or 0.0), 0.72)
                result.debug["user_taught_positive"] = True
            if not ok:
                stats["dropped_low_confidence"] += 1
                if ev.negative_motifs or (ev.rival_clip and not ev.visual_win):
                    stats["rejected_competing_subtype"] += 1
                continue
            if result.debug.get("clip_only"):
                from core.query_evidence import (
                    apply_query_evidence,
                    build_search_plan,
                    collect_candidate_evidence,
                )

                qplan = plan or build_search_plan(text)
                clip_here = self._candidate_clip_scores(fid, intent, clip_by_id, qplan)
                qev = collect_candidate_evidence(
                    qplan,
                    rec,
                    texture_map=rec.get("texture_map") or {},
                    clip_scores=clip_here,
                )
                score, qev = apply_query_evidence(score, qplan, qev)
                score = min(float(score), 0.86)
                result.debug["query_evidence_report"] = qev.to_dict()
                result.debug["search_evidence_plan"] = qplan.to_dict()
                result.debug["visual_verdict"] = qev.visual_verdict
                result.debug["visual_grade"] = qev.visual_grade
                result.debug["clip_only"] = True
                result.debug["query_evidence_applied"] = True
            if fid in feedback_adj:
                score = max(0.0, min(1.0, score + float(feedback_adj[fid])))
            result.score = score
            result.score_percent = round(score * 100, 1)
            result.category = TIER_TO_CATEGORY.get(tier, result.category)
            if intent.family == "animal_print" and tier == "Same Family":
                result.category = "animal_print"
            result.debug = {
                **result.debug,
                "semantic_tier": tier,
                "semantic_intent": intent.to_dict(),
                "semantic_evidence": ev.to_dict(),
                "clip_score": round(clip, 4),
                "clip_margin": ev.clip_margin,
                "rival_clip": ev.rival_clip,
                "visual_win": ev.visual_win,
                "negative_motifs": ev.negative_motifs,
                "ai_faiss_used": clip > 0,
                "feedback_training_hint": feedback_training_hint(rec),
                "user_feedback_delta": round(float(feedback_adj.get(fid, 0.0)), 4),
                "clip_object_prompt": bool(getattr(self, "_last_clip_object_prompt", False)),
            }
            result.breakdown = {**result.breakdown, "clip": clip, "hybrid": score}
            result.cluster_reason = tier
            if clip >= 0.20:
                stats["clip_contributed"] += 1
            if ev.filename_hit or ev.ocr_hit:
                stats["metadata_contributed"] += 1
            if ev.dna_hit or ev.subtype_hit:
                stats["dna_contributed"] += 1
            if ev.family_hit:
                stats["family_contributed"] += 1
            if ev.negative_motifs:
                stats["negative_evidence"] += 1
            stats["tiers"][tier] = stats["tiers"].get(tier, 0) + 1
            kept.append(result)

        kept = cap_confidence_padding(intent, kept)
        self._last_text_intel_meta = {
            "semantic_pattern_intel": True,
            "intent": intent.to_dict() if intent else {},
            **stats,
        }
        return kept

    def _apply_text_query_teach(
        self,
        text: str,
        results: list[SearchResult],
        customer: str,
    ) -> list[SearchResult]:
        """Inject/keep user-taught positives for this exact text query."""
        dbp = getattr(self.settings, "db_path", "")
        try:
            from core.search_memory import overlay_positive_ids, overlay_wrong_ids

            pos = overlay_positive_ids(dbp, text)
            wrong = overlay_wrong_ids(dbp, text)
        except Exception:
            return results
        if not pos and not wrong:
            return results
        kept: list[SearchResult] = []
        seen: set[int] = set()
        for row in results:
            fid = int(getattr(row, "file_id", 0) or 0)
            if fid in wrong:
                continue
            if fid in pos:
                dbg = dict(getattr(row, "debug", {}) or {})
                dbg["user_taught_positive"] = True
                row.debug = dbg
                row.score = max(float(row.score or 0.0), 0.72)
                row.score_percent = round(float(row.score) * 100, 1)
            kept.append(row)
            seen.add(fid)
        missing = [fid for fid in pos if fid not in seen and fid not in wrong]
        if missing:
            recs = self.db.get_indexed_files_by_ids(
                missing, include_processing_ready=True, lightweight=True
            )
            recs = self._records_with_family_overrides(recs)
            for rec in recs:
                fid = int(rec.get("id") or 0)
                if not fid or not self._record_in_scope(rec, customer):
                    continue
                row = self._to_result(rec, 0.72, {"text": 1.0, "user_taught": 1.0})
                row.debug["text_mode"] = True
                row.debug["user_taught_positive"] = True
                kept.append(row)
        return kept

    def _promote_learned_results(
        self,
        results: list[SearchResult],
        customer: str = "",
    ) -> list[SearchResult]:
        """Re-inject taught examples/neighbors after later rankers drop them."""
        pack = getattr(self, "_last_learned_concept", None) or {}
        if not pack:
            return results
        try:
            from core.learned_concept_search import apply_learned_to_results
        except Exception:
            return results
        have = {int(getattr(r, "file_id", 0) or 0) for r in results}
        missing = [
            int(fid)
            for fid in (pack.get("file_scores") or {})
            if int(fid) > 0 and int(fid) not in have
        ]
        extra_rows: list[SearchResult] = []
        if missing:
            recs = self.db.get_indexed_files_by_ids(
                missing[:800],
                include_processing_ready=True,
                lightweight=True,
            )
            recs = self._records_with_family_overrides(recs)
            scores = pack.get("file_scores") or {}
            for rec in recs:
                if not self._record_in_scope(rec, customer):
                    continue
                fid = int(rec.get("id") or 0)
                extra_rows.append(
                    self._to_result(
                        rec,
                        float(scores.get(fid, 0.72) or 0.72),
                        {"learned_concept": 1.0, "text": 0.0},
                    )
                )
        return apply_learned_to_results(results, pack, extra_rows)

    def search_by_text(
        self,
        text: str,
        limit: int | None = None,
        threshold: float | None = None,
        customer: str = "",
        result_callback: Callable[[list[SearchResult], int, int], None] | None = None,
    ) -> list[SearchResult]:
        from core.text_index import text_search_score
        from core.textile_terms import expand_query_terms, fuzzy_correct_query, query_family_hints
        from core.index_freeze import in_search_session, search_session

        if not in_search_session():
            with search_session():
                return self.search_by_text(
                    text,
                    limit=limit,
                    threshold=threshold,
                    customer=customer,
                    result_callback=result_callback,
                )

        requested_limit = self.settings.search_result_limit if limit is None else int(limit)
        unlimited = int(requested_limit) <= 0
        limit = 0 if unlimited else min(int(requested_limit), 800)
        threshold = (
            threshold if threshold is not None else self.settings.similarity_threshold
        )
        customer = customer or self.settings.customer_filter
        self._active_search_customer = str(customer or "").strip()
        text = (text or "").strip()
        if not text:
            return []
        user_query_text = text
        try:
            from core.search_memory import apply_query_memory

            text, mem_meta = apply_query_memory(
                self.settings.db_path, text, customer=customer
            )
            self._last_memory_meta = dict(mem_meta or {})
        except Exception:
            self._last_memory_meta = {}

        try:
            from core.query_intent_router import classify_query as _classify_route

            _route = _classify_route(
                text, db_path=getattr(self.settings, "db_path", "")
            )
            from core.query_intent_router import intent_retrieval_needles as _intent_needles

            _concept_needles = _intent_needles(_route)
            self._last_query_intent = _route
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                "search_channel": _route.channel,
                "search_channel_label": _route.label,
                "search_channels": [c.to_dict() for c in _route.channels],
                "unknown_tokens": list(_route.unknown_tokens),
                "concept_needles": list(_concept_needles),
                "query_intent": _route.to_dict(),
            }
        except Exception:
            self._last_query_intent = None
            pass

        # Query Intent Router + Face Intelligence.
        # Identity/name queries may short-circuit only when the persistent face
        # gallery has a real match. Gender queries are deliberately NOT
        # short-circuited: they combine the face index with UVI semantic evidence,
        # so an incomplete/background face index cannot hide valid images.
        face_query_intent = None
        face_ids = []
        if bool(getattr(self.settings, "face_index_enabled", False)):
            try:
                from core.query_intent_router import classify_query
                face_query_intent = classify_query(text)
                from core.face_search import FaceSearch
                _fs = FaceSearch(getattr(self.settings, "face_db_path", ""))
                face_ids = _fs.file_ids(text, limit=limit)

                if face_query_intent.kind in {"person", "person_name"} and face_ids:
                    out = []
                    for fid in face_ids:
                        rec = self.db.get_file_by_id(fid)
                        if not rec:
                            continue
                        out.append(self._to_result(
                            rec, 1.0,
                            {
                                "face": 1.0,
                                "face_query_kind": face_query_intent.kind,
                                "face_match": True,
                                "face_search_source": "persistent_gallery",
                            },
                        ))
                    if out:
                        return out

                # A named person may exist in the main file index before the
                # face scanner has reached that file. Use normalized filename
                # metadata as a safe fallback; this is metadata evidence, not
                # a biometric identity claim.
                if face_query_intent.kind == "person_name" and not face_ids:
                    name_terms = list(dict.fromkeys(
                        [face_query_intent.value]
                        + list(face_query_intent.tokens)
                    ))
                    meta_candidates = self.db.search_text_candidates(
                        name_terms,
                        limit=min(limit, 400),
                        customer=customer,
                    )
                    if meta_candidates:
                        from core.textile_terms import normalize_turkish as _norm_name
                        name_tokens = [str(t) for t in face_query_intent.tokens if len(str(t)) >= 3]
                        out = []
                        for rec in meta_candidates:
                            if not rec:
                                continue
                            hay = _norm_name(
                                f"{rec.get('filename','')} {rec.get('path','')}"
                            )
                            # Metadata fallback is an exact multi-token filename/path
                            # check. It must not turn a two-word textile query into
                            # an accidental person-name search.
                            if name_tokens and not all(tok in hay for tok in name_tokens):
                                continue
                            out.append(self._to_result(
                                rec, 0.995,
                                {
                                    "face": 0.0,
                                    "face_query_kind": "person_name",
                                    "person_name_metadata": True,
                                    "face_match": False,
                                    "name_match": True,
                                },
                            ))
                        if out:
                            return out
            except Exception:
                # Face subsystem must never break the existing pattern search.
                face_query_intent = face_query_intent

        # Spatial Intelligence V1: OBJECT+RELATION+OBJECT on object bboxes.
        # Separate from Pattern/DINO/CLIP weights. Read-only object_index.
        from core.spatial.spatial_query import parse_spatial_query

        spatial_spec = parse_spatial_query(text)
        if spatial_spec:
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                "spatial_query": True,
                "spatial_relation": str(spatial_spec["relation"].value),
                "clip_as_relation": False,
            }
            obj_on = bool(getattr(self.settings, "object_index_enabled", False))
            obj_path = getattr(self.settings, "object_db_path", "") or ""
            from core.textile_motif_v4 import textile_spatial_query

            tpath = getattr(self.settings, "textile_motif_db_path", "") or ""
            _textile_spatial = textile_spatial_query(spatial_spec)
            if _textile_spatial:
                if not tpath or not Path(tpath).is_file():
                    self._last_text_intel_meta["spatial_status"] = "SPATIAL_EVIDENCE_UNAVAILABLE"
                    return []
            elif not obj_on or not obj_path:
                self._last_text_intel_meta["spatial_status"] = "SPATIAL_EVIDENCE_UNAVAILABLE"
                return []
            try:
                from core.spatial.spatial_engine import SPATIAL_EVIDENCE_UNAVAILABLE, SpatialEngine

                engine = SpatialEngine(
                    obj_path or tpath,
                    settings=self.settings,
                    readonly=True,
                    textile_db_path=tpath if _textile_spatial else None,
                )
                image_sizes: dict[int, tuple[int, int]] = {}
                cand = engine.store.search_labels_and(
                    [
                        {"labels": spatial_spec["object_a"]["labels"], "min_count": 1},
                        {"labels": spatial_spec["object_b"]["labels"], "min_count": 1},
                    ],
                    limit=max(limit or 400, 4000),
                )
                for fid in cand:
                    rec = self.db.get_file_by_id(fid)
                    if rec:
                        image_sizes[int(fid)] = (
                            int(rec.get("width") or 0),
                            int(rec.get("height") or 0),
                        )
                found = engine.search(text, limit=limit or 400, image_sizes=image_sizes)
                if found.status == SPATIAL_EVIDENCE_UNAVAILABLE:
                    self._last_text_intel_meta["spatial_status"] = SPATIAL_EVIDENCE_UNAVAILABLE
                    return []
                out = []
                for hit in found.hits:
                    rec = self.db.get_file_by_id(hit.file_id)
                    if not rec:
                        continue
                    result = self._to_result(
                        rec,
                        float(hit.spatial_score),
                        {
                            "spatial_score": float(hit.spatial_score),
                            "spatial_confidence": float(hit.spatial_confidence),
                            "spatial_relation": hit.spatial_relation,
                            "spatial_evidence": hit.spatial_evidence,
                            "clip_as_detection": False,
                            "clip_as_relation": False,
                        },
                    )
                    result.debug.update(
                        {
                            "spatial_score": float(hit.spatial_score),
                            "spatial_confidence": float(hit.spatial_confidence),
                            "spatial_relation": hit.spatial_relation,
                            "spatial_evidence": hit.spatial_evidence,
                        }
                    )
                    result.match_explanations = [hit.explanation]
                    out.append(result)
                self._last_text_intel_meta["spatial_status"] = found.status
                self._last_text_intel_meta["spatial_explanation"] = found.explanation
                return out
            except Exception:
                logger.warning("Spatial search skipped; not mixing into pattern ranking", exc_info=True)
                self._last_text_intel_meta["spatial_status"] = "SPATIAL_EVIDENCE_UNAVAILABLE"
                return []

        # Object Intelligence: read-only detector evidence. Off by default.
        # Never writes object_index.db or Pattern Index. CLIP is not detection.
        # Object-class queries do not mix into pattern ranking.
        if bool(getattr(self.settings, "object_index_enabled", False)):
            obj_spec = None
            try:
                from core.object_search import ObjectSearch, parse_object_query

                obj_spec = parse_object_query(text)
                if obj_spec:
                    _os = ObjectSearch(getattr(self.settings, "object_db_path", "") or "", readonly=True)
                    obj_ids = _os.file_ids(text, limit=limit or 400)
                    out = []
                    for fid in obj_ids:
                        rec = self.db.get_file_by_id(fid)
                        if not rec:
                            continue
                        out.append(self._to_result(
                            rec, 1.0,
                            {
                                "object": 1.0,
                                "object_match": True,
                                "object_query_kind": obj_spec.get("kind"),
                                "object_evidence": "detector_class",
                                "clip_as_detection": False,
                            },
                        ))
                    return out
            except Exception:
                logger.warning("Object search skipped; not mixing into pattern ranking", exc_info=True)
                if obj_spec:
                    return []

        # Yazım düzeltme — ontoloji terimleri (lale vb.) korunur
        # Case / TR / typo normalization (QUERY CORRECTION only).
        intel_on = self._intel_enabled()
        intent = None
        from core.concept_query_normalize import normalize_concept_query

        _nq = normalize_concept_query(text)
        if intel_on:
            from core.semantic_pattern_intel import parse_pattern_intent, should_skip_fuzzy

            intent = parse_pattern_intent(text)
            if should_skip_fuzzy(text):
                corrected_text, was_corrected = text, False
            elif _nq.typo_tier == "auto" and _nq.was_typo_corrected:
                corrected_text, was_corrected = _nq.corrected, True
            else:
                corrected_text, was_corrected = fuzzy_correct_query(text)
        else:
            if _nq.typo_tier == "auto" and _nq.was_typo_corrected:
                corrected_text, was_corrected = _nq.corrected, True
            else:
                corrected_text, was_corrected = fuzzy_correct_query(text)
        if was_corrected and corrected_text and corrected_text != text:
            import logging
            logging.getLogger(__name__).info(
                "Yazım düzeltme: %r → %r", text, corrected_text
            )
            try:
                from core.search_memory import remember_query_rewrite

                remember_query_rewrite(
                    self.settings.db_path,
                    locals().get("user_query_text", text),
                    corrected_text,
                    kind="typo",
                )
            except Exception:
                pass
        _search_texts = list(dict.fromkeys(
            [t for t in [text, corrected_text if was_corrected else None] if t]
        ))

        results: list[SearchResult] = []
        v2_on = bool(getattr(self.settings, "pattern_intelligence_v2_enabled", True))
        uvi_on = bool(getattr(self.settings, "universal_visual_intel_enabled", True))
        v2q = None
        uq = None
        # Yazım düzeltmesi yapıldıysa AI sorgu planı DA düzeltilmiş metni
        # kullanmalı. Aksi halde "gül leoaprd" için doğal dil katmanı
        # leopard'ı bulsa bile V2/UVI yalnızca "gül" planıyla kalır.
        query_for_intel = corrected_text if was_corrected else text
        if intel_on and v2_on:
            from core.pattern_intelligence_v2 import parse_pattern_query_v2

            v2q = parse_pattern_query_v2(query_for_intel)
        if intel_on and uvi_on:
            from core.universal_visual_intel import parse_universal_query

            uq = parse_universal_query(query_for_intel)
            # Preserve the explicit query route for diagnostics/ranking.
            if face_query_intent is not None:
                self._last_text_intel_meta = {
                    **(getattr(self, "_last_text_intel_meta", {}) or {}),
                    "query_intent": face_query_intent.to_dict(),
                }
        # Marka sorgusu CLIP'i kapatmaz: görsel benzerlik marka kovasının
        # içinde ve markasız lookalike'larda ikinci sinyal olarak kalır.
        clip_by_id, clip_meta = self._clip_text_hits(
            text,
            intent,
            customer,
            intel_on=intel_on,
            threshold=float(threshold),
            uq=uq,
        )
        if v2q is not None:
            clip_by_id = self._extend_v2_clip(v2q, clip_by_id, customer)
        if uq is not None:
            clip_by_id = self._extend_uvi_clip(uq, clip_by_id, customer)
        if not intel_on:
            visual_weight = 0.15 if self.settings.search_text_visual else 0.0
            for file_id, score in clip_by_id.items():
                rec = self.db.get_file_by_id(file_id)
                if not rec:
                    continue
                results.append(
                    self._to_result(
                        rec,
                        score * visual_weight + 0.1,
                        {"clip": score, "text": 0.0},
                    )
                )

        semantic_enabled = bool(
            getattr(self.settings, "semantic_text_search_enabled", False)
        )
        from core.brand_aliases import query_brand_needles
        from core.category_tree import resolve_category_query
        from core.natural_language_query import (
            COLOR_PHRASES,
            expand_parsed_terms,
            parse_natural_query,
        )
        from core.textile_terms import normalize_turkish

        # Her iki formun (orijinal + düzeltilmiş) birleşik term listesi
        effective_text = corrected_text if was_corrected else text
        _search_db_path = getattr(
            getattr(self, "db", None),
            "db_path",
            getattr(self.settings, "db_path", ""),
        )
        parsed = parse_natural_query(effective_text)
        terms = expand_query_terms(effective_text, include_semantic=semantic_enabled)
        # Orijinal form da eklenir — doğruysa kayıp olmasın
        if was_corrected:
            orig_terms = expand_query_terms(text, include_semantic=False)
            terms = list(dict.fromkeys(terms + orig_terms))
        terms = list(dict.fromkeys(terms + expand_parsed_terms(parsed)))
        cat_match = resolve_category_query(effective_text)
        # Marka sorguları özel modda çalışır:
        #   "marka" -> tüm marka kanıtları
        #   "amiri/dior/..." -> yalnızca o markanın kanıtlı kayıtları
        # Böylece marka adı Monogram/Logo gibi başka bir ailede kayıtlı olsa bile
        # marka aramasına dahil edilir; ancak başka marka sonuçları sızmaz.
        brand_query_parent = bool(
            cat_match.category_path
            and cat_match.primary_family == normalize_turkish("Marka")
            and not cat_match.primary_subtype
        )
        brand_query_child = bool(
            cat_match.category_path
            and cat_match.primary_family == normalize_turkish("Marka")
            and bool(cat_match.primary_subtype)
        )
        from core.brand_aliases import resolve_brand_alias
        dynamic_brand = resolve_brand_alias(effective_text, getattr(getattr(self, "db", None), "db_path", getattr(self.settings, "db_path", ""))) or ""
        brand_query_canonical = ""
        if brand_query_child:
            brand_query_canonical = normalize_turkish(
                dynamic_brand
                or resolve_brand_alias(cat_match.primary_subtype, _search_db_path)
                or cat_match.primary_subtype
            )
        elif dynamic_brand:
            brand_query_child = True
            brand_query_canonical = normalize_turkish(dynamic_brand)
        if brand_query_canonical:
            from core.brand_aliases import normalize_brand_key as _nbk

            brand_query_canonical = _nbk(brand_query_canonical)
        # Marka araması ranking sinyalidir, sert filtre değil.
        brand_search_mode = bool(brand_query_parent or brand_query_child)
        if cat_match.category_path:
            terms = list(dict.fromkeys(terms + list(cat_match.aliases)))
        hints = query_family_hints(effective_text, include_semantic=semantic_enabled)
        q_norm = normalize_turkish(effective_text)
        terms_norm = [normalize_turkish(t) for t in terms if str(t).strip()]
        brand_needles = query_brand_needles(effective_text, _search_db_path)

        from core.query_evidence import build_search_plan, plan_retrieval_needles

        evidence_plan = build_search_plan(
            effective_text, parsed=parsed, v2q=v2q, hints=hints
        )
        # Aday: düzeltilmiş + orijinal form (FTS + dosya adı LIKE)
        short_terms = [effective_text]
        if was_corrected:
            short_terms.append(text)
        short_terms += plan_retrieval_needles(evidence_plan)
        _route = getattr(self, "_last_query_intent", None)
        from core.concept_channel_retrieval import (
            apply_compound_evidence_floor,
            is_compound_intent,
            primary_evidenced_ids,
            retrieve_compound,
        )

        _compound_v3 = is_compound_intent(_route)
        _v3_report = None
        _union_fb = False
        if _route is not None and not brand_query_canonical:
            for _ch in getattr(_route, "channels", ()) or ():
                if str(getattr(_ch, "channel", "") or "") == "brand":
                    from core.brand_aliases import resolve_brand_alias as _route_brand

                    brand_query_canonical = normalize_turkish(
                        _route_brand(str(_ch.value or ""), _search_db_path)
                        or str(_ch.value or "")
                    )
                    brand_query_child = True
                    brand_search_mode = True
                    break
        if brand_query_canonical:
            from core.brand_aliases import normalize_brand_key as _nbk2

            brand_query_canonical = _nbk2(brand_query_canonical)
        # Compound (amiri çiçek) marka sinyalini kapatmaz; kanıt + görsel birlikte sıralanır.
        if _route is not None:
            from core.query_intent_router import intent_retrieval_needles

            _ch_needles = intent_retrieval_needles(_route)
            short_terms += _ch_needles
            terms = list(dict.fromkeys(list(terms) + _ch_needles))
            terms_norm = [normalize_turkish(t) for t in terms if str(t).strip()]
        color_skip = {
            normalize_turkish(x)
            for x in list(COLOR_PHRASES.keys()) + list(COLOR_PHRASES.values())
        }
        extra = []
        for t in terms:
            if len(str(t).split()) > 2:
                continue
            tn = normalize_turkish(t)
            if tn in color_skip or tn in {"red", "blue", "pink", "green"}:
                continue
            extra.append(t)
            if len(extra) >= 6:
                break
        short_terms += extra
        short_terms = list(dict.fromkeys(short_terms))
        sf = self._search_filter(customer)
        cand_limit = min(400, max(limit * 2, 200))
        # Detector exact-id union is independent of the FTS/CLIP window.
        # 600 bird files were truncated to 400 here, then CLIP-thresholded to 2.
        object_index_retrieval_limit = 8000
        if _compound_v3 and _route is not None:
            def _channel_search(needles: list[str]) -> list[int]:
                bn = {
                    normalize_turkish(x)
                    for x in (brand_needles or [])
                    if str(x).strip() and (len(str(x).strip()) >= 4 or "&" in str(x))
                }
                nt = {normalize_turkish(x) for x in (needles or []) if str(x).strip()}
                use_ungated = bool(bn and (bn & nt))
                rows = (
                    self.db.search_filename_path_candidates(
                        needles,
                        limit=cand_limit,
                        customer=customer,
                        source_types=sf.source_types(),
                        source_ids=sf.source_ids or None,
                    )
                    if use_ungated
                    else self.db.search_text_candidates(
                        needles,
                        limit=cand_limit,
                        customer=customer,
                        source_types=sf.source_types(),
                        source_ids=sf.source_ids or None,
                    )
                )
                out: list[int] = []
                for rec in rows or []:
                    try:
                        fid = int(rec.get("id", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if fid > 0:
                        out.append(fid)
                return out

            def _motif_box_search(needles: list[str]) -> list[int]:
                from core.textile_motif_v4 import MOTIF_CLASSES, open_motif_store

                store = open_motif_store(
                    getattr(self.settings, "textile_motif_db_path", "") or ""
                )
                if store is None:
                    return []
                wanted = [x for x in needles if x in MOTIF_CLASSES]
                if not wanted:
                    return []
                return store.search_label(wanted, limit=cand_limit)

            _v3_report = retrieve_compound(
                _route,
                _channel_search,
                per_channel_limit=cand_limit,
                motif_box_search=_motif_box_search,
            )
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                **_v3_report.to_meta(),
            }
            if _v3_report.candidate_ids:
                candidates = self.db.get_indexed_files_by_ids(
                    _v3_report.candidate_ids[: max(cand_limit * 2, 200)],
                    include_processing_ready=True,
                    include_pending=True,
                    lightweight=True,
                )
            else:
                candidates = []
        elif brand_query_parent or brand_query_child:
            # Child (Amiri/Versace/…): önce filename/path + FTS/OCR.
            # Aday varsa 8454'lük havuzu tarama. Boşsa eski full-scan.
            # Parent ("marka"): tam havuz + filename ekleri (değişmez).
            extra_needles = [
                n
                for n in (brand_needles or [])
                if len(str(n).strip()) >= 4 or "&" in str(n)
            ] or [effective_text]
            kw = dict(
                customer=customer,
                source_types=sf.source_types(),
                source_ids=sf.source_ids or None,
            )

            def _merge_brand_rows(*groups: list) -> list:
                out: list = []
                have: set[int] = set()
                for group in groups:
                    for rec in group or []:
                        fid = int(rec.get("id") or 0)
                        if fid > 0 and fid not in have:
                            out.append(rec)
                            have.add(fid)
                return out

            if brand_query_child:
                # db.search_filename_path_candidates: 0 -> 400; sınırsızda tavanı yükselt.
                brand_fast_limit = (
                    50_000 if int(limit or 0) <= 0 else max(int(cand_limit), 400)
                )
                extra = self.db.search_filename_path_candidates(
                    extra_needles, limit=brand_fast_limit, **kw
                )
                fts_hits = []
                _stc = getattr(self.db, "search_text_candidates", None)
                if callable(_stc):
                    fts_hits = _stc(extra_needles, limit=brand_fast_limit, **kw) or []
                fast = _merge_brand_rows(extra, fts_hits)
                if fast:
                    candidates = fast
                else:
                    candidates = self._indexed_files(customer, lightweight=True)
            else:
                candidates = self._indexed_files(customer, lightweight=True)
                extra = self.db.search_filename_path_candidates(
                    extra_needles,
                    limit=max(cand_limit, 400),
                    **kw,
                )
                if extra:
                    candidates = _merge_brand_rows(candidates, extra)
            taught_ids: list[int] = []
            if brand_query_canonical:
                try:
                    from core.search_memory import file_ids_for_taught_brand

                    taught_ids = file_ids_for_taught_brand(
                        _search_db_path, brand_query_canonical
                    )
                except Exception:
                    taught_ids = []
            if taught_ids:
                getter = getattr(self.db, "get_indexed_files_by_ids", None)
                taught_rows = (
                    getter(
                        taught_ids[:800],
                        include_processing_ready=True,
                        lightweight=True,
                    )
                    if callable(getter)
                    else []
                )
                if taught_rows:
                    candidates = _merge_brand_rows(candidates, taught_rows)
        else:
            candidates = self.db.search_text_candidates(
                short_terms,
                limit=cand_limit,
                customer=customer,
                source_types=sf.source_types(),
                source_ids=sf.source_ids or None,
            )
        candidates = self._records_with_family_overrides(candidates)
        if _compound_v3 and _v3_report is not None:
            _v3_gap = any(
                str(getattr(layer, "status", "") or "") != "HAS_EVIDENCE"
                for layer in (_v3_report.layers or [])
            )
            if _v3_gap or str(getattr(_v3_report, "policy", "") or "") != "AND":
                # Missing channel must not AND-zero the query.
                evidence_plan.and_groups = []
            _union_fb = not bool(evidence_plan.and_groups or [])

        # Gender/person queries use the persistent face index as an additional
        # retrieval source.  The face gallery is intentionally not the only
        # source (UVI also finds illustrations/silhouettes without a detected
        # face), but known gender matches must never be lost because the normal
        # FTS candidate window does not contain the words "kadın/erkek".
        if face_query_intent is not None and face_query_intent.kind == "gender" and face_ids:
            gender_rows = self.db.get_indexed_files_by_ids(
                [int(fid) for fid in face_ids[:limit] if int(fid) > 0],
                include_processing_ready=True,
                lightweight=True,
            )
            if gender_rows:
                gender_rows = self._records_with_family_overrides(gender_rows)
                existing_ids = {int(r.get("id", 0) or 0) for r in candidates}
                for rec in gender_rows:
                    fid = int(rec.get("id", 0) or 0)
                    if fid <= 0 or fid in existing_ids:
                        continue
                    rec["_face_gender_match"] = True
                    candidates.append(rec)

        # Görsel Nesne/Kavram DNA: filename gerekmez. Read-only object_index union.
        try:
            from core.visual_concept_dna import lookup_concept_file_ids, parse_search_bags

            _bags = parse_search_bags(effective_text)
            _lemmas = list(_bags.objects) + list(_bags.persons)
            _parent_lemmas = [x for x in (_bags.parent_objects or []) if x not in _lemmas]
            if _bags.need_object_retrieval and (_lemmas or _parent_lemmas):
                _obj_path = str(getattr(self.settings, "object_db_path", "") or "")
                _dna_ids = lookup_concept_file_ids(
                    _obj_path, _lemmas, limit=object_index_retrieval_limit,
                ) if _lemmas else []
                _parent_ids = lookup_concept_file_ids(
                    _obj_path, _parent_lemmas, limit=cand_limit,
                ) if _parent_lemmas else []
                _exact = set(_dna_ids)
                _merged = list(_dna_ids)
                for i in _parent_ids:
                    if i not in _exact and i not in _merged:
                        _merged.append(i)
                self._last_text_intel_meta = {
                    **(getattr(self, "_last_text_intel_meta", {}) or {}),
                    "object_index_funnel": {
                        "lemmas": list(_lemmas),
                        "parent_lemmas": list(_parent_lemmas),
                        "exact_ids": len(_dna_ids),
                        "parent_ids": len(_parent_ids),
                        "merged_ids": len(_merged),
                    },
                }
                if _merged:
                    _dna_rows = self.db.get_indexed_files_by_ids(
                        _merged,
                        include_processing_ready=True,
                        lightweight=True,
                    )
                    if _dna_rows:
                        _have = {int(r.get("id", 0) or 0) for r in candidates}
                        for rec in _dna_rows:
                            fid = int(rec.get("id", 0) or 0)
                            if fid <= 0:
                                continue
                            if fid in _exact:
                                rec["_concept_dna_hit"] = True
                            else:
                                rec["_concept_dna_parent"] = True
                            if fid in _have:
                                # Keep evidence flags on FTS/CLIP rows already in the pool.
                                for old in candidates:
                                    if int(old.get("id", 0) or 0) == fid:
                                        if rec.get("_concept_dna_hit"):
                                            old["_concept_dna_hit"] = True
                                        if rec.get("_concept_dna_parent"):
                                            old["_concept_dna_parent"] = True
                                        break
                                continue
                            candidates.append(rec)
                            _have.add(fid)
        except Exception as exc:
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                "object_index_funnel": {
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }

        try:
            from core.ovd_index import query_ovd_labels
            from core.object_index import ObjectIndexStore

            _ovd_labs = query_ovd_labels(effective_text)
            _obj_path = str(getattr(self.settings, "object_db_path", "") or "")
            if _ovd_labs and _obj_path and bool(getattr(self.settings, "open_vocab_object_enabled", False)):
                _ovd_store = ObjectIndexStore(_obj_path, readonly=True)
                _ovd_ids = _ovd_store.search_ovd_labels(_ovd_labs, limit=object_index_retrieval_limit)
            else:
                _ovd_ids = []
            if _ovd_ids:
                self._last_text_intel_meta = {
                    **(getattr(self, "_last_text_intel_meta", {}) or {}),
                    "open_vocab_index_hits": len(_ovd_ids),
                    "open_vocab_from_index": True,
                    "gdino_query_time": False,
                }
                _ovd_rows = self.db.get_indexed_files_by_ids(
                    _ovd_ids,
                    include_processing_ready=True,
                    lightweight=True,
                )
                if _ovd_rows:
                    _have = {int(r.get("id", 0) or 0) for r in candidates}
                    for rec in _ovd_rows:
                        fid = int(rec.get("id", 0) or 0)
                        if fid <= 0:
                            continue
                        rec["_open_vocab_hit"] = True
                        if fid in _have:
                            for old in candidates:
                                if int(old.get("id", 0) or 0) == fid:
                                    old["_open_vocab_hit"] = True
                                    break
                            continue
                        candidates.append(rec)
                        _have.add(fid)
        except Exception:
            pass

        # CLIP / OpenCLIP text-embedding hits must enter the candidate pool
        # for every non-brand text query. Composite and UVI used to be the
        # only unions; a single motif such as "leopar" then depended on FTS
        # filename/OCR windows and lost visually matching prints.
        if (
            _compound_v3
            and _v3_report is not None
            and any(
                str(getattr(layer, "status", "") or "") != "HAS_EVIDENCE"
                for layer in (_v3_report.layers or [])
            )
        ):
            for layer in _v3_report.layers or []:
                if str(getattr(layer, "status", "") or "") != "HAS_EVIDENCE":
                    continue
                if str(getattr(layer, "channel", "") or "") not in (
                    "pattern",
                    "texture_style",
                ):
                    continue
                q = str(getattr(layer, "token", "") or getattr(layer, "value", "") or "").strip()
                if len(q) < 3:
                    continue
                extra_clip, _meta = self._clip_text_hits(
                    q,
                    intent,
                    customer,
                    intel_on=intel_on,
                    threshold=float(threshold),
                    uq=None,
                )
                for fid, sc in (extra_clip or {}).items():
                    try:
                        i = int(fid)
                    except (TypeError, ValueError):
                        continue
                    clip_by_id[i] = max(float(clip_by_id.get(i, 0) or 0), float(sc or 0))

        if clip_by_id:
            extra_ids = _visual_candidate_ids(candidates, clip_by_id)
            if extra_ids:
                extra_rows = self.db.get_indexed_files_by_ids(
                    extra_ids[:800],
                    include_processing_ready=True,
                    lightweight=True,
                )
                if extra_rows:
                    candidates.extend(self._records_with_family_overrides(extra_rows))

        learned_pack: dict[str, Any] = {}
        try:
            from core.learned_concept_search import (
                collect_learned_hits,
                tag_candidate_records,
            )

            learned_pack = collect_learned_hits(
                str(getattr(self.settings, "db_path", "") or ""),
                effective_text,
                db=self.db,
                faiss_store=getattr(self, "faiss", None),
                customer_key=str(customer or "").strip(),
            ) or {}
            self._last_learned_concept = learned_pack
            extra_learned = [
                int(fid)
                for fid in (learned_pack.get("file_scores") or {})
            ]
            more = _visual_candidate_ids(candidates, extra_learned)
            if more:
                extra_rows = self.db.get_indexed_files_by_ids(
                    more[:800],
                    include_processing_ready=True,
                    lightweight=True,
                )
                if extra_rows:
                    candidates.extend(self._records_with_family_overrides(extra_rows))
            tag_candidate_records(candidates, learned_pack)
        except Exception:
            learned_pack = {}
            self._last_learned_concept = {}
        taught_conflict: set[int] = set()
        try:
            from core.learned_concept_search import conflicting_taught_ids

            taught_conflict = conflicting_taught_ids(
                str(getattr(self.settings, "db_path", "") or ""),
                effective_text,
            )
        except Exception:
            taught_conflict = set()

        def _brand_evidence(rec: dict[str, Any]) -> set[str]:
            # Belirli marka ve genel marka sorguları AYNI kanıt kümesini
            # kullanmalıdır. Böylece `marka`, `dior`, `amiri`, vb. sonuçlarını
            # eksik bırakamaz. Ortak çıkarıcı kategori_aliases/OCR/semantic/
            # filename gibi tüm desteklenen kanıtları birleştirir.
            from core.brand_evidence import extract_brand_evidence
            return extract_brand_evidence(rec, getattr(self.settings, "db_path", ""))


        seen: set[int] = set()
        stream_started = False
        stream_last_emit = 0.0
        stream_emit_interval = 0.06 if brand_search_mode else 0.18
        stream_min_results = 1 if brand_search_mode else max(4, min(8, int(limit or 8)))
        emit_limit = max(int(limit or 0), 1)
        zs_spec = None
        try:
            from core.search_evidence_gate import ZERO_SHOT_FLOOR, stamp_zero_shot, zero_shot_spec

            zs_spec = zero_shot_spec(effective_text)
        except Exception:
            zs_spec = None
        if brand_search_mode or limit <= 0:
            # Marka sorgusunda sonuç kümesi kanıt filtresiyle belirlendiği için
            # 50/32 gibi yapay bir alt küme oluşturma; tüm eşleşmeleri yayınla.
            emit_limit = 10**9

        def _emit_text_progress(force: bool = False) -> None:
            nonlocal stream_started, stream_last_emit
            if not result_callback or not results:
                return
            now = time.perf_counter()
            high = max((float(r.score or 0.0) for r in results), default=0.0) >= float(threshold)
            if not force and stream_started and now - stream_last_emit < stream_emit_interval:
                return
            if not force and not stream_started and not high and len(results) < stream_min_results:
                return
            snapshot = sorted(
                results,
                key=lambda r: (
                    -float(r.score or 0.0),
                    -float(r.breakdown.get("brand_alias_score", 0.0) or 0.0),
                    int(r.file_id),
                ),
            )
            result_callback(
                snapshot[:emit_limit] if emit_limit > 0 else snapshot,
                len(results),
                len(candidates),
            )
            stream_started = True
            stream_last_emit = now

        for rec in candidates:
            fid = int(rec["id"])
            if fid in seen:
                continue
            if taught_conflict and fid in taught_conflict:
                continue
            if not self._record_in_scope(rec, customer):
                continue
            brand_ev = _brand_evidence(rec) if brand_search_mode else set()
            brand_hit = False
            if brand_query_child:
                brand_hit = bool(
                    brand_query_canonical and brand_query_canonical in brand_ev
                )
            elif brand_query_parent:
                brand_hit = bool(brand_ev)
            clip_here = self._candidate_clip_scores(fid, intent, clip_by_id, evidence_plan)
            score, breakdown, reasons = text_search_score(
                effective_text,
                rec,
                semantic_enabled=semantic_enabled,
                terms=terms,
                hints=hints,
                parsed=parsed,
                cat_match=cat_match,
                q_norm=q_norm,
                terms_norm=terms_norm,
                brand_needles=brand_needles,
                clip_scores=clip_here,
            )
            if rec.get("_learned_concept_exact"):
                score = max(float(score or 0.0), 0.96)
                breakdown["learned_concept"] = 1.0
                if "Öğrenilmiş kavram" not in reasons:
                    reasons.insert(0, "Öğrenilmiş kavram")
            elif rec.get("_learned_concept_clip"):
                learned_sc = float(rec.get("_learned_concept_clip") or 0.0)
                score = max(float(score or 0.0), learned_sc)
                breakdown["learned_concept"] = learned_sc
                if "Öğrenilmiş kavram" not in reasons:
                    reasons.append("Öğrenilmiş kavram")
            # Çoklu kavram sorgusu: AND grubunun bütün zorunlu kavramları
            # desteklenmiyorsa aday sonuç değildir. Böylece "gül leopard"
            # aramasında yalnızca leopard olan desenler listeye sızmaz.
            qev = breakdown.get("query_evidence_report") or {}
            and_groups = (breakdown.get("search_evidence_plan") or {}).get("and_groups") or []
            if and_groups and not brand_search_mode and not _compound_v3:
                composite = str(qev.get("composite") or "")
                concepts = qev.get("concepts") or {}
                supported_any = any(
                    str((ev or {}).get("state") or "") == "SUPPORTED"
                    for ev in concepts.values()
                )
                zs_clip = float(clip_here.get("_similarity") or 0.0)
                zs_keep = bool(
                    zs_spec
                    and zs_clip >= float(ZERO_SHOT_FLOOR)
                    and not rec.get("_face_gender_match")
                    and self._zero_shot_clip_ok(zs_spec, clip_here, rec)
                )
                if (
                    composite != "PASS"
                    and not supported_any
                    and not rec.get("_concept_dna_hit")
                    and not zs_keep
                ):
                    continue
            if _compound_v3 and _v3_report is not None:
                _p_ids, _p_kind = primary_evidenced_ids(_v3_report)
                if fid in _p_ids and _p_kind == "brand":
                    score = max(float(score or 0.0), 0.90)
                    if "Kanıtlı marka kanalı" not in reasons:
                        reasons.append("Kanıtlı marka kanalı")
            if brand_hit:
                # Kanıtlı marka görsel lookalike'ın önüne geçer; kümeden silinmez.
                score = max(score, 0.90)
                breakdown["brand_alias_score"] = max(
                    float(breakdown.get("brand_alias_score", 0.0) or 0.0), 0.90
                )
                breakdown["brand_evidence_hit"] = 1.0
                if "Marka kanıtı" not in reasons:
                    reasons.append("Marka kanıtı")
            elif brand_search_mode:
                breakdown["brand_evidence_hit"] = 0.0
                # Marka kanıtı yoksa yalnızca görsel lookalike kalır; logo/alias sızıntısı olmaz.
                clip_sim = float(clip_here.get("_similarity") or 0.0)
                score = min(0.74, clip_sim) if clip_sim > 0.0 else 0.0
                breakdown["brand_alias_score"] = min(
                    float(breakdown.get("brand_alias_score", 0.0) or 0.0), 0.50
                )
            if score <= 0.0 and not rec.get("_concept_dna_hit") and not rec.get("_concept_dna_parent") and not rec.get("_open_vocab_hit") and not rec.get("_learned_concept_exact") and not rec.get("_learned_concept_clip"):
                zs_clip = float(clip_here.get("_similarity") or 0.0)
                if not (
                    zs_spec
                    and zs_clip >= float(ZERO_SHOT_FLOOR)
                    and not rec.get("_face_gender_match")
                    and self._zero_shot_clip_ok(zs_spec, clip_here, rec)
                ):
                    continue
            if rec.get("_concept_dna_hit"):
                # Keep exact detector rows in ranking even when CLIP/FTS is weak.
                score = max(float(score or 0.0), 0.62)
                breakdown["object_index"] = 1.0
            elif score <= 0.0 and rec.get("_concept_dna_parent"):
                score = 0.34
                breakdown["object_index_parent"] = 0.5
            elif zs_spec and not rec.get("_face_gender_match"):
                zs_clip = float(clip_here.get("_similarity") or 0.0)
                if zs_clip >= float(ZERO_SHOT_FLOOR) and self._zero_shot_clip_ok(zs_spec, clip_here, rec):
                    score = max(float(score or 0.0), zs_clip)
                    breakdown["zero_shot_visual"] = zs_clip
            seen.add(fid)
            result = self._to_result(rec, score, breakdown)
            _mcp = str(rec.get("manual_category_path") or "").strip()
            _cp = str(rec.get("category_path") or "").strip()
            if not _mcp or not _cp:
                _tm = rec.get("texture_map") or {}
                if isinstance(_tm, dict):
                    _mcp = _mcp or str(_tm.get("manual_category_path") or "").strip()
                    _cp = _cp or str(_tm.get("category_path") or "").strip()
            if _mcp:
                result.debug["manual_category_path"] = _mcp
            if _cp:
                result.debug["category_path"] = _cp
            if rec.get("_learned_concept_exact") or rec.get("_learned_concept_clip"):
                result.debug["learned_concept"] = True
                result.debug["learned_concept_exact"] = bool(rec.get("_learned_concept_exact"))
                if rec.get("_learned_concept_exact"):
                    result.debug["user_taught_positive"] = True
            if rec.get("_concept_dna_hit"):
                result.debug["object_index_hit"] = True
                result.debug["visual_concept_detector"] = True
            if rec.get("_open_vocab_hit"):
                result.debug["open_vocab_object"] = True
                result.debug["evidence_type"] = "OPEN_VOCAB_OBJECT"
                result.debug["ovd_source"] = "object_index"
                result.debug["gdino_query_time"] = False
                if not rec.get("_concept_dna_hit"):
                    score = max(float(score or 0.0), 0.62)
                    breakdown["open_vocab_object"] = 1.0
                    result.score = score
                    result.score_percent = round(float(score) * 100, 1)
            if rec.get("_concept_dna_parent"):
                result.debug["object_index_parent_hit"] = True
            if _compound_v3 and _route is not None:
                from core.concept_channel_retrieval import evidence_for_record

                def _motif_boxes_fn(row: dict[str, Any]) -> list:
                    from core.textile_motif_v4 import detections_from_store, open_motif_store

                    store = open_motif_store(
                        getattr(self.settings, "textile_motif_db_path", "") or ""
                    )
                    try:
                        fid = int(row.get("id") or 0)
                    except (TypeError, ValueError):
                        return []
                    return detections_from_store(store, fid) if fid else []

                _ev = evidence_for_record(
                    rec,
                    _route,
                    _v3_report,
                    db_path=getattr(self.settings, "db_path", ""),
                    motif_boxes_fn=_motif_boxes_fn,
                )
                for _k, _v in _ev.items():
                    result.debug[_k] = _v
                    breakdown[_k] = _v
            if bool(rec.get("_face_gender_match")):
                result.debug["face_gender_match"] = True
                result.debug["face_gender_value"] = (face_query_intent.value if face_query_intent else "")
            # Açıklamayı hafif tut
            result.text_match_reason = " | ".join(reasons[:3])
            result.match_explanations = [str(x) for x in reasons[:4] if x]
            if result.match_explanations:
                result.text_match_reason = " · ".join(
                    f"✓ {x}" for x in result.match_explanations[:4]
                )
            if was_corrected:
                result.debug["spell_corrected"] = corrected_text
                result.debug["spell_original"] = text
            result.text_score_breakdown = breakdown
            result.breakdown = {**result.breakdown, **breakdown}
            pf = rec.get("pattern_family") or ""
            _vpq = False
            try:
                from core.visual_pattern_query import is_visual_pattern_query as _is_vpq

                _vpq = _is_vpq(effective_text)
            except Exception:
                _vpq = False
            if hints.get("pattern_family") and pf and pf != hints["pattern_family"]:
                if _vpq or breakdown.get("filename_score", 0) < 0.7:
                    # UNION fallback: missing concept (çiçek/gold) must not
                    # 0.4x a proven leopard/brand/dantel hit below threshold.
                    if not _union_fb:
                        result.score *= 0.4
                        result.score_percent = round(result.score * 100, 1)
            reliable_pf = pf if breakdown.get("family_reliable", 1.0) >= 0.5 else ""
            if not reliable_pf and breakdown.get("filename_score", 0.0) >= 0.70 and not _vpq:
                reliable_pf = hints.get("pattern_family", "")
            result.pattern_family = reliable_pf or "unknown"
            self._apply_family_overrides(result)
            result.category = self._text_category(
                result.score, result.pattern_family, hints
            )
            result.cluster_group = self._text_cluster_group(
                result.pattern_family, hints
            )
            qev_dbg = breakdown.get("query_evidence_report") or {}
            clip_dbg = float(clip_here.get("_similarity") or 0.0)
            if intent is not None and getattr(intent, "motif", ""):
                clip_dbg = max(
                    clip_dbg,
                    float(clip_here.get(str(intent.motif), 0.0) or 0.0),
                )
            _tm = rec.get("texture_map") or {}
            if not isinstance(_tm, dict):
                _tm = {}
            result.debug = {
                **result.debug,
                "text_mode": True,
                "matched_reason": result.text_match_reason,
                "threshold_passed": result.score >= threshold,
                "parsed_query": parsed.to_dict() if parsed.has_components else {},
                "nl_query_v2": getattr(self, "_last_nl_analysis", None) or {},
                "query_evidence_report": qev_dbg,
                "search_evidence_plan": breakdown.get("search_evidence_plan") or {},
                "visual_verdict": qev_dbg.get("visual_verdict") or "",
                "visual_grade": qev_dbg.get("visual_grade") or "",
                "clip_object_prompt": bool(getattr(self, "_last_clip_object_prompt", False)),
                "clip_score": round(clip_dbg, 4),
                "texture_map": _tm,
                "clip_as_detector": False,
            }
            if (
                zs_spec
                and not rec.get("_concept_dna_hit")
                and not rec.get("_face_gender_match")
                and not result.debug.get("object_index_hit")
                and not result.debug.get("face_gender_match")
            ):
                clip_sim = max(
                    float(clip_dbg or 0.0),
                    float(clip_here.get("_similarity") or 0.0),
                )
                rivals = {
                    str(k): float(v or 0.0)
                    for k, v in clip_here.items()
                    if k != "_similarity"
                }
                if rivals:
                    result.debug["zs_clip_rivals"] = rivals
                if clip_sim >= ZERO_SHOT_FLOOR and self._zero_shot_clip_ok(zs_spec, clip_here, rec):
                    stamp_zero_shot(result, zs_spec, clip_sim)
                    if float(result.score or 0.0) < clip_sim:
                        result.score = clip_sim
                        result.score_percent = round(clip_sim * 100.0, 1)
            results.append(result)
            _emit_text_progress()

        _emit_text_progress(force=True)

        # ── Yeni sıralama (Name Search v2 + Semantic Intel) ───────────
        # Human/gender queries are NOT textile semantic-intel queries.
        # The generic SemanticPatternIntel gate was the main regression here:
        # it could correctly retrieve a woman/man with CLIP, then discard the
        # candidate because the record did not contain textile evidence.
        # Build the human candidate union directly and let UVI perform the
        # person/gender-specific ranking below.
        human_semantic_mode = bool(
            face_query_intent is not None and face_query_intent.kind == "gender"
        ) or bool(
            uq is not None and getattr(uq, "node_id", "") in {"person", "female_person", "male_person"}
        )
        try:
            from core.search_evidence_gate import classify_accuracy_lane as _acc_lane

            object_lane_query = _acc_lane(effective_text) == "object"
        except Exception:
            object_lane_query = False
        object_lane_query = object_lane_query or any(
            bool((getattr(r, "debug", {}) or {}).get("object_index_hit"))
            and not (getattr(r, "debug", {}) or {}).get("object_index_parent_hit")
            for r in results
        )
        if human_semantic_mode and not brand_search_mode:
            seen_human = {int(r.file_id) for r in results}
            gender_only_intent = (
                face_query_intent is not None
                and getattr(face_query_intent, "kind", "") == "gender"
            )
            # Bare "kadın"/"erkek": do not flood with generic CLIP (kumaş %95).
            # "kadın yüzü" is kind=pattern and still uses CLIP inject below.
            if not gender_only_intent:
                for fid, clip in clip_by_id.items():
                    fid = int(fid)
                    if fid in seen_human or float(clip or 0.0) <= 0.0:
                        continue
                    recs = self.db.get_indexed_files_by_ids(
                        [fid], include_processing_ready=True, lightweight=True
                    )
                    if not recs or not self._record_in_scope(recs[0], customer):
                        continue
                    sc = min(1.0, max(0.0, float(clip)))
                    r = self._to_result(recs[0], sc, {"clip": sc, "text": 0.0})
                    r.debug = {**r.debug, "text_mode": True, "human_semantic_only": True,
                               "clip_only": True, "clip_score": round(sc, 4)}
                    results.append(r)
                    seen_human.add(fid)

            # Persistent face-gender hits are authoritative and must survive
            # even when filename/FTS/CLIP has no evidence for the word itself.
            for fid in face_ids:
                fid = int(fid)
                if fid in seen_human:
                    continue
                rec = self.db.get_file_by_id(fid)
                if not rec or not self._record_in_scope(rec, customer):
                    continue
                r = self._to_result(rec, 0.92, {"face": 1.0, "text": 0.0})
                r.debug = {
                    **r.debug,
                    "text_mode": True,
                    "human_semantic_only": True,
                    "face_gender_match": True,
                    "face_gender_value": face_query_intent.value if face_query_intent else "",
                }
                results.append(r)
                seen_human.add(fid)
            self._last_text_intel_meta = {
                "semantic_pattern_intel": False,
                "human_semantic_mode": True,
                **clip_meta,
            }
        elif (
            intel_on
            and not object_lane_query
            and intent is not None
            and (not brand_search_mode or _compound_v3)
        ):
            results = self._merge_semantic_intel(
                text, intent, results, clip_by_id, customer, plan=evidence_plan
            )
            self._last_text_intel_meta = {
                **getattr(self, "_last_text_intel_meta", {}),
                **clip_meta,
            }
        else:
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                **clip_meta,
                "semantic_pattern_intel": False,
                **({"object_lane_skip_textile_intel": True} if object_lane_query else {}),
            }
        results = self._apply_text_query_teach(text, results, customer)
        try:
            from core.learned_concept_search import apply_learned_to_results

            pack = getattr(self, "_last_learned_concept", None) or learned_pack
            extra_rows = []
            have = {int(getattr(r, "file_id", 0) or 0) for r in results}
            missing = [
                fid
                for fid in (pack.get("file_scores") or {})
                if int(fid) not in have
            ]
            if missing:
                recs = self.db.get_indexed_files_by_ids(
                    missing[:800], include_processing_ready=True, lightweight=True
                )
                recs = self._records_with_family_overrides(recs)
                scores = pack.get("file_scores") or {}
                for rec in recs:
                    fid = int(rec.get("id") or 0)
                    extra_rows.append(
                        self._to_result(
                            rec,
                            float(scores.get(fid, 0.96) or 0.96),
                            {"learned_concept": 1.0, "text": 0.0},
                        )
                    )
            results = apply_learned_to_results(results, pack, extra_rows)
        except Exception:
            pass
        if (
            v2q is not None
            and not object_lane_query
            and (not brand_search_mode or _compound_v3)
            and not human_semantic_mode
        ):
            from core.pattern_intelligence_v2 import rank_v2_results

            concept_scores = dict(getattr(self, "_last_v2_concepts", {}) or {})
            for fid, rival in (getattr(self, "_last_clip_rivals", {}) or {}).items():
                concept_scores.setdefault(int(fid), {}).update(rival)
            results, v2_stats = rank_v2_results(
                v2q,
                results,
                clip_by_id=concept_scores,
                rep_by_id=getattr(self, "_last_v2_rep", {}) or {},
            )
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                "pattern_intelligence_v2": v2_stats,
                "pattern_query_v2": v2q.to_dict(),
            }
        if uq is not None and (not brand_search_mode or _compound_v3):
            from core.universal_visual_intel import apply_universal_ranking

            clip_active = bool(
                self.extractor.ai_available
                and getattr(self.extractor, "_clip_model", None) is not None
                and self.faiss.available
                and self.faiss.clip_count > 0
            )
            concept_scores = dict(getattr(self, "_last_v2_concepts", {}) or {})
            for fid, rival in (getattr(self, "_last_clip_rivals", {}) or {}).items():
                concept_scores.setdefault(int(fid), {}).update(rival)
            results, uvi_stats = apply_universal_ranking(
                uq,
                results,
                clip_by_id=concept_scores,
                clip_active=clip_active,
            )
            _funnel = dict(
                ((getattr(self, "_last_text_intel_meta", {}) or {}).get("object_index_funnel") or {})
            )
            if _funnel:
                _funnel["after_uvi"] = sum(
                    1
                    for r in results
                    if (getattr(r, "debug", {}) or {}).get("object_index_hit")
                )
                _funnel["uvi_dropped_unspecific"] = int((uvi_stats or {}).get("dropped_unspecific") or 0)
                _funnel["uvi_dropped_mismatch"] = int((uvi_stats or {}).get("dropped_mismatch") or 0)
                self._last_text_intel_meta = {
                    **(getattr(self, "_last_text_intel_meta", {}) or {}),
                    "object_index_funnel": _funnel,
                }
            # Canonicalize the human-result contract at the SearchEngine
            # boundary.  This is intentionally done AFTER UVI ranking so a
            # result cannot lose its low-but-valid gender evidence when the
            # generic score remains below the textile threshold.
            if human_semantic_mode:
                human_node = str(getattr(uq, "node_id", "") or "")
                gender_only_intent = (
                    face_query_intent is not None
                    and getattr(face_query_intent, "kind", "") == "gender"
                )
                for r in results:
                    dbg = dict(getattr(r, "debug", {}) or {})
                    face_match = bool(dbg.get("face_gender_match"))
                    human_score = float(
                        dbg.get(
                            "gender_visual_score",
                            dbg.get("human_semantic_score", 0.0),
                        )
                        or 0.0
                    )
                    if not human_score and human_node:
                        cmap = concept_scores.get(int(r.file_id), {}) or {}
                        human_score = float(cmap.get(human_node, 0.0) or 0.0)
                    if gender_only_intent and not face_match:
                        # Keep taught / manual-category evidence; do not demote.
                        try:
                            from core.teach_search_wire import (
                                keep_taught_evidence_on_gender,
                            )

                            _qtoks = [
                                t
                                for t in str(effective_text or text or "").split()
                                if t.strip()
                            ]
                            _keep_taught = keep_taught_evidence_on_gender(
                                dbg, _qtoks
                            )
                        except Exception:
                            _keep_taught = False
                        if not _keep_taught:
                            # Do not re-copy OpenCLIP/concept scores onto gender
                            # after UVI already refused generic CLIP as evidence.
                            dbg.pop("gender_visual_score", None)
                            dbg["human_semantic_score"] = 0.0
                            r.debug = dbg
                            continue
                    if face_match or human_score >= 0.18:
                        dbg["human_semantic_mode"] = True
                        dbg["human_semantic_only"] = True
                        dbg["human_semantic_score"] = round(human_score, 4)
                        if human_score >= 0.18:
                            dbg["gender_visual_score"] = round(human_score, 4)
                        r.debug = dbg
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                "universal_visual_intel": uvi_stats,
                "universal_query": uq.to_dict(),
                "human_semantic_mode": bool(human_semantic_mode),
            }

        # Stages 2A–2E via unified intelligence chain (no new AI engine).
        # Learned authority already applied above; user_protected layers skip demote.
        if not human_semantic_mode and not brand_search_mode:
            try:
                from core.search_intelligence_chain import (
                    apply_search_intelligence_chain,
                )

                _nl = getattr(self, "_last_nl_analysis", None)
                _analysis = _nl if isinstance(_nl, dict) and _nl.get("raw") else None
                # If search used stripped search_text, prefer stashed full NL analysis.
                results = apply_search_intelligence_chain(
                    results,
                    effective_text or text,
                    index_db=str(getattr(self.settings, "db_path", "") or ""),
                    analysis=_analysis,
                    customer_registry=self._customer_registry_cached(),
                    customer=str(customer or "").strip(),
                )
            except Exception:
                # Fallback: legacy 2B then 2A if chain import/runtime fails.
                try:
                    from core.query_attribute_intel import apply_query_attribute_intel

                    results = apply_query_attribute_intel(
                        results,
                        effective_text or text,
                    )
                except Exception:
                    pass
                try:
                    from core.object_pattern_gate import apply_object_pattern_gate

                    results = apply_object_pattern_gate(
                        results,
                        effective_text or text,
                    )
                except Exception:
                    pass

        try:
            from core.intent_evidence_routing import apply_intent_evidence_routing

            results = apply_intent_evidence_routing(
                results,
                effective_text or text,
                has_image=False,
                db_path=str(getattr(self.settings, "db_path", "") or ""),
            )
        except Exception:
            pass

        _SEM_RANK = {
            "Exact": 0,
            "Very Similar": 1,
            "Variant": 2,
            "Same Family": 3,
            "Semantic Similar": 4,
        }

        def _sort_key(r: SearchResult) -> tuple:
            bd = r.breakdown
            brand_hit = 0
            if brand_search_mode:
                brand_hit = 1 if float(bd.get("brand_evidence_hit", 0.0) or 0.0) >= 0.5 or float(bd.get("brand_alias_score", 0.0) or 0.0) >= 0.85 else 0
            visual = float((r.debug or {}).get("clip_score") or 0.0)
            if brand_search_mode and not (intel_on and _compound_v3):
                # Marka kanıtı > görsel benzerlik. Kovaların içinde CLIP/skor.
                return (
                    -brand_hit,
                    -visual,
                    -float(bd.get("brand_alias_score", 0.0) or 0.0),
                    -float(bd.get("ocr_score", 0.0) or 0.0),
                    -float(r.score or 0.0),
                    int(r.file_id),
                )
            fname = bd.get("filename_score", 0.0)
            ocr   = bd.get("ocr_score", 0.0)
            brand = bd.get("brand_alias_score", 0.0)
            fam   = bd.get("family_score", 0.0)
            tex   = bd.get("texture_score", 0.0)
            nl    = bd.get("nl_score", 0.0)
            dna   = float(bd.get("dna_score") or bd.get("pattern_dna_score") or 0.0)
            _vp_sort = False
            try:
                from core.visual_pattern_query import is_visual_pattern_query as _is_vp_sort

                _vp_sort = (not brand_search_mode) and _is_vp_sort(text)
            except Exception:
                _vp_sort = False
            if intel_on:
                st = _SEM_RANK.get(str(r.debug.get("semantic_tier") or ""), 5)
                raw_bucket = r.debug.get("v2_bucket")
                bucket = 5 if raw_bucket is None else int(raw_bucket)
                raw_uvi = r.debug.get("uvi_rank")
                uvi = 5 if raw_uvi is None else int(raw_uvi)
                margin = float(r.debug.get("clip_margin") or 0.0)
                visual = float(r.debug.get("clip_score") or 0.0)
                gender_score = float(
                    (r.debug or {}).get("gender_visual_score")
                    or (r.debug or {}).get("gender_score")
                    or 0.0
                )
                cr = r.debug.get("composite_ranker") or {}
                vp_neg = 1 if (r.debug or {}).get("pattern_visual_negative") else 0
                if _vp_sort:
                    return (
                        -brand_hit, vp_neg, -dna, -fam, uvi, bucket, st,
                        -margin, -visual, -float(r.score or 0.0), fname,
                    )
                if cr.get("query_is_composite"):
                    return (-brand_hit, -float(r.score or 0.0), bucket, uvi, st, -margin, -visual, -fname)
                if human_semantic_mode:
                    return (-brand_hit, -gender_score, uvi, bucket, st, -margin, -visual, -r.score, -fname)
                return (-brand_hit, uvi, bucket, st, -margin, -visual, -r.score, -fname)
            if _vp_sort:
                vp_neg = 1 if (r.debug or {}).get("pattern_visual_negative") else 0
                dino = float((r.debug or {}).get("dino_score") or bd.get("dino") or 0.0)
                clip = float((r.debug or {}).get("clip_score") or 0.0)
                return (vp_neg, -dna, -fam, -tex, -dino, -clip, -r.score, fname, ocr)
            if fname >= 0.85:
                tier = 0
            elif ocr >= 0.90:
                tier = 1
            elif brand >= 0.85:
                tier = 2
            elif fam >= 0.75:
                tier = 3
            else:
                tier = 4
            return (-tier, -r.score, -fam, -tex, -nl)

        results.sort(key=_sort_key)
        if intel_on and (not brand_search_mode or _compound_v3) and not human_semantic_mode:
            from core.query_evidence import visual_verdict_text
            from core.unified_relevance_ranker import _is_leaf_query

            for r in results:
                qev = (r.debug or {}).get("query_evidence_report") or {}
                plan = (r.debug or {}).get("search_evidence_plan") or {}
                if qev:
                    r.debug["visual_verdict"] = visual_verdict_text(qev)
                    r.debug["visual_grade"] = qev.get("visual_grade") or "visual_unknown"
                if (r.debug or {}).get("query_evidence_applied"):
                    r.score_percent = round(float(r.score or 0.0) * 100, 1)
                    continue
                delta = float(qev.get("delta") or 0.0)
                r.score = min(1.0, max(0.0, float(r.score or 0.0) + delta))
                objs = [c for c in (plan.get("objects") or []) if c.get("required", True)]
                concepts = qev.get("concepts") or {}
                if objs and not any(
                    str((concepts.get(str(c.get("concept_id") or "")) or {}).get("state"))
                    == "SUPPORTED"
                    for c in objs
                ):
                    if not _is_leaf_query(r.debug or {}):
                        r.score = min(float(r.score), 0.48)
                r.score_percent = round(r.score * 100, 1)
                r.debug["query_evidence_applied"] = True
            results.sort(key=_sort_key)

        # Faz 3: composite sıralama EN SON uygulanır.
        # Önceki sürümde composite ranker UVI'dan önce çalışıyordu; ardından
        # query-evidence skoru tekrar değiştiği için "yalnız leopard" gibi bir
        # kayıt yeniden öne çıkabiliyordu. Burada tüm kanıtlar kesinleştikten
        # sonra tek kez uygulanır. Tekli aramalarda hiçbir etkisi yoktur.
        composite_query = False
        if (not brand_search_mode or _compound_v3) and not human_semantic_mode:
            from core.composite_ranker import apply_composite_ranking

            composite_query = bool(
                (v2q is not None and getattr(v2q, "composition", "") == "composite")
                or len(
                    [c for c in getattr(evidence_plan, "objects", []) if getattr(c, "required", True)]
                ) >= 2
            )
            # v2'nin `required` listesi her zaman tüm sorgu kavramlarını
            # taşımaz (ör. "gül leopard" içinde yalnızca leopard kalabilir).
            # Composite sıralamanın AND mantığını kaybetmemesi için v2 +
            # Query Evidence planındaki tüm zorunlu kavramları birleştir.
            required = list(getattr(v2q, "required", []) or []) if v2q is not None else []
            plan_required = [
                str(c.concept_id)
                for c in (
                    list(getattr(evidence_plan, "objects", []) or [])
                    + list(getattr(evidence_plan, "attributes", []) or [])
                )
                if getattr(c, "required", True)
            ]
            for cid in plan_required:
                if cid and cid not in required:
                    required.append(cid)
            results, composite_stats = apply_composite_ranking(
                results,
                query_is_composite=composite_query,
                required=required,
                required_count=len(required) if required else None,
            )
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                "composite_ranker": composite_stats,
            }

        if not brand_search_mode or _compound_v3:
            from core.entity_intelligence import annotate_search_results
            from core.unified_relevance_ranker import apply_unified_text_ranking

            annotate_search_results(
                results,
                text,
                object_db_path=str(getattr(self.settings, "object_db_path", "") or ""),
            )

            visual_retrieval = bool(clip_meta.get("clip_attempted") or clip_by_id)
            if not composite_query:
                results = apply_unified_text_ranking(
                    results,
                    human_query=human_semantic_mode,
                    visual_retrieval=visual_retrieval,
                    query_text=text,
                )
                self._last_text_intel_meta = {
                    **(getattr(self, "_last_text_intel_meta", {}) or {}),
                    "unified_relevance_ranking": True,
                    "visual_retrieval": visual_retrieval,
                }

        try:
            from core.fusion_query_v2 import apply_fusion_query_v2
            from core.index_freeze import IndexFrozenWriteBlocked

            results = apply_fusion_query_v2(
                results,
                text,
                object_db_path=str(getattr(self.settings, "object_db_path", "") or ""),
            )
            from core.fusion_query_v2 import last_evidence_gate_meta

            gate_meta = last_evidence_gate_meta()
            _funnel = dict(
                ((getattr(self, "_last_text_intel_meta", {}) or {}).get("object_index_funnel") or {})
            )
            if _funnel:
                _funnel["after_fusion_gate"] = int(gate_meta.get("evidence_count") or 0)
                _funnel["accuracy_lane"] = str(gate_meta.get("accuracy_lane") or "")
            self._last_text_intel_meta = {
                **(getattr(self, "_last_text_intel_meta", {}) or {}),
                "fusion_query_v2": True,
                **{k: gate_meta[k] for k in (
                    "accuracy_lane", "empty_state", "clip_as_detector",
                    "similarity_is_not_truth", "evidence_count",
                ) if k in gate_meta},
                **({"object_index_funnel": _funnel} if _funnel else {}),
            }
        except IndexFrozenWriteBlocked:
            raise
        except Exception:
            pass

        # Son semantic/rerank aşamasından sonra UI'ye bir kez daha nihai sıra
        # gönder. Böylece yeni yüksek puanlı kayıtlar geldikçe liste gerçekten
        # yeniden sıralanır; marka sorgusunda ise bütün kanıtlı kayıtlar korunur.
        if result_callback and results:
            keep_detector = any(
                (getattr(r, "debug", {}) or {}).get("object_index_hit")
                and not (getattr(r, "debug", {}) or {}).get("object_index_parent_hit")
                for r in results
            )
            final_snapshot = (
                results
                if brand_search_mode or limit <= 0 or keep_detector
                else results[:limit]
            )
            result_callback(final_snapshot, len(results), len(candidates))
        try:
            from core.search_memory import overlay_adjustments, overlay_wrong_ids

            qkeys = [text]
            uq = locals().get("user_query_text")
            if uq and uq not in qkeys:
                qkeys.append(uq)
            adj: dict[int, float] = {}
            wrong: set[int] = set()
            dbp = getattr(self.settings, "db_path", "")
            for qk in qkeys:
                for fid, delta in overlay_adjustments(
                    dbp, qk, customer=str(customer or "")
                ).items():
                    adj[fid] = adj.get(fid, 0.0) + float(delta)
                wrong |= overlay_wrong_ids(dbp, qk, customer=str(customer or ""))
            if adj or wrong:
                kept = []
                for r in results:
                    fid = int(r.file_id)
                    if fid in wrong:
                        continue
                    if fid in adj:
                        r.score = max(0.0, min(1.0, float(r.score or 0.0) + adj[fid]))
                        r.score_percent = round(float(r.score) * 100, 1)
                        r.debug = {
                            **(r.debug or {}),
                            "memory_overlay_delta": round(adj[fid], 4),
                        }
                    kept.append(r)
                results = kept
                results.sort(key=_sort_key)
        except Exception:
            pass
        if _compound_v3 and _v3_report is not None:
            apply_compound_evidence_floor(
                results, _v3_report, threshold=float(threshold)
            )
            primary_ids, kind = primary_evidenced_ids(_v3_report)
            policy = str(getattr(_v3_report, "policy", "") or "")
            if kind == "other" and policy != "AND":
                note = "Kanıtlı kanal (eksik kavram yok sayıldı)"
                for r in results:
                    try:
                        fid = int(r.file_id)
                    except (TypeError, ValueError):
                        continue
                    pf = str(getattr(r, "pattern_family", "") or "")
                    has_pat = bool((r.debug or {}).get("pattern_evidence"))
                    if fid not in primary_ids and pf != "animal_print" and not has_pat:
                        continue
                    sc = max(float(r.score or 0.0), 0.62)
                    r.score = sc
                    r.score_percent = round(sc * 100, 1)
                    notes = list(r.match_explanations or [])
                    if note not in notes:
                        notes.append(note)
                        r.match_explanations = notes
                    if isinstance(r.debug, dict):
                        r.debug["threshold_passed"] = sc >= float(threshold)
            have = {int(r.file_id) for r in results}
            missing = [
                i for i in (_v3_report.candidate_ids or [])
                if i in primary_ids and i not in have
            ]
            missing += [i for i in primary_ids if i not in have and i not in missing]
            # Do not dump a 400-wide FTS window. Brand pools are small; pattern
            # UNION floors survivors instead of rehydrating the whole layer.
            if kind != "brand" and len(missing) > 80:
                missing = []
            if missing:
                rows = self.db.get_indexed_files_by_ids(
                    missing,
                    include_processing_ready=True,
                    include_pending=True,
                    lightweight=True,
                )
                floor = 0.90 if kind == "brand" else 0.62
                note = (
                    "Kanıtlı marka kanalı"
                    if kind == "brand"
                    else "Kanıtlı kanal (eksik kavram yok sayıldı)"
                )
                for rec in rows or []:
                    if not self._record_in_scope(rec, customer):
                        continue
                    rec = (self._records_with_family_overrides([rec]) or [rec])[0]
                    r = self._to_result(rec, floor, {"compound_union_keep": True})
                    r.match_explanations = [note]
                    r.text_match_reason = note
                    r.debug = {
                        **(r.debug or {}),
                        "threshold_passed": True,
                        "compound_union_keep": True,
                    }
                    results.append(r)
                apply_compound_evidence_floor(
                    results, _v3_report, threshold=float(threshold)
                )
            results.sort(key=_sort_key)
        zs_spec = locals().get("zs_spec")
        if zs_spec:
            def _zs_tier(row: Any) -> int:
                d = getattr(row, "debug", {}) or {}
                if d.get("object_index_hit") and not d.get("object_index_parent_hit"):
                    return 0
                if (
                    str(d.get("evidence_type") or "") == "OPEN_VOCAB_OBJECT"
                    or d.get("open_vocab_object")
                ) and not d.get("ovd_rejected"):
                    return 1
                if str(d.get("evidence_type") or "") == "ZERO_SHOT_VISUAL" or d.get("zero_shot_visual"):
                    return 2
                if d.get("object_index_parent_hit"):
                    return 3
                return 4

            results.sort(
                key=lambda row: (_zs_tier(row), -float(getattr(row, "score", 0) or 0))
            )
        keep_detector = any(
            (getattr(r, "debug", {}) or {}).get("object_index_hit")
            and not (getattr(r, "debug", {}) or {}).get("object_index_parent_hit")
            for r in results
        )
        keep_zs = bool(zs_spec) and any(
            (getattr(r, "debug", {}) or {}).get("zero_shot_visual")
            for r in results
        )
        keep_ovd = any(
            (getattr(r, "debug", {}) or {}).get("open_vocab_object")
            and not (getattr(r, "debug", {}) or {}).get("ovd_rejected")
            for r in results
        )
        results = self._promote_learned_results(results, customer)
        try:
            from core.learned_concept_search import filter_taught_conflicts

            results = filter_taught_conflicts(
                results,
                str(getattr(self.settings, "db_path", "") or ""),
                effective_text,
            )
        except Exception:
            pass
        keep_learned = any(
            (getattr(r, "debug", {}) or {}).get("learned_concept_exact")
            or (getattr(r, "debug", {}) or {}).get("learned_concept")
            for r in results
        )
        return results if brand_search_mode or limit <= 0 or keep_detector or keep_zs or keep_ovd or keep_learned else results[:limit]

    @staticmethod
    def _text_category(score: float, pattern_family: str, hints: dict) -> str:
        if score >= 0.90:
            return CATEGORY_EXACT
        if score >= 0.75:
            return CATEGORY_SIMILAR
        if pattern_family in ("animal_print", "floral"):
            return (
                CATEGORY_ANIMAL_PRINT
                if pattern_family == "animal_print"
                else CATEGORY_TEXTILE_TEXTURE
            )
        return CATEGORY_STYLE

    @staticmethod
    def _text_cluster_group(pattern_family: str, hints: dict) -> str:
        from core.texture_profile import (
            CLUSTER_COLOR,
            CLUSTER_DISTANT,
            CLUSTER_RELATED,
            CLUSTER_SAME_CLOSE,
            CLUSTER_SAME_STYLE,
        )

        hint_pf = hints.get("pattern_family", "")
        hint_pt = hints.get("pattern_type", "")
        pf = pattern_family or hint_pf
        if hint_pf == "animal_print" or pf == "animal_print":
            if hint_pt and pattern_family == "animal_print":
                return CLUSTER_SAME_CLOSE
            return CLUSTER_RELATED
        if hint_pf == "floral" or pf == "floral":
            if hint_pt in ("small_floral", "ditsy_floral"):
                return CLUSTER_SAME_STYLE
            return CLUSTER_SAME_CLOSE
        if hint_pf == "marble_abstract" or pf in (
            "marble_abstract",
            "marble",
            "abstract",
        ):
            return CLUSTER_SAME_CLOSE
        if hint_pf == "stripe" or pf == "stripe":
            return CLUSTER_SAME_CLOSE
        if hint_pf == "plaid_check" or pf == "plaid_check":
            return CLUSTER_SAME_CLOSE
        return CLUSTER_DISTANT

    @staticmethod
    def _text_match_score(query: str, rec: dict[str, Any]) -> float:
        return text_match_score(query, rec)

    def search_similar_to_file(
        self,
        file_id: int,
        limit: int = 50,
        customer: str = "",
    ) -> list[SearchResult]:
        rec = self.db.get_file_by_id(file_id)
        if not rec:
            return []
        # Indexed candidate: önce yerel feature/thumbnail cache kullan.
        # NAS/orijinal dosyayı sırf "benzerlerini bul" için tekrar açma.
        feature_preview = str(rec.get("feature_preview_path") or "").strip()
        if feature_preview and os.path.isfile(feature_preview):
            return self.search_by_image(
                feature_preview, limit=limit, customer=customer
            )
        thumb = str(rec.get("thumbnail_path") or "").strip()
        if thumb and os.path.isfile(thumb):
            return self.search_by_image(thumb, limit=limit, customer=customer)
        # Cache yoksa yalnızca yerel kaynakta son çare olarak orijinali kullan.
        path = str(rec.get("path") or "").strip()
        if path and os.path.isfile(path):
            return self.search_by_image(path, limit=limit, customer=customer)
        return []

    def _adaptive_expand_scored(
        self,
        initial: list[SearchResult],
        *,
        features: ExtractedFeatures,
        indexed_pool: list[dict[str, Any]],
        customer: str,
        query_path: str,
        query_partial_hash: str,
        text_query: str,
        crop_search: bool,
        query_profile: TextureProfile,
        feedback_adj: dict[int, float],
        pattern_group_ids: set[int],
        wrong_ids: set[int],
        result_callback: Callable[[list[SearchResult], int, int], None] | None,
        threshold: float,
        protected_ids: set[int],
    ) -> list[SearchResult]:
        """İlk 800 sonrası FAISS batch genişletme — eski sonuçlar yeniden skorlanmaz."""
        lat = self._latency
        merged = list(initial or [])
        by_id: dict[int, SearchResult] = {int(r.file_id): r for r in merged}
        pool_ids = {int(r["id"]) for r in indexed_pool if r.get("id")}
        seen = set(by_id)
        retrieve_k = max(len(merged), int(getattr(self.settings, "prefilter_faiss_top_k", 800) or 800))
        index_n = min(len(pool_ids), int(self.faiss.dino_count or 0) or len(pool_ids))
        batches = [{"k": retrieve_k, "new": 0, "relevant_new": len(merged), "stop": ""}]
        extra_strong = 0
        if not features.dino_embedding or not self.faiss.available:
            self._last_retrieve_meta = {
                "adaptive_retrieve": False,
                "retrieve_k": retrieve_k,
                "batches": batches,
                "extra_strong": 0,
                "stop_reason": "no_dino_faiss",
            }
            return merged

        try:
            if len(features.dino_embedding) % 4 != 0 or len(features.dino_embedding) != 384 * 4:
                raise ValueError("invalid_dino_query_embedding")
            vec = np.frombuffer(features.dino_embedding, dtype=np.float32)
        except (TypeError, ValueError):
            logger.warning("Arama: geçersiz DINO sorgu embedding'i; FAISS adımı atlandı")
            self._last_retrieve_meta = {
                "adaptive_retrieve": False,
                "retrieve_k": retrieve_k,
                "batches": batches,
                "extra_strong": 0,
                "stop_reason": "invalid_dino_query_embedding",
            }
            return merged
        hits0 = self.faiss.search_dino(vec, k=retrieve_k)
        tail = float(hits0[-1][1]) if hits0 else None
        floor = float(threshold or 0.40)

        def _emit_merged() -> None:
            if not result_callback:
                return
            snapshot = sorted(merged, key=self._engine_sort_key)
            result_callback(snapshot, len(merged), retrieve_k)

        while True:
            nxt = next_retrieve_k(retrieve_k, index_n)
            if nxt is None:
                batches.append({"k": retrieve_k, "new": 0, "relevant_new": 0, "stop": "index_exhausted"})
                break
            with lat.span("faiss_expand"):
                hits = self.faiss.search_dino(vec, k=nxt)
            new_pairs = [
                (int(fid), float(sc))
                for fid, sc in hits
                if int(fid) not in seen and int(fid) in pool_ids
            ]
            new_ids = [fid for fid, _ in new_pairs]
            faiss_scores = [sc for _, sc in new_pairs]
            median = (
                float(sorted(faiss_scores)[len(faiss_scores) // 2])
                if faiss_scores
                else None
            )
            stop, reason = should_stop_expansion(
                new_id_count=len(new_ids),
                relevant_new=99,
                max_new_score=1.0,
                faiss_median_new=median,
                faiss_tail_ref=tail,
                floor=floor,
            )
            if stop and reason == "no_new_ids":
                batches.append({"k": nxt, "new": 0, "relevant_new": 0, "stop": reason})
                break
            if stop and reason == "faiss_similarity_drop":
                batches.append(
                    {
                        "k": nxt,
                        "new": len(new_ids),
                        "relevant_new": 0,
                        "stop": reason,
                    }
                )
                break
            if not new_ids:
                break
            with lat.span("hydrate_expand"):
                recs = self.db.get_indexed_files_by_ids(
                    new_ids,
                    include_processing_ready=True,
                    lightweight=False,
                )
                recs = merge_candidate_records(recs, indexed_pool, protected_ids)
                recs = self._records_with_family_overrides(recs)

            def _cb(batch_partial, n: int, tot: int) -> None:
                if result_callback is None:
                    return
                extra = [
                    r
                    for r in batch_partial
                    if int(r.file_id) not in by_id
                ]
                combined = merged + extra
                result_callback(
                    sorted(combined, key=self._engine_sort_key),
                    len(by_id) + n,
                    nxt,
                )

            batch_results = self._search_with_features(
                features,
                customer=customer,
                query_path=query_path,
                query_partial_hash=query_partial_hash,
                text_query=text_query,
                crop_search=crop_search,
                indexed=recs,
                query_profile=query_profile,
                feedback_adjustments=feedback_adj,
                pattern_group_ids=pattern_group_ids,
                wrong_result_ids=wrong_ids,
                result_callback=_cb if result_callback else None,
                stream_threshold=threshold,
            )
            relevant_new = 0
            max_sc = 0.0
            for r in batch_results:
                fid = int(r.file_id)
                if fid in by_id:
                    continue
                by_id[fid] = r
                merged.append(r)
                seen.add(fid)
                max_sc = max(max_sc, float(r.score or 0))
                if is_expand_worthy(r):
                    relevant_new += 1
                    if float(r.score or 0) >= 0.70:
                        extra_strong += 1
            retrieve_k = nxt
            stop, reason = should_stop_expansion(
                new_id_count=len(new_ids),
                relevant_new=relevant_new,
                max_new_score=max_sc,
                faiss_median_new=median,
                faiss_tail_ref=tail,
                floor=floor,
            )
            batches.append(
                {
                    "k": nxt,
                    "new": len(new_ids),
                    "relevant_new": relevant_new,
                    "stop": reason if stop else "",
                }
            )
            _emit_merged()
            if stop:
                break

        self._last_retrieve_meta = {
            "adaptive_retrieve": True,
            "retrieve_k": retrieve_k,
            "candidates_scored": len(merged),
            "batches": batches,
            "extra_strong": extra_strong,
            "stop_reason": (batches[-1].get("stop") if batches else ""),
            "initial_k": max(len(initial), 1),
        }
        if getattr(lat, "enabled", False):
            lat.set("retrieve_k", retrieve_k)
            lat.set("extra_strong", extra_strong)
            lat.set("expand_batches", len(batches))
        return merged

    def _adaptive_deep_file_ids(self, results: list[SearchResult]) -> list[int]:
        """Patch bütçesi: net eşleşmede küçük, kalabalık/belirsiz ailede geniş."""
        if not results:
            return []
        base = max(40, int(getattr(self.settings, "progressive_patch_base", 80) or 80))
        max_n = max(base, int(getattr(self.settings, "progressive_patch_max", 200) or 200))
        ranked = sorted(results, key=self._engine_sort_key)
        must: list[int] = []
        rest: list[int] = []
        for r in ranked:
            fid = int(r.file_id)
            if r.is_self_match or (r.debug or {}).get("protected_exact"):
                must.append(fid)
            else:
                rest.append(fid)
        rest_results = [r for r in ranked if int(r.file_id) in set(rest)]
        strong = sum(1 for r in rest_results if float(r.score) >= 0.70)
        top = float(rest_results[0].score) if rest_results else 0.0
        close = sum(1 for r in rest_results if top - float(r.score) <= 0.08)
        n = base
        if strong > 80:
            n = min(max_n, base + (strong - 80) // 2)
        if close > 40:
            n = min(max_n, base + close // 2)
        room = max(0, n - len(must))
        out: list[int] = []
        seen: set[int] = set()
        for fid in must + rest[:room]:
            if fid in seen:
                continue
            seen.add(fid)
            out.append(fid)
        return out

    def _score_indexed_record(
        self,
        rec: dict[str, Any],
        *,
        query: ExtractedFeatures,
        query_path: str,
        query_partial_hash: str,
        faiss_boost: float,
        text_query: str,
        crop_search: bool,
        query_profile: TextureProfile,
        wrong_ids: set[int],
        feedback_adj: dict[int, float],
        group_ids: set[int],
        skip_patch: bool,
    ) -> SearchResult:
        score, breakdown, category, is_self, debug, cluster = self._score_record(
            rec,
            query,
            query_path,
            query_partial_hash,
            faiss_boost,
            text_query=text_query,
            crop_search=crop_search,
            query_profile=query_profile,
            user_marked_wrong=int(rec["id"]) in wrong_ids,
            skip_patch=skip_patch,
        )
        fid = rec["id"]
        if fid in wrong_ids:
            score = min(score, 0.12)
            debug["user_wrong_block"] = True
            category = CATEGORY_WEAK
            cluster = {
                **cluster,
                "cluster_group": CLUSTER_UNRELATED,
                "cluster_reason": "Kullanıcı: yanlış eşleşme",
            }
            debug["protected_exact"] = False
        if fid in feedback_adj:
            delta = feedback_adj[fid]
            score = max(0.0, min(1.0, score + delta))
            debug["user_feedback_delta"] = round(delta, 4)
        if fid in group_ids and not is_self:
            score = max(0.0, min(1.0, score + 0.05))
            debug["pattern_group_delta"] = 0.05
        debug["final_score"] = round(score, 4)
        result = self._to_result(rec, score, breakdown)
        result.category = category
        result.is_self_match = is_self
        result.debug = debug
        from core.similarity_explanation import build_match_explanations

        result.match_explanations = build_match_explanations(
            score_percent=result.score_percent,
            debug=debug,
            text_breakdown=breakdown,
            query_ocr="",
            cand_ocr=str(rec.get("ocr_text") or ""),
            text_mode=bool(text_query),
        )
        if result.match_explanations:
            result.text_match_reason = " · ".join(
                f"✓ {x}" for x in result.match_explanations[:4]
            )
        result.cluster_group = cluster.get("cluster_group", "")
        result.cluster_reason = cluster.get("cluster_reason", "")
        if result.cluster_reason and not result.match_explanations:
            result.text_match_reason = result.cluster_reason
        result.hierarchy_score = cluster.get("hierarchy_score", 0.0)
        result.palette_similarity = cluster.get("palette_similarity", 0.0)
        result.texture_family_score = cluster.get("texture_family_score", 0.0)
        result.color_family = cluster.get("color_family", "")
        result.pattern_family = cluster.get("pattern_family", "")
        result.animal_print_type = cluster.get("animal_print_type", "")
        result.same_color_family = cluster.get("same_color_family", False)
        result.same_pattern_family = cluster.get("same_pattern_family", False)
        result.same_animal_family = cluster.get("same_animal_family", False)
        if rec.get("feedback_family_override"):
            result.debug["feedback_family_override"] = rec.get(
                "feedback_family_override"
            )
        self._apply_family_overrides(result)
        self._apply_learned_tier(result, rec)
        return result

    def _search_with_features(
        self,
        query: ExtractedFeatures,
        customer: str,
        query_path: str = "",
        query_partial_hash: str = "",
        text_query: str = "",
        crop_search: bool = False,
        indexed: list[dict[str, Any]] | None = None,
        query_profile: TextureProfile | None = None,
        feedback_adjustments: dict[int, float] | None = None,
        pattern_group_ids: set[int] | None = None,
        wrong_result_ids: set[int] | None = None,
        result_callback: Callable[[list[SearchResult], int, int], None] | None = None,
        stream_threshold: float = 0.0,
    ) -> list[SearchResult]:
        indexed = (
            indexed if indexed is not None else self._indexed_files(customer=customer)
        )
        if not indexed:
            logger.warning(
                "Arama: indexlenmiş dosya yok (filtre=%s)", self.settings.search_scope
            )
            return []

        qp = query_profile or TextureProfile.from_dict(query.texture_map or {})
        lat = self._latency
        lat_on = getattr(lat, "enabled", False)

        faiss_boost: dict[int, float] = {}
        limit_hint = max(
            int(getattr(self.settings, "prefilter_faiss_top_k", 800) or 800), 500
        )
        with lat.span("faiss_boost"):
            if (
                query.dino_embedding
                and self.faiss.available
                and self.settings.search_visual
            ):
                try:
                    if len(query.dino_embedding) % 4 != 0 or len(query.dino_embedding) != 384 * 4:
                        raise ValueError("invalid_dino_query_embedding")
                    dino_vec = np.frombuffer(query.dino_embedding, dtype=np.float32)
                except (TypeError, ValueError):
                    logger.warning("Arama: geçersiz DINO sorgu embedding'i; FAISS DINO atlandı")
                else:
                    for fid, score in self.faiss.search_dino(dino_vec, k=limit_hint * 5):
                        faiss_boost[fid] = faiss_boost.get(fid, 0.0) + score * 0.35
            if (
                query.clip_embedding
                and self.faiss.available
                and self.settings.search_visual
            ):
                try:
                    if len(query.clip_embedding) % 4 != 0 or len(query.clip_embedding) != 512 * 4:
                        raise ValueError("invalid_clip_query_embedding")
                    clip_vec = np.frombuffer(query.clip_embedding, dtype=np.float32)
                except (TypeError, ValueError):
                    logger.warning("Arama: geçersiz CLIP sorgu embedding'i; FAISS CLIP atlandı")
                else:
                    for fid, score in self.faiss.search_clip(clip_vec, k=limit_hint * 5):
                        faiss_boost[fid] = faiss_boost.get(fid, 0.0) + score * 0.25
        if lat_on:
            lat.set("faiss_boost_ids", len(faiss_boost))
            lat.set("clip_map_unavailable", bool(getattr(self.faiss, "clip_map_unavailable", False)))

        results: list[SearchResult] = []
        feedback_adj = feedback_adjustments or {}
        group_ids = pattern_group_ids or set()
        wrong_ids = wrong_result_ids or set()
        stream_started = False
        stream_last_emit = 0.0
        stream_last_count = 0
        total_candidates = len(indexed)
        has_patches = bool(query.patch_embeddings_meta)
        progressive = bool(has_patches and total_candidates > 64)

        def _emit(force: bool = False) -> None:
            nonlocal stream_started, stream_last_emit, stream_last_count
            if not result_callback or not results:
                return
            now = time.perf_counter()
            qualifies = any(
                r.score >= stream_threshold
                or r.is_self_match
                or (r.debug or {}).get("protected_exact")
                for r in results[-12:]
            ) or stream_started
            interval = 0.20 if progressive else 0.65
            should = (
                force
                or (not stream_started and (qualifies or len(results) >= 8))
                or (stream_started and now - stream_last_emit >= interval)
            )
            if not should:
                return
            snapshot = sorted(results, key=self._engine_sort_key)
            result_callback(snapshot, len(results), total_candidates)
            stream_started = stream_started or qualifies
            stream_last_emit = now
            stream_last_count = len(results)

        score_kw = dict(
            query=query,
            query_path=query_path,
            query_partial_hash=query_partial_hash,
            text_query=text_query,
            crop_search=crop_search,
            query_profile=qp,
            wrong_ids=wrong_ids,
            feedback_adj=feedback_adj,
            group_ids=group_ids,
        )
        with lat.span("cheap_scoring"):
            for rec in indexed:
                result = self._score_indexed_record(
                    rec,
                    faiss_boost=faiss_boost.get(rec["id"], 0.0),
                    skip_patch=progressive,
                    **score_kw,
                )
                results.append(result)
                if lat_on and not lat.counts.get("ttfr_ms"):
                    if (
                        result.score >= stream_threshold
                        or result.is_self_match
                        or result.debug.get("protected_exact")
                    ):
                        lat.set("ttfr_ms", round(lat.elapsed_ms(), 3))
                _emit()
        _emit(force=True)

        if progressive:
            deep_ids = self._adaptive_deep_file_ids(results)
            if getattr(lat, "enabled", False):
                lat.set("deep_patch_count", len(deep_ids))
            rec_by_id = {int(r["id"]): r for r in indexed}
            pos = {int(r.file_id): i for i, r in enumerate(results)}
            with lat.span("deep_patch"):
                for n, fid in enumerate(deep_ids):
                    rec = rec_by_id.get(int(fid))
                    if rec is None:
                        continue
                    verified = apply_verified_hashes(rec)
                    if verified.get("_hash_stale"):
                        from core.index_freeze import INDEX_FROZEN

                        if not INDEX_FROZEN:
                            self._persist_repaired_hashes(verified)
                    result = self._score_indexed_record(
                        verified,
                        faiss_boost=faiss_boost.get(int(fid), 0.0),
                        skip_patch=False,
                        **score_kw,
                    )
                    result.debug["analysis_stage"] = "verified"
                    idx = pos.get(int(fid))
                    if idx is not None:
                        results[idx] = result
                    if n == 0 or (n + 1) % 8 == 0:
                        _emit(force=True)
            _emit(force=True)
        if result_callback and results and stream_last_count != len(results):
            result_callback(
                sorted(results, key=self._engine_sort_key),
                len(results),
                total_candidates,
            )
        return results

    def _apply_learned_tier(self, result: SearchResult, rec: dict[str, Any]) -> None:
        """Kullanıcının öğrettiği benzerlik sınıfını sonuca uygula."""
        tm = dict(rec.get("texture_map") or {})
        tier_id = str(tm.get("user_similarity_tier") or "").strip()
        tier_label = str(tm.get("user_tier_label") or "").strip()
        if not tier_id and not tier_label:
            # Never hit SQLite per candidate — that alone breaks the 500ms gate.
            cached = rec.get("_learned_tier")
            if isinstance(cached, dict):
                tier_id = str(cached.get("tier_id") or "").strip()
                tier_label = str(cached.get("tier_label") or "").strip()
        if not tier_id and not tier_label:
            return
        from core.similarity_tiers import tier_info
        from core.similarity_tiers import tier_label as tier_display

        if tier_id:
            info = tier_info(tier_id)
            if info.get("cluster_group"):
                result.cluster_group = info["cluster_group"]
            else:
                result.cluster_group = tier_id
        if tier_label:
            result.cluster_reason = f"Öğrenilmiş sınıf: {tier_label}"
            result.debug["learned_tier_label"] = tier_label
        elif tier_id:
            result.cluster_reason = f"Öğrenilmiş sınıf: {tier_display(tier_id)}"
            result.debug["learned_tier_label"] = tier_display(tier_id)
        if tm.get("pattern_family"):
            result.pattern_family = str(tm.get("pattern_family"))
        result.debug["learned_tier_id"] = tier_id

    def _search_text_internal(
        self,
        text: str,
        customer: str,
        threshold: float | None = None,
        result_callback: Callable[[list[SearchResult], int, int], None] | None = None,
    ) -> list[SearchResult]:
        thr = threshold if threshold is not None else 0.30
        return self.search_by_text(
            text,
            limit=400,
            threshold=thr,
            customer=customer,
            result_callback=result_callback,
        )

    def _apply_structured_query_ranking(
        self,
        results: list[SearchResult],
        parsed: Any,
        text: str,
        *,
        dynamic_brand: str = "",
        brand_needles: list[str] | None = None,
    ) -> list[SearchResult]:
        """Çoklu doğal dil sorgusunu tek bir metin puanı yerine boyut bazında sıralar.

        Marka + renk + motif + tekrar + stil + doku gibi koşullar aynı anda
        geldiyse, sorguda gerçekten bulunan boyutlara dinamik ağırlık verir.
        Marka koşulu mevcutsa marka kanıtı görsel benzerlikten daha baskın olur.
        """
        if not results or parsed is None:
            return results

        from core.brand_evidence import extract_brand_evidence
        from core.textile_terms import normalize_turkish

        def norm_tokens(value: str, tokens: list[str] | None = None) -> set[str]:
            vals = list(tokens or [])
            if value:
                vals.append(value)
            out = set()
            for v in vals:
                n = normalize_turkish(str(v))
                out.update(x for x in re.split(r"[^a-z0-9çğıöşü]+", n) if len(x) >= 2)
            return out

        component_specs = [
            ("color", 0.0, getattr(parsed, "color", ""), getattr(parsed, "color_tokens", [])),
            ("motif", 0.0, getattr(parsed, "motif", ""), getattr(parsed, "motif_tokens", [])),
            ("repeat", 0.0, getattr(parsed, "repeat", ""), getattr(parsed, "repeat_tokens", [])),
            ("style", 0.0, getattr(parsed, "style", ""), getattr(parsed, "style_tokens", [])),
            ("texture", 0.0, getattr(parsed, "texture", ""), getattr(parsed, "texture_tokens", [])),
        ]

        brand_text = normalize_turkish(dynamic_brand or getattr(parsed, "brand", "") or "")
        brand_present = bool(brand_text or brand_needles)
        active = [(k, v, ts) for k, v, ts, toks in component_specs if ts or toks]
        active_count = len(active) + (1 if brand_present else 0)
        if active_count < 2:
            return results

        # Aktif boyutların toplam katkısı 0.45'e kadar çıkabilir; kalan görsel
        # skor korunur. Marka varsa görsel ağırlık bilinçli olarak düşürülür.
        if brand_present:
            base_visual = 0.48
            weights = {"brand": 0.30, "color": 0.07, "motif": 0.07, "repeat": 0.04, "style": 0.03, "texture": 0.03}
        else:
            base_visual = 0.64
            weights = {"brand": 0.0, "color": 0.10, "motif": 0.10, "repeat": 0.06, "style": 0.05, "texture": 0.05}

        active_weight_sum = sum(weights[k] for k, _, _ in active)
        # Sorguda bulunmayan boyutların ağırlığını görsel skora geri bırak.
        unused = max(0.0, 1.0 - base_visual - weights["brand"] - active_weight_sum)
        base_visual += unused

        for r in results:
            rec_text = " ".join([
                str(r.filename or ""),
                str(r.path or ""),
                str(r.debug.get("ocr_raw", "") or ""),
                str(r.debug.get("category_path", "") or ""),
                str(r.debug.get("manual_category_path", "") or ""),
                str(r.debug.get("text_search_blob", "") or ""),
                str(r.pattern_family or ""),
            ])
            hay = normalize_turkish(rec_text)
            hay_tokens = set(re.split(r"[^a-z0-9çğıöşü]+", hay))

            scores = {}
            if brand_present:
                ev = {normalize_turkish(x) for x in extract_brand_evidence({
                    "brand": brand_text,
                    "category_path": r.debug.get("category_path", ""),
                    "manual_category_path": r.debug.get("manual_category_path", ""),
                    "ocr_text": r.debug.get("ocr_raw", ""),
                    "filename": r.filename,
                    "path": r.path,
                    "text_search_blob": r.debug.get("text_search_blob", ""),
                }, getattr(self.settings, "db_path", ""))}
                needles = {normalize_turkish(x) for x in (brand_needles or []) if str(x).strip()}
                if brand_text and brand_text in ev:
                    scores["brand"] = 1.0
                elif needles and (ev & needles):
                    scores["brand"] = 1.0
                elif brand_text and brand_text in hay:
                    scores["brand"] = 0.88
                else:
                    scores["brand"] = float(r.breakdown.get("brand_alias_score", 0.0) or 0.0)

            for key, _, val, toks in component_specs:
                if not val and not toks:
                    continue
                qtok = norm_tokens(val, toks)
                if not qtok:
                    scores[key] = 0.0
                    continue
                overlap = len(qtok & hay_tokens) / max(1, len(qtok))
                phrase = normalize_turkish(val)
                phrase_hit = 1.0 if phrase and phrase in hay else 0.0
                scores[key] = max(overlap, phrase_hit)

            # Residual koşullar: anlamlı kalan kelimelerin eşleşmesi.
            residual = norm_tokens(getattr(parsed, "residual", ""))
            residual_score = (
                len(residual & hay_tokens) / max(1, len(residual))
                if residual else 0.0
            )

            structured = base_visual * float(r.score or 0.0)
            structured += weights["brand"] * scores.get("brand", 0.0)
            for key in ("color", "motif", "repeat", "style", "texture"):
                structured += weights[key] * scores.get(key, 0.0)
            if residual:
                structured += 0.05 * residual_score
            structured = min(1.0, max(0.0, structured))

            r.debug["structured_query"] = {
                "active_components": [k for k, _, _ in active],
                "brand_present": brand_present,
                "component_scores": {k: round(float(v), 4) for k, v in scores.items()},
                "residual_score": round(float(residual_score), 4),
                "visual_weight": round(float(base_visual), 4),
            }
            r.breakdown["structured_query"] = structured
            r.score = structured
            r.score_percent = round(structured * 100.0, 1)
        return results

    def _apply_hybrid_text_boost(
        self,
        results: list[SearchResult],
        text: str,
    ) -> list[SearchResult]:
        from core.text_index import text_search_score
        from core.textile_terms import expand_query_terms, query_family_hints

        semantic_enabled = bool(
            getattr(self.settings, "semantic_text_search_enabled", False)
        )
        variants = expand_query_terms(text, include_semantic=semantic_enabled)
        hints = query_family_hints(text, include_semantic=semantic_enabled)
        for r in results:
            if r.is_self_match or r.debug.get("protected_exact"):
                if r.debug.get("protected_exact"):
                    r.debug["text_boost_skipped"] = "protected_exact"
                continue
            rec = {
                "filename": r.filename,
                "path": r.path,
                "ocr_text": r.debug.get("ocr_raw", ""),
                "customer": r.customer,
                "source_name": r.source_name,
                "pattern_family": r.pattern_family,
                "texture_map": {},
                "text_search_blob": "",
            }
            _, breakdown, reasons = text_search_score(
                text, rec, semantic_enabled=semantic_enabled,
            )
            fname_score = breakdown.get("filename_score", 0.0)
            ocr_score = breakdown.get("ocr_score", 0.0)
            family_score = breakdown.get("family_score", 0.0)
            visual = r.score
            texture = r.breakdown.get("texture", 0)
            hybrid = (
                0.65 * visual
                + 0.15 * texture
                + 0.10 * fname_score
                + 0.05 * ocr_score
                + 0.05 * family_score
            )
            # Metin yanlış family'ye zorlamasın — sadece desteklesin
            if hints.get("pattern_family") and r.pattern_family:
                if hints["pattern_family"] != r.pattern_family and family_score < 0.5:
                    r.debug["text_family_conflict"] = (
                        f"Görsel: {r.pattern_family} / Metin: {hints['pattern_family']}"
                    )
            r.score = min(1.0, hybrid)
            r.score_percent = round(r.score * 100, 1)
            r.breakdown["filename_text"] = fname_score
            r.breakdown["ocr_text"] = ocr_score
            r.breakdown["family_text"] = family_score
            r.debug["final_score"] = round(r.score, 4)
            r.debug["filename_score"] = round(fname_score, 4)
            r.debug["ocr_score"] = round(ocr_score, 4)
            r.debug["threshold_passed"] = r.score >= self.settings.similarity_threshold
            if reasons:
                r.text_match_reason = " | ".join(reasons[:2])
            r.debug["matched_reason"] = (
                r.debug.get("matched_reason", "")
                + f"\nMetin boost: {text} → dosya adı %{fname_score * 100:.0f}"
            )
        return results

    def _score_record(
        self,
        rec: dict[str, Any],
        query: ExtractedFeatures,
        query_path: str,
        query_partial_hash: str,
        faiss_boost: float,
        text_query: str = "",
        crop_search: bool = False,
        query_profile: TextureProfile | None = None,
        user_marked_wrong: bool = False,
        skip_patch: bool = False,
    ) -> tuple[float, dict[str, float], str, bool, dict[str, Any], dict[str, Any]]:
        rec_path = normalize_path(rec.get("path", ""))
        is_self = bool(query_path and rec_path == query_path)
        qp = query_profile or TextureProfile.from_dict(query.texture_map or {})
        lat_on = getattr(self._latency, "enabled", False)
        t = time.perf_counter() if lat_on else 0.0

        phash_sim = (
            phash_similarity(rec.get("phash", ""), query.phash) if query.phash else 0.0
        )
        dhash_sim = (
            phash_similarity(rec.get("dhash", ""), query.dhash) if query.dhash else 0.0
        )
        whash_sim = (
            phash_similarity(rec.get("whash", ""), query.whash) if query.whash else 0.0
        )
        if lat_on:
            t = self._lat_add("rank.hash", t)

        color_sim = (
            FeatureExtractor.color_similarity(
                query.color_hist, rec.get("color_hist") or b""
            )
            if self.settings.search_color
            else 0.0
        )
        if lat_on:
            t = self._lat_add("rank.color", t)
        tex_b = rec.get("texture_features")
        if isinstance(tex_b, str):
            import json

            tex_b = json.loads(tex_b)
        texture_sim = (
            FeatureExtractor.texture_similarity(query.texture_features, tex_b or [])
            if self.settings.search_texture
            else 0.0
        )
        if lat_on:
            t = self._lat_add("rank.texture", t)
        if skip_patch:
            patch_sim, multi_scale_patch = 0.0, 0.0
        else:
            patch_sim, multi_scale_patch = self._patch_similarity(
                query.patch_embeddings_meta,
                rec.get("patch_embeddings_meta") or [],
            )
        if lat_on:
            t = self._lat_add("rank.patch", t)
        cand_texture_map = rec.get("texture_map") or {}
        if not isinstance(cand_texture_map, dict):
            cand_texture_map = {}
        query_texture_map = query.texture_map or {}
        if not isinstance(query_texture_map, dict):
            query_texture_map = {}
        dna_sim = self._pattern_dna_similarity(query_texture_map, cand_texture_map)
        if lat_on:
            t = self._lat_add("rank.dna", t)
        dna_match_fields = self._dna_match_fields(query_texture_map, cand_texture_map)
        semantic_sim = self._semantic_tag_similarity(query_texture_map, cand_texture_map)
        if lat_on:
            t = self._lat_add("rank.semantic", t)
        repeat_sim = self._repeat_structure_similarity(query_texture_map, cand_texture_map)
        motif_sim = self._motif_similarity(query_texture_map, cand_texture_map)
        filename_score = (
            filename_text_boost_only(text_query, rec) if text_query else 0.0
        )

        dino_sim = FeatureExtractor.embedding_similarity(
            query.dino_embedding,
            rec.get("dino_embedding") or b"",
        )
        if lat_on:
            t = self._lat_add("rank.dino", t)
        clip_sim = FeatureExtractor.embedding_similarity(
            query.clip_embedding,
            rec.get("clip_embedding") or b"",
        )
        if lat_on:
            t = self._lat_add("rank.clip", t)

        breakdown = {
            "phash": phash_sim,
            "dhash": dhash_sim,
            "whash": whash_sim,
            "color": color_sim,
            "texture": texture_sim,
            "patch": patch_sim,
            "multi_scale_patch": multi_scale_patch,
            "dna": dna_sim,
            "semantic": semantic_sim,
            "repeat": repeat_sim,
            "motif": motif_sim,
            "filename_text": filename_score,
            "ocr_text": 0.0,
            "dino": dino_sim,
            "clip": clip_sim,
            "faiss_boost": faiss_boost,
        }

        # Kademe A — perceptual hash (AI kapalıyken ana skor)
        hash_score = self._tier_a_score(phash_sim, dhash_sim, whash_sim)
        exact_search_score = max(hash_score, phash_sim, dhash_sim * 0.96, whash_sim * 0.92)

        # Kademe B — doku / renk / AI
        tier_b = self._tier_b_score(breakdown, dino_sim, clip_sim, faiss_boost)

        use_ai = (
            self.settings.ai_embedding_enabled
            and self.settings.search_visual
            and (query.dino_embedding or query.clip_embedding)
            and (rec.get("dino_embedding") or rec.get("clip_embedding"))
        )

        if use_ai:
            w = self.settings.effective_weights()
            score = (
                w.get("phash", 0) * phash_sim
                + w.get("color", 0) * color_sim
                + w.get("texture", 0) * texture_sim
                + w.get("dino", 0) * dino_sim
                + w.get("clip", 0) * clip_sim
            )
            score = max(score, hash_score * 0.85 + tier_b * 0.15)
        else:
            score = self.settings.classical_score(
                phash_sim,
                dhash_sim,
                whash_sim,
                texture_sim,
                color_sim,
            )

        global_score = self._tier_a_score(phash_sim, dhash_sim, whash_sim)
        structure_score = min(
            1.0, 0.45 * global_score + 0.40 * patch_sim + 0.15 * multi_scale_patch
        )
        palette_score = color_sim
        if self.settings.color_weight_mode == "ignore":
            palette_score = palette_score * 0.15
            texture_sim = min(1.0, texture_sim * 1.12)
            patch_sim = max(patch_sim, multi_scale_patch * 0.95)
            structure_score = min(1.0, structure_score * 1.08)

        if self.settings.search_mode == "exact":
            score = max(score, global_score)
            if patch_sim >= 0.90:
                score = max(score, 0.92 * global_score + 0.08 * patch_sim)
        elif self.settings.search_mode == "style":
            color_w = 0.02 if self.settings.color_weight_mode == "ignore" else 0.10
            score = max(
                score,
                0.15 * global_score
                + 0.45 * texture_sim
                + 0.30 * patch_sim
                + color_w * color_sim
                + 0.05 * filename_score,
            )
        else:
            color_w = 0.03 if self.settings.color_weight_mode == "ignore" else 0.10
            patch_pattern_score = (
                0.20 * global_score
                + 0.35 * patch_sim
                + 0.20 * multi_scale_patch
                + 0.18 * texture_sim
                + color_w * color_sim
                + 0.05 * filename_score
            )
            score = max(score, patch_pattern_score)

        # Area Select uses the same score composition as full-image search.
        # (Former crop-only patch/texture rewrite diverged from pattern ranking.)

        debug: dict[str, Any] = {}
        knowledge_explanation: dict[str, Any] = {}
        kb: dict[str, Any] = {}
        if rec.get("_hash_stale"):
            debug["hash_repaired_from"] = rec.get("_hash_verified_from", "")

        protected_match = (
            False
            if user_marked_wrong
            else is_protected_exact_match(
                is_self=is_self,
                phash_sim=phash_sim,
                dhash_sim=dhash_sim,
                whash_sim=whash_sim,
                score=score,
                query_partial_hash=query_partial_hash,
                rec_partial_hash=rec.get("partial_hash", ""),
                query_full_hash="",
                rec_full_hash=rec.get("full_hash", ""),
            )
        )

        floor = (
            0.0
            if user_marked_wrong
            else protected_score_floor(
                is_self=is_self,
                phash_sim=phash_sim,
                dhash_sim=dhash_sim,
                whash_sim=whash_sim,
                different_extension=bool(
                    query_path
                    and Path(rec_path).suffix.lower() != Path(query_path).suffix.lower()
                ),
            )
        )
        if floor > 0:
            score = max(score, floor)

        query_family = effective_query_family(qp)
        cand_prof_early = TextureProfile.from_dict(rec.get("texture_map") or {})
        from core.taxonomy import normalize_family as _norm_fam

        cf_n_early = _norm_fam(cand_prof_early.pattern_family or "unknown")
        qf_n_early = _norm_fam(query_family or qp.pattern_family or "unknown")

        leopard_query = (
            is_confident_animal_print(qp) and qp.animal_print_type in LEOPARD_FAMILY
        )
        leopard_candidate = (
            is_confident_animal_print(cand_prof_early)
            or cand_prof_early.animal_print_type in LEOPARD_FAMILY
        )

        different_extension = bool(
            query_path
            and Path(rec_path).suffix.lower() != Path(query_path).suffix.lower()
        )

        # Boost kuralları — bilinmeyen/zemin aileleri ASLA "aynı aile" sayılmaz.
        # Önceki sürümde qf="unknown" olduğunda `same_family=True` oluyordu;
        # bu da alakasız fotoğrafların UI'da "Aynı Desen Ailesi" görünmesine ve
        # family skorunun görsel benzerlikten yapay biçimde yükselmesine neden
        # oluyordu.
        _NON_FAMILY = {"", "unknown", "texture_ground", "plain", "document", "garment_photo", "icon_logo_non_textile"}
        same_family = bool(
            qf_n_early
            and cf_n_early
            and qf_n_early == cf_n_early
            and qf_n_early not in _NON_FAMILY
        )
        same_family_evidence = same_family
        query_dna = query_texture_map.get("pattern_dna") if isinstance(query_texture_map.get("pattern_dna"), dict) else {}
        cand_dna = cand_texture_map.get("pattern_dna") if isinstance(cand_texture_map.get("pattern_dna"), dict) else {}
        same_collection = bool(
            self._field_terms(query_dna.get("collection"))
            and self._field_terms(query_dna.get("collection")) == self._field_terms(cand_dna.get("collection"))
        )
        same_series = bool(
            self._field_terms(query_dna.get("series"))
            and self._field_terms(query_dna.get("series")) == self._field_terms(cand_dna.get("series"))
        )
        same_designer_style = bool(
            self._field_terms(query_dna.get("designer_style") or query_dna.get("brand_style") or query_dna.get("style"))
            and self._field_terms(query_dna.get("designer_style") or query_dna.get("brand_style") or query_dna.get("style"))
            == self._field_terms(cand_dna.get("designer_style") or cand_dna.get("brand_style") or cand_dna.get("style"))
        )
        # DNA/semantic/motif benzerliği tek başına "aynı aile" kanıtı değildir.
        # Özellikle insan fotoğrafı ↔ çiçek/ayakkabı gibi yanlış adaylarda
        # yüksek embedding/semantic skorları family katmanını şişirmemeli.
        from core.group_gates import family_relationship as _family_relationship
        _family_compatible = bool(
            qf_n_early not in _NON_FAMILY
            and cf_n_early not in _NON_FAMILY
            and _family_relationship(qf_n_early, cf_n_early) in ("same", "related")
        )
        same_animal_print = bool(
            str(qp.animal_print_type or "").strip()
            and str(cand_prof_early.animal_print_type or "").strip()
            and str(qp.animal_print_type).strip().casefold()
            == str(cand_prof_early.animal_print_type).strip().casefold()
        )
        family_evidence = max(
            1.0 if same_family_evidence else 0.0,
            1.0 if same_collection else 0.0,
            1.0 if same_series else 0.0,
            1.0 if same_animal_print else 0.0,
            dna_sim if (_family_compatible and dna_match_fields.get("Family")) else 0.0,
            semantic_sim if (_family_compatible and semantic_sim >= 0.62) else 0.0,
            motif_sim if (_family_compatible and motif_sim >= 0.62) else 0.0,
        )
        embedding_sim = max(dino_sim, clip_sim, faiss_boost)
        pattern_family_score = min(
            1.0,
            0.18 * dna_sim
            + 0.14 * semantic_sim
            + 0.10 * motif_sim
            + 0.10 * repeat_sim
            + 0.20 * patch_sim
            + 0.16 * texture_sim
            + 0.07 * multi_scale_patch
            + 0.03 * embedding_sim
            + 0.02 * palette_score,
        )
        try:
            from core.knowledge_graph import reason_knowledge

            visual_proxy = max(global_score, patch_sim, texture_sim, phash_sim * 0.9)
            t_kb = time.perf_counter() if lat_on else 0.0
            reasoned = reason_knowledge(
                text_query=str(text_query or ""),
                query_family=qf_n_early or query_family,
                query_motif=str(
                    (query_texture_map or {}).get("pattern_subtype")
                    or (query_texture_map or {}).get("motif")
                    or qp.animal_print_type
                    or ""
                ),
                query_color=str(qp.color_family or ""),
                cand_family=cf_n_early or cand_prof_early.pattern_family,
                cand_motif=str(
                    cand_texture_map.get("pattern_subtype")
                    or cand_texture_map.get("motif")
                    or cand_prof_early.animal_print_type
                    or ""
                ),
                cand_color=str(cand_prof_early.color_family or ""),
                cand_filename=str(rec.get("filename") or ""),
                visual_score=float(visual_proxy or 0.0),
                dna_score=float(dna_sim or 0.0),
                semantic_score=float(semantic_sim or 0.0),
                has_ocr=bool(rec.get("ocr_text") or text_query),
            )
            if lat_on:
                t_kb = self._lat_add("rank.knowledge", t_kb)
            kb = {
                "kb_combined": reasoned.knowledge_match,
                "kb_weight": reasoned.knowledge_weight,
                "kb_contribution": reasoned.knowledge_contribution,
                "kb_family": 1.0 if reasoned.family_ids else 0.0,
                "kb_motif": reasoned.knowledge_match,
                **{f"w_{k}": v for k, v in reasoned.modality_weights.items()},
            }
            kb_w = float(reasoned.knowledge_weight or 0.0)
            pattern_family_score = min(
                1.0,
                pattern_family_score + kb_w * float(reasoned.knowledge_match or 0.0),
            )
            # Kontrollü aile yayılımı: genişleyen aileler aynı kökteyse hafif bonus
            if reasoned.family_ids and cf_n_early in reasoned.family_ids:
                pattern_family_score = min(1.0, pattern_family_score + 0.06 * kb_w)
            knowledge_explanation = reasoned.explanation
        except Exception:
            kb = {}
            knowledge_explanation = {}
        if same_family_evidence:
            pattern_family_score = min(1.0, pattern_family_score + 0.12)
        if same_animal_print:
            # Same indexed animal_print_type = pattern identity across views/reps
            # even when pattern_family label disagrees (e.g. floral mislabel).
            pattern_family_score = min(1.0, pattern_family_score + 0.20)
        if same_collection:
            pattern_family_score = min(1.0, pattern_family_score + 0.10)
        if same_series:
            pattern_family_score = min(1.0, pattern_family_score + 0.06)
        family_relief = bool(
            same_family_evidence
            or same_collection
            or same_series
            or same_animal_print
            or (_family_compatible and dna_sim >= 0.72 and dna_match_fields.get("Family"))
            or (_family_compatible and semantic_sim >= 0.70 and motif_sim >= 0.55)
        )
        # Yapısal benzerlik düşükse doku/patch tek başına skoru yükseltemesin.
        # Aynı family / aynı animal_print_type / güçlü DNA varsa gate gevşer
        # (model ön ↔ kumaş/metraj: phash düşük olsa da pattern identity korunur).
        if not protected_match and global_score < 0.62 and phash_sim < 0.85:
            structural_cap = 0.28 + global_score * 0.50 + patch_sim * 0.05
            if family_relief:
                structural_cap = max(
                    structural_cap,
                    0.58 + pattern_family_score * 0.22 + max(patch_sim, multi_scale_patch) * 0.08,
                )
            elif leopard_query and leopard_candidate and patch_sim >= 0.45:
                structural_cap = max(
                    structural_cap, 0.55 + patch_sim * 0.25 + texture_sim * 0.12
                )
            score = min(score, structural_cap)
        if not protected_match and family_evidence >= 0.55:
            score = max(score, min(0.88, pattern_family_score))
        # pattern_first_soft applied after structural/low-hash caps (below)
        strong_hash = phash_sim >= 0.95 and dhash_sim >= 0.90
        if is_self:
            score = 1.0
        elif (
            query_partial_hash
            and rec.get("partial_hash") == query_partial_hash
            and strong_hash
        ):
            score = max(score, 0.98)
        elif (
            different_extension
            and global_score >= 0.80
            and patch_sim >= 0.98
            and texture_sim >= 0.85
            and (protected_match or same_family or phash_sim >= 0.92)
        ):
            score = max(score, 0.96)
        elif (
            different_extension
            and same_family
            and dhash_sim >= 0.98
            and whash_sim >= 0.90
            and texture_sim >= 0.95
            and color_sim >= 0.95
        ):
            # Kayıplı JPG dönüşümü pHash'i düşürebilir; diğer bağımsız
            # kanıtlar aynı görüntüyü doğruluyorsa format varyantını koru.
            score = max(score, 0.95)
        elif (
            dhash_sim >= 0.98
            and whash_sim >= 0.88
            and texture_sim >= 0.85
            and phash_sim >= 0.88
        ):
            score = max(score, 0.96 + 0.04 * color_sim)
        elif phash_sim >= 0.98 and dhash_sim >= 0.95:
            score = max(score, 0.98)
        elif phash_sim >= 0.98:
            score = max(score, 0.98)
        elif phash_sim >= 0.95 and dhash_sim >= 0.95:
            score = max(score, 0.94)

        if same_animal_print:
            # Family label mismatch must not punish shared animal_print_type
            # (e.g. animal_print query vs floral-mislabeled leopard fabric).
            score, cluster_group_pre, cluster_reason_pre = score, "", ""
            debug["family_mismatch_skipped_same_animal"] = True
        else:
            score, cluster_group_pre, cluster_reason_pre = (
                self._apply_family_mismatch_penalty(
                    query_family,
                    cand_prof_early,
                    score,
                    "",
                    "",
                    phash_sim=phash_sim,
                    patch_sim=patch_sim,
                )
            )

        # Düşük yapısal benzerlik alakasız görselleri aşağı iter; ancak
        # güçlü Pattern Family kanıtı varsa family skorunu tekrar ezme.
        if phash_sim < 0.70 and dhash_sim < 0.78:
            structural_floor_cap = max(
                0.42, patch_sim * 0.75, texture_sim * 0.65
            )
            if family_relief:
                structural_floor_cap = max(
                    structural_floor_cap,
                    min(
                        0.90,
                        pattern_family_score,
                        0.58
                        + pattern_family_score * 0.22
                        + max(patch_sim, multi_scale_patch) * 0.08,
                    ),
                )
            score = min(score, structural_floor_cap)
        if phash_sim < 0.55 and dhash_sim < 0.65:
            low_hash_cap = max(0.35, patch_sim * 0.70, texture_sim * 0.60)
            if family_relief:
                low_hash_cap = max(
                    low_hash_cap,
                    min(0.88, pattern_family_score),
                )
            if same_animal_print:
                low_hash_cap = max(low_hash_cap, 0.62)
            score = min(score, low_hash_cap)

        # Pattern-first soft path AFTER structural caps: garment/model full-frame
        # vs fabric/detail must keep shared animal_print_type above burying gates.
        # Intent gate: person/object-only text must not lift fabric-only leopard peers.
        _suppress_pf = False
        if text_query:
            try:
                from core.intent_evidence_routing import (
                    build_intent_evidence_plan,
                    should_suppress_pattern_first,
                )

                _suppress_pf = should_suppress_pattern_first(
                    build_intent_evidence_plan(text_query)
                )
            except Exception:
                _suppress_pf = False
        if (
            not _suppress_pf
            and not protected_match
            and (
                same_animal_print
                or (
                    qf_n_early == "animal_print"
                    and dna_sim >= 0.72
                    and _family_compatible
                )
            )
        ):
            lifted = min(0.90, max(float(score), 0.45 * float(score) + 0.55 * pattern_family_score))
            if same_animal_print:
                lifted = max(lifted, 0.62)
            score = max(float(score), lifted)
            debug["pattern_first_soft"] = True
            debug["pattern_first_same_animal"] = bool(same_animal_print)
            if crop_search:
                debug["pattern_first_crop_aligned"] = True
        elif _suppress_pf and same_animal_print:
            debug["pattern_first_suppressed_by_intent"] = True

        # Renk varyantı: yapı benzer, renk farklı — farklı aileye uygulanmaz
        is_color_variant = phash_sim >= 0.75 and dhash_sim >= 0.70 and color_sim < 0.55
        if (
            is_color_variant
            and qf_n_early not in ("unknown", "texture_ground", "plain", "")
            and cf_n_early != qf_n_early
        ):
            is_color_variant = False

        category = self._categorize(score, is_color_variant, is_self)
        if not is_self and different_extension and score >= 0.90:
            category = CATEGORY_FORMAT_VARIANT
        elif not is_self and is_color_variant and score >= 0.55:
            category = CATEGORY_COLOR_VARIANT
        elif (
            patch_sim >= 0.55
            and texture_sim >= 0.50
            and score >= 0.45
            and cand_prof_early.pattern_family != "floral"
        ):
            category = CATEGORY_TEXTILE_TEXTURE
        reasons = []
        if is_self:
            reasons.append("Aynı dosya yolu")
        if (
            not is_self
            and different_extension
            and (phash_sim >= 0.98 or dhash_sim >= 0.98)
        ):
            reasons.append("Aynı görsel - farklı format")
        if phash_sim >= 0.85:
            reasons.append(f"Görsel yapı benzerliği: %{phash_sim * 100:.0f}")
        if texture_sim >= 0.5:
            reasons.append(f"Doku benzerliği: %{texture_sim * 100:.0f}")
        if patch_sim >= 0.5:
            reasons.append(f"Patch/tile benzerliği: %{patch_sim * 100:.0f}")
        if filename_score >= 0.5 and cand_prof_early.pattern_family != "floral":
            reasons.append(f"Dosya adı metin ipucu: %{filename_score * 100:.0f}")
        if color_sim >= 0.5:
            reasons.append(f"Renk benzerliği: %{color_sim * 100:.0f}")
        if crop_search:
            reasons.append("Seçili bölgeyle arandı")

        cand_prof = cand_prof_early
        pal_sim = palette_similarity(qp.color_family, cand_prof.color_family)
        tex_fam_sc = texture_family_score(qp, cand_prof)
        t_fam = time.perf_counter() if lat_on else 0.0
        cluster_group, cluster_reason = assign_cluster_group(
            qp,
            cand_prof,
            is_self=is_self,
            score=score,
            phash_sim=phash_sim,
            dhash_sim=dhash_sim,
            patch_sim=patch_sim,
            different_extension=different_extension,
            is_color_variant=is_color_variant,
            filename=rec.get("filename", ""),
        )
        if lat_on:
            self._lat_add("rank.family", t_fam)
        hierarchy_sc = compute_hierarchy_score(
            qp,
            cand_prof,
            global_score,
            patch_sim,
            pal_sim,
            tex_fam_sc,
        )
        if self.settings.search_mode in ("style", "comprehensive"):
            score = max(score, 0.50 * score + 0.50 * hierarchy_sc)
        if not protected_match and family_evidence >= 0.55:
            pattern_family_score = min(
                1.0,
                max(pattern_family_score, 0.72 * pattern_family_score + 0.28 * hierarchy_sc),
            )
            score = max(score, min(0.90, pattern_family_score))

        cluster_group, cluster_reason, score, gate_note = enforce_cluster_gate(
            QueryContext(
                pattern_family=qp.pattern_family,
                pattern_subtype=qp.pattern_subtype,
                animal_print_type=qp.animal_print_type,
                classification_confidence=qp.classification_confidence,
            ),
            cand_prof,
            cluster_group,
            cluster_reason,
            score,
            protected_exact=protected_match,
            phash_sim=phash_sim,
            dhash_sim=dhash_sim,
            patch_sim=patch_sim,
            query_animal_type=qp.animal_print_type or "",
            candidate_animal_type=cand_prof.animal_print_type or "",
        )
        if cluster_group_pre and cluster_group_pre in (
            CLUSTER_DISTANT,
            CLUSTER_UNRELATED,
        ):
            cluster_group = cluster_group_pre
            if cluster_reason_pre:
                cluster_reason = cluster_reason_pre
        if gate_note and gate_note not in cluster_reason:
            cluster_reason = (
                f"{cluster_reason} ({gate_note})" if cluster_reason else gate_note
            )

        if self.settings.search_mode == "comprehensive" and cluster_group in (
            CLUSTER_UNRELATED,
            CLUSTER_DISTANT,
        ):
            score = min(score, 0.38)

        layer = self._result_layer(
            cluster_group,
            protected_exact=protected_match,
            is_self=is_self,
            pattern_family_score=pattern_family_score,
        )
        variant_badge = self._variant_badge(
            is_self=is_self,
            protected_exact=protected_match,
            cluster_group=cluster_group,
            same_family=same_family_evidence,
            color_variant=is_color_variant,
            repeat_sim=repeat_sim,
            motif_sim=motif_sim,
            semantic_sim=semantic_sim,
            texture_sim=texture_sim,
            patch_sim=patch_sim,
            multi_scale_patch=multi_scale_patch,
            global_score=global_score,
            pattern_family_score=pattern_family_score,
        )
        family_explanations = self._family_explanations(
            same_family=same_family_evidence,
            same_collection=same_collection,
            same_series=same_series,
            same_designer_style=same_designer_style,
            dna_sim=dna_sim,
            semantic_sim=semantic_sim,
            texture_sim=texture_sim,
            repeat_sim=repeat_sim,
            motif_sim=motif_sim,
            structure_score=structure_score,
            patch_sim=patch_sim,
            multi_scale_patch=multi_scale_patch,
            dna_match_fields=dna_match_fields,
        )

        reasons.append(cluster_reason)
        debug.update(
            {
                "has_thumbnail": bool(rec.get("thumbnail_path")),
                "has_features": bool(rec.get("phash")),
                "is_self_match": is_self,
                "tier_a_hash": round(hash_score, 4),
                "tier_b": round(tier_b, 4),
                "filtered_out": False,
                "matched_reason": (
                    "\n".join(f"- {x}" for x in reasons)
                    if reasons
                    else "- Genel benzerlik"
                ),
                "ocr_raw": rec.get("ocr_text", ""),
                "final_score": round(score, 4),
                "result_layer": layer,
                "family_variant_badge": variant_badge,
                "family_explanations": family_explanations,
                "same_collection": same_collection,
                "same_series": same_series,
                "same_designer_style": same_designer_style,
                "exact_score": round(exact_search_score, 4),
                "exact_search_score": round(exact_search_score, 4),
                "family_variant_score": round(pattern_family_score, 4),
                "pattern_family_score": round(pattern_family_score, 4),
                "exact_breakdown": {
                    "phash": round(phash_sim, 4),
                    "dhash": round(dhash_sim, 4),
                    "whash": round(whash_sim, 4),
                    "structure": round(structure_score, 4),
                    "crop_patch": round(max(patch_sim, multi_scale_patch), 4),
                },
                "family_breakdown": {
                    "dna": round(dna_sim, 4),
                    "semantic": round(semantic_sim, 4),
                    "texture": round(texture_sim, 4),
                    "patch": round(patch_sim, 4),
                    "multi_scale_patch": round(multi_scale_patch, 4),
                    "repeat": round(repeat_sim, 4),
                    "motif": round(motif_sim, 4),
                    "embedding": round(embedding_sim, 4),
                    "color": round(palette_score, 4),
                    "same_family_bonus": 1.0 if same_family_evidence else 0.0,
                    "kb_family": round(float(kb.get("kb_family") or 0.0), 4),
                    "kb_motif": round(float(kb.get("kb_motif") or 0.0), 4),
                    "kb_combined": round(float(kb.get("kb_combined") or 0.0), 4),
                    "kb_weight": round(float(kb.get("kb_weight") or 0.0), 4),
                    "kb_contribution": round(float(kb.get("kb_contribution") or 0.0), 4),
                },
                "knowledge_explanation": knowledge_explanation,
                "knowledge_match_labels": list(
                    (knowledge_explanation or {}).get("knowledge_match_labels") or []
                ),
                "knowledge_score": round(
                    float((knowledge_explanation or {}).get("knowledge_score") or 0.0), 4
                ),
                "knowledge_weight": round(
                    float((knowledge_explanation or {}).get("knowledge_weight") or 0.0), 4
                ),
                "modality_weights": dict(
                    (knowledge_explanation or {}).get("modality_weights") or {}
                ),
                "contribution_scores": {
                    "exact": round(exact_search_score, 4),
                    "dna": round(dna_sim, 4),
                    "texture": round(texture_sim, 4),
                    "semantic": round(semantic_sim, 4),
                    "patch": round(max(patch_sim, multi_scale_patch), 4),
                },
                "dna_match_fields": dna_match_fields,
                "same_family": same_family_evidence,
                "semantic_match": semantic_sim >= 0.55,
                "texture_match": texture_sim >= 0.55,
                "patch_match": max(patch_sim, multi_scale_patch) >= 0.55,
                "repeat_match": repeat_sim >= 0.50,
                "motif_match": motif_sim >= 0.50,
                "structure_match": structure_score >= 0.60,
                "phash_score": round(phash_sim, 4),
                "dhash_score": round(dhash_sim, 4),
                "whash_score": round(whash_sim, 4),
                "texture_score": round(texture_sim, 4),
                "patch_score": round(patch_sim, 4),
                "color_score": round(color_sim, 4),
                "filename_score": round(
                    max(filename_score, breakdown.get("filename_text", 0.0)), 4
                ),
                "ocr_score": round(breakdown.get("ocr_text", 0.0), 4),
                "ai_score": round(max(dino_sim, clip_sim, faiss_boost), 4),
                "category": category,
                "source_filter_passed": True,
                "threshold_passed": score >= self.settings.similarity_threshold,
                "reader_used": "pillow",
                "file_status": rec.get("status", ""),
                "cluster_group": cluster_group,
                "pattern_family": cand_prof.pattern_family,
                "animal_print_type": cand_prof.animal_print_type,
                "result_family": cand_prof.pattern_family,
                "result_subtype": cand_prof.pattern_subtype,
                "result_confidence": round(cand_prof.classification_confidence, 4),
                "query_family": query_family,
                "query_subtype": qp.pattern_subtype,
                "query_confidence": round(qp.classification_confidence, 4),
                "animal_score": float(
                    (cand_prof.texture_map or {}).get("animal_score", 0)
                ),
                "floral_score": float(
                    (cand_prof.texture_map or {}).get("floral_score", 0)
                ),
                "marble_score": float(
                    (cand_prof.texture_map or {}).get("marble_score", 0)
                ),
                "why_family_selected": (cand_prof.texture_map or {}).get(
                    "why_family_selected", ""
                ),
                "why_animal_rejected": (cand_prof.texture_map or {}).get(
                    "why_animal_rejected", ""
                ),
                "color_family": cand_prof.color_family,
                "texture_family_score": round(tex_fam_sc, 4),
                "palette_similarity": round(pal_sim, 4),
                "hierarchy_score": round(hierarchy_sc, 4),
                "visual_score": round(global_score, 4),
                "global_visual_score": round(global_score, 4),
                "multi_scale_patch_score": round(multi_scale_patch, 4),
                "dna_score": round(dna_sim, 4),
                "semantic_score": round(semantic_sim, 4),
                "repeat_score": round(repeat_sim, 4),
                "motif_score": round(motif_sim, 4),
                "structure_score": round(structure_score, 4),
                "palette_score": round(palette_score, 4),
                "protected_exact": protected_match,
                # Inspector/category prediction must use the selected candidate's
                # persisted DNA, never the query family or similarity score.
                "texture_map": dict(rec.get("texture_map") or {}),
                **{k: round(v, 4) for k, v in breakdown.items()},
            }
        )
        cluster_info = {
            "cluster_group": cluster_group,
            "cluster_reason": cluster_reason,
            "hierarchy_score": hierarchy_sc,
            "palette_similarity": pal_sim,
            "texture_family_score": tex_fam_sc,
            "color_family": cand_prof.color_family,
            "pattern_family": cand_prof.pattern_family,
            "animal_print_type": cand_prof.animal_print_type,
            "same_color_family": pal_sim >= 0.75,
            "same_pattern_family": (
                qp.pattern_family == cand_prof.pattern_family
                and qp.pattern_family not in ("", "unknown")
            ),
            "same_animal_family": bool(
                qp.animal_print_type
                and qp.animal_print_type == cand_prof.animal_print_type
            ),
        }
        debug["analysis_stage"] = "preliminary" if skip_patch else "verified"
        debug["patch_deferred"] = bool(skip_patch)
        return score, breakdown, category, is_self, debug, cluster_info

    @staticmethod
    def _is_protected_match(
        is_self: bool,
        phash_sim: float,
        dhash_sim: float,
        score: float,
        query_partial_hash: str,
        rec_partial_hash: str,
    ) -> bool:
        return is_protected_exact_match(
            is_self=is_self,
            phash_sim=phash_sim,
            dhash_sim=dhash_sim,
            score=score,
            query_partial_hash=query_partial_hash,
            rec_partial_hash=rec_partial_hash,
        )

    @staticmethod
    def _apply_family_mismatch_penalty(
        query_family: str,
        cand_prof: TextureProfile,
        score: float,
        cluster_group: str,
        cluster_reason: str,
        *,
        phash_sim: float = 0.0,
        patch_sim: float = 0.0,
    ) -> tuple[float, str, str]:
        from core.group_gates import family_relationship
        from core.pattern_classifier import is_confident_animal_print
        from core.taxonomy import normalize_family

        qf = normalize_family(query_family)
        cf = normalize_family(cand_prof.pattern_family or "unknown")
        if qf == cf or qf in ("unknown", "", "texture_ground", "plain"):
            return score, cluster_group, cluster_reason

        rel = family_relationship(qf, cf)
        if rel == "same":
            return score, cluster_group, cluster_reason

        penalty = 1.0
        if qf == "marble_abstract" and cf == "animal_print":
            penalty = 0.42
            cluster_group = CLUSTER_DISTANT
            cluster_reason = "Family uyuşmuyor — mermer sorgu / animal aday"
        elif qf == "floral" and cf == "animal_print":
            penalty = 0.38
            cluster_group = CLUSTER_UNRELATED
            cluster_reason = "Family uyuşmuyor — floral sorgu / animal aday"
        elif qf == "animal_print" and cf in ("floral", "marble_abstract"):
            penalty = 0.45
            cluster_group = CLUSTER_DISTANT
            cluster_reason = f"Family uyuşmuyor — animal sorgu / {cf}"

        if penalty < 1.0 and phash_sim < 0.90 and patch_sim < 0.80:
            score = score * penalty
        elif rel == "unrelated" and phash_sim < 0.92:
            cap = min(0.68, 0.30 + phash_sim * 0.35 + patch_sim * 0.15)
            if score > cap:
                score = cap
                if not cluster_group:
                    cluster_group = CLUSTER_UNRELATED if cap < 0.55 else CLUSTER_DISTANT
                    cluster_reason = f"Farklı desen ailesi — {qf} / {cf}"
        elif rel == "related" and phash_sim < 0.88:
            score = min(score, 0.72)

        return score, cluster_group, cluster_reason

    @staticmethod
    def _result_layer(
        cluster_group: str,
        *,
        protected_exact: bool = False,
        is_self: bool = False,
        pattern_family_score: float = 0.0,
    ) -> str:
        if is_self or protected_exact or cluster_group in (
            "exact_same",
            "format_variant",
            "resolution_variant",
            "crop_variant",
        ):
            return "same_files"
        if cluster_group in ("color_variant", "same_family_close"):
            return "same_pattern_family"
        if cluster_group in ("same_family_style", "related_family"):
            return "similar_patterns"
        if pattern_family_score >= 0.55:
            return "similar_patterns"
        return "other_results"

    @staticmethod
    def _family_explanations(
        *,
        same_family: bool,
        same_collection: bool,
        same_series: bool,
        same_designer_style: bool,
        dna_sim: float,
        semantic_sim: float,
        texture_sim: float,
        repeat_sim: float,
        motif_sim: float,
        structure_score: float,
        patch_sim: float,
        multi_scale_patch: float,
        dna_match_fields: dict[str, bool],
    ) -> list[str]:
        reasons: list[str] = []
        if same_collection:
            reasons.append("Aynı koleksiyon")
        if same_series:
            reasons.append("Aynı seri")
        if same_designer_style:
            reasons.append("Aynı tasarım dili")
        if same_family:
            reasons.append("Aynı desen ailesi")
        if repeat_sim >= 0.60 or dna_match_fields.get("Repeat"):
            reasons.append("Aynı repeat yapısı")
        if motif_sim >= 0.55 or dna_match_fields.get("Motif"):
            reasons.append("Aynı motif kullanılıyor")
        if texture_sim >= 0.55 or dna_match_fields.get("Texture"):
            reasons.append("Texture çok benzer")
        if semantic_sim >= 0.55 or dna_match_fields.get("Semantic"):
            reasons.append("Semantik yapı benzer")
        if structure_score >= 0.60 or dna_match_fields.get("Structure"):
            reasons.append("Çizim stili / yapı benzer")
        if dna_sim >= 0.62:
            reasons.append("Aynı Pattern DNA")
        if max(patch_sim, multi_scale_patch) >= 0.70:
            reasons.append("Patch yerleşimi benzer")
        return reasons[:6]

    @staticmethod
    def _variant_badge(
        *,
        is_self: bool,
        protected_exact: bool,
        cluster_group: str,
        same_family: bool,
        color_variant: bool,
        repeat_sim: float,
        motif_sim: float,
        semantic_sim: float,
        texture_sim: float,
        patch_sim: float,
        multi_scale_patch: float,
        global_score: float,
        pattern_family_score: float,
    ) -> str:
        if is_self or protected_exact or cluster_group == "exact_same":
            return "EXACT"
        if cluster_group == "crop_variant":
            return "CROP VARIANT"
        if cluster_group in ("format_variant", "resolution_variant"):
            return "SCALE VARIANT"
        if color_variant:
            return "COLOR VARIANT"
        if same_family and repeat_sim >= 0.70:
            return "REPEAT VARIANT"
        if same_family and patch_sim >= 0.86 and multi_scale_patch >= 0.84 and global_score < 0.80:
            return "SCALE VARIANT"
        if same_family and patch_sim >= 0.82 and semantic_sim >= 0.55 and global_score < 0.78:
            return "ROTATION VARIANT"
        if same_family and patch_sim >= 0.80 and texture_sim >= 0.72 and global_score < 0.74:
            return "MIRROR VARIANT"
        if same_family and (motif_sim >= 0.55 or semantic_sim >= 0.55):
            return "STYLE VARIANT"
        # `pattern_family_score` görsel/doku benzerliğidir; family kimliği değildir.
        # Bu yüzden tek başına SAME/RELATED FAMILY rozeti üretemez.
        if same_family:
            return "SAME FAMILY"
        if cluster_group in ("same_family_style", "related_family"):
            return "RELATED FAMILY"
        if texture_sim >= 0.55:
            return "SIMILAR TEXTURE"
        return "OTHER RESULTS"

    @staticmethod
    def _field_terms(value: Any) -> set[str]:
        if not value:
            return set()
        if isinstance(value, dict):
            out: set[str] = set()
            for v in value.values():
                out.update(SearchEngine._field_terms(v))
            return out
        if isinstance(value, (list, tuple, set)):
            out: set[str] = set()
            for item in value:
                out.update(SearchEngine._field_terms(item))
            return out
        text = str(value).strip().lower().replace("_", " ")
        return {text} if text and text != "unknown" else set()

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    @staticmethod
    def _pattern_dna_similarity(query_tm: dict[str, Any], cand_tm: dict[str, Any]) -> float:
        qdna = query_tm.get("pattern_dna") if isinstance(query_tm, dict) else {}
        cdna = cand_tm.get("pattern_dna") if isinstance(cand_tm, dict) else {}
        if not isinstance(qdna, dict):
            qdna = {}
        if not isinstance(cdna, dict):
            cdna = {}
        score = 0.0
        for key, weight in {
            "family": 0.30,
            "subfamily": 0.16,
            "collection": 0.16,
            "series": 0.12,
            "motif": 0.16,
            "pattern_type": 0.10,
            "repeat_type": 0.10,
            "style": 0.08,
            "designer_style": 0.08,
            "brand_style": 0.06,
            "material_hint": 0.04,
            "repeat_class": 0.05,
            "texture_class": 0.05,
            "motif_class": 0.05,
            "complexity": 0.04,
            "visual_signature": 0.06,
        }.items():
            qa = SearchEngine._field_terms(qdna.get(key) or query_tm.get(key))
            ca = SearchEngine._field_terms(cdna.get(key) or cand_tm.get(key))
            if qa and ca:
                score += weight * SearchEngine._jaccard(qa, ca)
        qtags = SearchEngine._field_terms(qdna.get("semantic_tags"))
        ctags = SearchEngine._field_terms(cdna.get("semantic_tags"))
        if qtags and ctags:
            score = max(score, 0.70 * score + 0.30 * SearchEngine._jaccard(qtags, ctags))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _dna_match_fields(query_tm: dict[str, Any], cand_tm: dict[str, Any]) -> dict[str, bool]:
        qdna = query_tm.get("pattern_dna") if isinstance(query_tm, dict) else {}
        cdna = cand_tm.get("pattern_dna") if isinstance(cand_tm, dict) else {}
        if not isinstance(qdna, dict):
            qdna = {}
        if not isinstance(cdna, dict):
            cdna = {}
        checks = {
            "Family": ("family", "pattern_family"),
            "SubFamily": ("subfamily", "pattern_subtype"),
            "Collection": ("collection", ""),
            "Series": ("series", ""),
            "Motif": ("motif", ""),
            "Repeat": ("repeat_type", ""),
            "Style": ("style", ""),
            "Designer Style": ("designer_style", ""),
            "Pattern Type": ("pattern_type", "pattern_type"),
            "Brand Style": ("brand_style", ""),
            "Material Hint": ("material_hint", ""),
            "Repeat Class": ("repeat_class", ""),
            "Texture Class": ("texture_class", ""),
            "Motif Class": ("motif_class", ""),
            "Texture": ("texture", "texture_family"),
            "Complexity": ("complexity", ""),
            "Density": ("density", ""),
            "Color Style": ("color_family", "color_family"),
            "Semantic": ("semantic_tags", "semantic_tags"),
            "Visual Signature": ("visual_signature", ""),
            "Structure": ("structure", ""),
        }
        out: dict[str, bool] = {}
        for label, (dna_key, tm_key) in checks.items():
            q_terms = SearchEngine._field_terms(qdna.get(dna_key) or query_tm.get(tm_key))
            c_terms = SearchEngine._field_terms(cdna.get(dna_key) or cand_tm.get(tm_key))
            out[label] = bool(q_terms and c_terms and q_terms & c_terms)
        return out

    @staticmethod
    def _semantic_tag_similarity(query_tm: dict[str, Any], cand_tm: dict[str, Any]) -> float:
        qsem = query_tm.get("semantic_tags") if isinstance(query_tm, dict) else {}
        csem = cand_tm.get("semantic_tags") if isinstance(cand_tm, dict) else {}
        if not isinstance(qsem, dict):
            qsem = {}
        if not isinstance(csem, dict):
            csem = {}
        score = SearchEngine._jaccard(
            SearchEngine._field_terms(qsem),
            SearchEngine._field_terms(csem),
        )
        q_family = SearchEngine._field_terms(qsem.get("family") or query_tm.get("pattern_family"))
        c_family = SearchEngine._field_terms(csem.get("family") or cand_tm.get("pattern_family"))
        if q_family and c_family and q_family == c_family:
            score = max(score, 0.65)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _motif_similarity(query_tm: dict[str, Any], cand_tm: dict[str, Any]) -> float:
        qsem = query_tm.get("semantic_tags") if isinstance(query_tm, dict) else {}
        csem = cand_tm.get("semantic_tags") if isinstance(cand_tm, dict) else {}
        if not isinstance(qsem, dict):
            qsem = {}
        if not isinstance(csem, dict):
            csem = {}
        qdna = query_tm.get("pattern_dna") if isinstance(query_tm.get("pattern_dna"), dict) else {}
        cdna = cand_tm.get("pattern_dna") if isinstance(cand_tm.get("pattern_dna"), dict) else {}
        return SearchEngine._jaccard(
            SearchEngine._field_terms([qsem.get("motif"), qsem.get("motifs"), qdna.get("motif")]),
            SearchEngine._field_terms([csem.get("motif"), csem.get("motifs"), cdna.get("motif")]),
        )

    @staticmethod
    def _repeat_structure_similarity(query_tm: dict[str, Any], cand_tm: dict[str, Any]) -> float:
        def _f(tm: dict[str, Any], key: str) -> float:
            try:
                return float(tm.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        qsem = query_tm.get("semantic_tags") if isinstance(query_tm.get("semantic_tags"), dict) else {}
        csem = cand_tm.get("semantic_tags") if isinstance(cand_tm.get("semantic_tags"), dict) else {}
        qdna = query_tm.get("pattern_dna") if isinstance(query_tm.get("pattern_dna"), dict) else {}
        cdna = cand_tm.get("pattern_dna") if isinstance(cand_tm.get("pattern_dna"), dict) else {}
        repeat_label = SearchEngine._jaccard(
            SearchEngine._field_terms(qdna.get("repeat_type") or qsem.get("repeat_type")),
            SearchEngine._field_terms(cdna.get("repeat_type") or csem.get("repeat_type")),
        )
        density = 1.0 - min(1.0, abs(_f(query_tm, "repeat_density") - _f(cand_tm, "repeat_density")))
        stripe = 1.0 - min(1.0, abs(_f(query_tm, "stripe_score") - _f(cand_tm, "stripe_score")))
        scale = 1.0 - min(1.0, abs(_f(query_tm, "scale_pattern_score") - _f(cand_tm, "scale_pattern_score")))
        organic = 1.0 - min(1.0, abs(_f(query_tm, "organic_blob_score") - _f(cand_tm, "organic_blob_score")))
        has_numeric = any(
            _f(query_tm, k) or _f(cand_tm, k)
            for k in ("repeat_density", "stripe_score", "scale_pattern_score", "organic_blob_score")
        )
        numeric = 0.35 * density + 0.25 * stripe + 0.20 * scale + 0.20 * organic
        return max(repeat_label, numeric if has_numeric else 0.0)

    @staticmethod
    def _patch_similarity(
        query_patches: list[dict[str, Any]],
        candidate_patches: list[dict[str, Any]],
    ) -> tuple[float, float]:
        if not query_patches or not candidate_patches:
            return 0.0, 0.0
        return multiscale_patch_similarity(
            query_patches,
            candidate_patches,
            phash_similarity,
            FeatureExtractor.texture_similarity,
            SearchEngine._mean_color_similarity,
        )

    @staticmethod
    def _mean_color_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != 3 or len(b) != 3:
            return 0.0
        av = np.array(a, dtype=np.float32)
        bv = np.array(b, dtype=np.float32)
        dist = float(np.linalg.norm(av - bv))
        return max(0.0, 1.0 - dist / 441.7)

    @staticmethod
    def _filename_family_score(text_query: str, rec: dict[str, Any]) -> float:
        """Geriye uyumluluk — sadece dosya adı token boost."""
        return filename_text_boost_only(text_query, rec)

    @staticmethod
    def _tier_a_score(phash: float, dhash: float, whash: float) -> float:
        return (
            0.40 * phash + 0.25 * dhash + 0.15 * whash + 0.20 * max(phash, dhash, whash)
        )

    def _tier_b_score(
        self,
        breakdown: dict[str, float],
        dino: float,
        clip: float,
        faiss_boost: float,
    ) -> float:
        parts = []
        if self.settings.search_texture:
            parts.append(breakdown["texture"] * 0.4)
        if self.settings.search_color:
            parts.append(breakdown["color"] * 0.3)
        if self.settings.search_visual:
            parts.extend([dino * 0.15, clip * 0.15, faiss_boost])
        return sum(parts) / max(len(parts), 1) if parts else 0.0

    @staticmethod
    def _categorize(score: float, is_color_variant: bool, is_self: bool) -> str:
        if is_self or score >= 0.95:
            return CATEGORY_EXACT
        if is_color_variant and score >= 0.55:
            return CATEGORY_COLOR_VARIANT
        if score >= 0.85:
            return CATEGORY_NEAR
        if score >= 0.70:
            return CATEGORY_SIMILAR
        if score >= 0.50:
            return CATEGORY_STYLE
        return CATEGORY_WEAK

    def _to_result(
        self,
        rec: dict[str, Any],
        score: float,
        breakdown: dict[str, float],
    ) -> SearchResult:
        thumb = str(rec.get("thumbnail_path") or "").strip()
        fp = str(rec.get("feature_preview_path") or "").strip()
        from core.thumb_resolve import resolve_thumb_path

        resolved, status = resolve_thumb_path(
            thumb,
            cache_dir=str(self.settings.cache_dir or ""),
            feature_preview_path=fp,
        )
        if resolved:
            thumb = resolved
        elif thumb and not os.path.isfile(thumb):
            # Keep original DB path for scheduler recovery; also try feature preview path
            if fp and os.path.isfile(fp):
                thumb = fp
        return SearchResult(
            file_id=rec["id"],
            path=rec["path"],
            filename=rec["filename"],
            customer=rec.get("customer", ""),
            thumbnail_path=thumb,
            score=score,
            score_percent=round(score * 100, 1),
            feature_preview_path=fp,
            width=rec.get("width", 0),
            height=rec.get("height", 0),
            file_size=rec.get("file_size", 0),
            mtime=rec.get("mtime", 0),
            breakdown=breakdown,
            source_name=rec.get("source_name", "") or "",
            source_type=rec.get("source_type", "") or "",
        )

    def find_exact_duplicates(self, partial_hash: str) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE partial_hash = ? AND status='indexed'",
                (partial_hash,),
            ).fetchall()
            return [dict(r) for r in rows]
