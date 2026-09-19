"""Uygulama ayarları — ilk açılışta otomatik oluşturulur."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from plugins.registry import BUILTIN_PLUGIN_EXTENSIONS

APP_NAME = "Vezir Pattern Search"
APP_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_CONFIG_PATH = DEFAULT_DATA_DIR / "settings.json"
logger = logging.getLogger(__name__)

# CAD / plotter — DXF+HPGL gerçek render; DWG/DGN discovery'de dürüst unsupported
CAD_INDEX_EXTENSIONS = frozenset({".dxf", ".plt", ".hpgl", ".dwg", ".dgn"})
# pyembroidery stitch readers + proprietary (EMB/OFM = unsupported, sahte preview yok)
EMBROIDERY_INDEX_EXTENSIONS = frozenset(
    {
        ".dst",
        ".pes",
        ".pec",
        ".exp",
        ".jef",
        ".vp3",
        ".xxx",
        ".sew",
        ".hus",
        ".shv",
        ".dsb",
        ".dsz",
        ".tap",
        ".tbf",
        ".ksm",
        ".pcd",
        ".pcm",
        ".pcs",
        ".stc",
        ".zhs",
        ".pmv",
        ".emd",
        ".u01",
        ".bro",
        ".jpx",
        ".pcq",
        ".phb",
        ".phc",
        ".emb",
        ".ofm",
        ".art",
        ".vip",
        ".csd",
        ".ngs",
        ".wings",
    }
)

# Tekstil / desen görselleri — taranan uzantılar (görsel üretilebilen + dürüst unsupported)
SUPPORTED_EXTENSIONS = (
    frozenset(
        {
            ".tif",
            ".tiff",
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".psd",
            ".ai",
            ".eps",
            ".pdf",
            ".cdr",
        }
    )
    | BUILTIN_PLUGIN_EXTENSIONS
    | CAD_INDEX_EXTENSIONS
    | EMBROIDERY_INDEX_EXTENSIONS
)

# Önizleme üretimi gereken (vektör / katmanlı / belge) formatlar
OPTIONAL_PREVIEW_EXTENSIONS = frozenset(
    {".psd", ".ai", ".eps", ".pdf", ".svg", ".cdr", ".dxf", ".plt", ".hpgl"}
)


@dataclass
class AppSettings:
    """Kalıcı uygulama ayarları."""

    archive_root: str = ""
    cache_dir: str = str(DEFAULT_CACHE_DIR)
    db_path: str = str(DEFAULT_DATA_DIR / "patterns.db")
    face_db_path: str = str(DEFAULT_DATA_DIR / "face_index.db")
    face_index_enabled: bool = False
    object_db_path: str = str(DEFAULT_DATA_DIR / "object_index.db")
    object_index_enabled: bool = False
    # Archive Intelligence — soft self-audit (sidecar DB). Never blocks search/index.
    archive_intelligence_enabled: bool = True
    archive_intelligence_db_path: str = str(
        DEFAULT_DATA_DIR / "archive_intelligence.db"
    )
    # Query-time Grounding DINO. Off by default. Never rebuilds indexes.
    open_vocab_object_enabled: bool = False
    ovd_candidate_limit: int = 12
    ovd_min_confidence: float = 0.35
    ovd_max_box_area: float = 0.85
    # Isolated motif boxes (not patterns.db / FAISS). Read when file exists.
    textile_motif_db_path: str = str(
        DEFAULT_DATA_DIR / "search_memory" / "textile_motif_evidence.db"
    )
    object_auto_scan_on_startup: bool = False
    # Spatial Intelligence V1 — geometry thresholds (normalized image coords 0–1).
    spatial_near_center_ratio: float = 0.28
    spatial_near_bbox_ratio: float = 0.12
    spatial_far_center_ratio: float = 0.42
    spatial_size_k: float = 1.25
    spatial_axis_center_min: float = 0.04
    spatial_overlap_iou_min: float = 0.02
    spatial_inside_ratio: float = 0.85
    spatial_center_lo: float = 0.33
    spatial_center_hi: float = 0.67
    face_identity_threshold: float = 0.56
    face_identity_min_margin: float = 0.035
    face_identity_engine_version: int = 11
    faiss_dino_path: str = str(DEFAULT_DATA_DIR / "faiss_dino.index")
    faiss_clip_path: str = str(DEFAULT_DATA_DIR / "faiss_clip.index")
    thumbnail_max_edge: int = 256
    feature_preview_max_edge: int = 1024
    thumbnail_format: str = "webp"  # webp | jpg
    worker_count: int = 4
    use_gpu: bool = False
    ocr_enabled: bool = False
    # AI_FINAL sonrası ayrı kuyruklar (arama oturumunda başlamaz).
    auto_patch_after_ai_final: bool = True
    auto_ocr_after_ai_final: bool = True
    ai_embedding_enabled: bool = True
    ai_search_min_vectors: int = 500
    fast_hash_only: bool = False
    use_full_hash: bool = False
    network_timeout_sec: int = 30
    # V3 hafif index (thumb/preview/metadata) için sabit güvenli üst sınır.
    # NAS/TIFF normal okumaları 3-7 sn olabildiği için 8 sn gibi agresif bir
    # değer retry fırtınası oluşturur; production baseline 30 sn'dir.
    light_decode_timeout_sec: int = 30
    network_retry_count: int = 3
    search_result_limit: int = 0  # 0 = sınırsız; UI sanal kaydırmayla yükler
    similarity_threshold: float = 0.60
    color_weight_mode: str = "ignore"  # normal | ignore | important
    search_mode: str = "style"  # exact | similar | style | comprehensive
    search_preset: str = "texture"
    comprehensive_threshold: float = 0.50
    show_unrelated_results: bool = False
    show_near_below_threshold: bool = True

    # Büyük arşiv aday daraltma
    candidate_prefilter_enabled: bool = True
    prefilter_min_index_size: int = 500
    prefilter_faiss_top_k: int = 800
    prefilter_hash_bucket_bits: int = 12

    # Skor ağırlıkları
    weight_dino: float = 0.35
    weight_clip: float = 0.25
    weight_phash: float = 0.20
    weight_color: float = 0.10
    weight_texture: float = 0.10

    customer_filter: str = ""  # boş = tüm müşteriler

    # Çok kaynaklı index
    search_scope: str = "all"
    selected_source_id: int = 0
    selected_source_ids: list[int] = field(default_factory=list)
    selected_folder_path: str = ""
    quick_search_folder: str = ""
    auto_scan_on_startup: bool = False
    resume_index_on_startup: bool = True
    index_background_low_priority: bool = True
    # background | balanced | max_speed — varsayılan PC'yi kilitlemez
    work_profile: str = "background"
    resource_monitor_enabled: bool = True
    production_recovery_enabled: bool = True
    cache_max_mb: int = 4096
    index_throttle_on_search: bool = True
    quick_scan_interval_hours: int = 6
    deep_scan_interval_days: int = 7

    # Index hızı — AI aramada açık kalabilir, index sırasında kapalı (çok daha hızlı)
    index_skip_ai: bool = True
    index_light_first: bool = True  # önce thumb+hash+doku; AI/OCR/patch sonra
    index_defer_preview: bool = True  # hızlı taramada 1024px önizlemeyi ertele
    index_heavy_after_light: bool = False  # hafif index bitince AI/OCR kuyruğunu işle
    index_drain_backlog_first: bool = True
    index_backlog_skip_scan_threshold: int = 200
    index_scope: str = "selected_sources"  # all | selected_sources | single_source
    index_worker_count: int = 4
    heavy_index_worker_count: int = 2
    preview_worker_count: int = 4
    ai_worker_count: int = 2
    embedding_batch_size: int = 8
    max_network_workers: int = 2
    index_skip_reanalyse_dna: bool = True
    index_skip_reanalyse_semantic: bool = True
    index_skip_quarantine_retry: bool = True
    index_skip_heavy_formats: bool = True
    fast_tif_defer_mb: int = 64
    index_mode: str = "standard"  # fast_archive | night_complete | backfill | standard

    # Arama öncesi hızlı index
    auto_preflight_index_on_search: bool = True
    preflight_batch_size: int = 300
    preflight_folder_siblings: bool = True
    preflight_reindex_query_if_incomplete: bool = True
    preflight_query_only_on_search: bool = True

    # Görsel arama: önce hash sonuçları, sonra tam skor + yeniden sıralama
    search_progressive: bool = True
    live_score_updates: bool = True
    ai_live_refresh: bool = True
    lock_result_ranking: bool = False
    auto_search_on_image: bool = True
    visual_search_floor: float = 0.40
    search_scope_fallback_on_empty: bool = False

    # Klasör izleme — dosya ekleme/silme sonrası hızlı tarama
    folder_watch_enabled: bool = True
    folder_watch_debounce_sec: float = 3.0

    # Diskte silinen dosyaların index kaydını tamamen kaldır
    purge_missing_after_scan: bool = True

    # Arka plan full scan: eksik gap + silinen (missing) — index drain'den bağımsız
    background_full_scan_enabled: bool = True
    background_full_scan_interval_min: int = 30  # 0 = yalnız elle

    # Arama modları
    search_visual: bool = True
    search_texture: bool = True
    search_color: bool = True
    search_text_filename: bool = True
    search_text_ocr: bool = True
    search_text_visual: bool = False  # metin aramada görsel ağırlığı düşük
    search_method_mode: str = "hybrid"
    semantic_pattern_intel_enabled: bool = True
    pattern_intelligence_v2_enabled: bool = True
    universal_visual_intel_enabled: bool = True

    # Global nesne zekâsı (opsiyonel / güvenli geri dönüşlü). Mevcut desen aramasını bozmaz.
    global_object_intelligence_enabled: bool = True
    global_object_detection_min_confidence: float = 0.45
    global_object_max_detections: int = 50

    # AI / OCR / fine detail / panel görünürlüğü
    semantic_text_search_enabled: bool = True
    fine_detail_enabled: bool = True
    fine_detail_top_k: int = 80
    format_panel_enabled: bool = True
    category_tree_panel_enabled: bool = True

    # UI düzeni
    ui_window_width: int = 1600
    ui_window_height: int = 920
    ui_splitter_main: list[int] = field(default_factory=lambda: [620, 360])
    ui_splitter_center_v: list[int] = field(default_factory=lambda: [300, 650])
    ui_left_visible: bool = True
    ui_right_visible: bool = True
    results_view_mode: str = "card"
    ui_mode: str = "simple"
    user_role: str = "admin"
    user_name: str = "local_admin"
    ui_state: dict[str, Any] = field(default_factory=dict)

    def index_uses_ai(self) -> bool:
        """Index sırasında DINO/CLIP hesapla (yavaş, tek worker)."""
        return bool(self.ai_embedding_enabled and not self.index_skip_ai)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings":
        from core.path_safety import sanitize_settings_paths

        config_path = path or DEFAULT_CONFIG_PATH
        config_path = Path(config_path)
        logger.info("Loading settings: %s", config_path)

        def _create_default_settings() -> "AppSettings":
            settings = cls()
            settings.ensure_dirs()
            settings.save(config_path)
            return settings

        def _backup_corrupted_file() -> None:
            try:
                bad_path = config_path.with_suffix(".bad.json")
                bad_path.parent.mkdir(parents=True, exist_ok=True)
                bad_path.write_bytes(config_path.read_bytes())
            except OSError:
                logger.warning("Could not backup corrupted settings: %s", config_path)

        if not config_path.exists():
            logger.warning("Settings file missing. Creating default settings...")
            return _create_default_settings()
        try:
            if config_path.stat().st_size == 0:
                logger.warning("Settings file empty. Creating default settings...")
                return _create_default_settings()
        except OSError:
            logger.warning("Settings file not readable. Using defaults: %s", config_path)
            settings = cls()
            settings.ensure_dirs()
            return settings
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error("Settings file corrupted. Creating default settings...")
            _backup_corrupted_file()
            return _create_default_settings()
        except OSError:
            logger.warning("Settings file could not be opened. Using defaults: %s", config_path)
            settings = cls()
            settings.ensure_dirs()
            return settings
        settings = cls()
        for key, value in data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        # Face Intelligence V9 migration: old V8 configs commonly retained
        # the 0.62 threshold even after the engine became exemplar-first.
        if int(data.get("face_identity_engine_version", 0) or 0) < 11:
            settings.face_identity_threshold = 0.56
            settings.face_identity_min_margin = 0.035
            settings.face_identity_engine_version = 11
            try:
                settings.save(config_path)
            except Exception:
                logger.debug("Face Intelligence ayar göçü kaydedilemedi", exc_info=True)

        if "search_preset" not in data:
            settings.search_preset = "texture"
            settings.search_mode = "style"
            settings.color_weight_mode = "ignore"
            settings.similarity_threshold = 0.60
            settings.fast_hash_only = False
            settings.search_scope = "all"
        if sanitize_settings_paths(settings):
            settings.save(config_path)
        settings.ensure_dirs()
        return settings

    def save(self, path: Path | None = None) -> None:
        from core.path_safety import is_temporary_test_path

        config_path = path or DEFAULT_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # Test ortamında gerçek settings dosyasına test yolu yazma
        if path is None or path == DEFAULT_CONFIG_PATH:
            for key in (
                "db_path",
                "cache_dir",
                "faiss_dino_path",
                "faiss_clip_path",
                "archive_root",
            ):
                if is_temporary_test_path(data.get(key, "")):
                    defaults = AppSettings()
                    data[key] = getattr(defaults, key)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            # A running Windows instance can briefly lock settings.json.
            # Preserve a recovery copy without crashing interactive controls.
            fallback = (
                Path(tempfile.gettempdir()) / "vezir_pattern_search_settings.json"
            )
            try:
                with open(fallback, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except OSError:
                pass

    def ensure_dirs(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.cache_dir) / "thumbnails").mkdir(parents=True, exist_ok=True)
        (Path(self.cache_dir) / "feature_previews").mkdir(parents=True, exist_ok=True)
        (Path(self.cache_dir) / "embeddings").mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.face_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.object_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.faiss_dino_path).parent.mkdir(parents=True, exist_ok=True)

    def effective_weights(self) -> dict[str, float]:
        """Renk moduna göre ağırlıkları ayarla."""
        w = {
            "dino": self.weight_dino,
            "clip": self.weight_clip,
            "phash": self.weight_phash,
            "color": self.weight_color,
            "texture": self.weight_texture,
        }
        if self.color_weight_mode == "ignore":
            freed = w.pop("color", 0)
            total = sum(w.values()) or 1.0
            for k in w:
                w[k] += freed * (w[k] / total)
        elif self.color_weight_mode == "important":
            w["color"] = min(0.30, w["color"] * 2)
            total = sum(w.values()) or 1.0
            w = {k: v / total for k, v in w.items()}
        return w

    def classical_score(
        self,
        phash_sim: float,
        dhash_sim: float,
        whash_sim: float,
        texture_sim: float,
        color_sim: float,
    ) -> float:
        """AI kapalıyken klasik ağırlıklı skor."""
        if self.color_weight_mode == "ignore":
            score = (
                0.42 * phash_sim
                + 0.28 * dhash_sim
                + 0.18 * whash_sim
                + 0.12 * texture_sim
            )
        else:
            score = (
                0.40 * phash_sim
                + 0.25 * dhash_sim
                + 0.15 * whash_sim
                + 0.10 * texture_sim
                + 0.10 * color_sim
            )
        if phash_sim >= 0.98:
            score = max(score, 0.98)
        return score
