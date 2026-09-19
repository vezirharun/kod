"""Embedding / OCR / full_done durum teşhisi ve UI etiketleri."""

from __future__ import annotations

from typing import Any

from core.capability_check import probe_ai_dependencies, probe_ocr_dependencies
from core.settings import AppSettings

FULL_DONE_CRITERIA = (
    "index_stage='full_done' veya heavy_status='done' — "
    "güçlü analiz oturumu tamamlandı. "
    "Embedding ve OCR zorunlu değil; ayarlara bağlı."
)


def build_embedding_status(
    settings: AppSettings,
    dashboard: dict[str, int],
    embedding_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    ready = int(dashboard.get("embedding_ready", 0) or 0)
    dino = int((embedding_counts or {}).get("dino", 0) or 0)
    clip = int((embedding_counts or {}).get("clip", 0) or 0)
    patch = int(dashboard.get("patch_embedding_ready", 0) or 0)

    if not settings.ai_embedding_enabled:
        return {
            "status": "disabled",
            "label": "Kapalı — ayarlarda AI embedding kapalı",
            "ready": ready,
            "dino": dino,
            "clip": clip,
            "patch": patch,
            "reason": "settings.ai_embedding_enabled=false",
        }
    if settings.index_skip_ai:
        return {
            "status": "skipped",
            "label": "Atlandı — index modu AI embedding üretmiyor",
            "ready": ready,
            "dino": dino,
            "clip": clip,
            "patch": patch,
            "reason": "index_skip_ai=true",
        }
    ok, msg = probe_ai_dependencies()
    if not ok:
        return {
            "status": "model_missing",
            "label": "Model yok — DINO/CLIP yüklenemedi",
            "ready": ready,
            "dino": dino,
            "clip": clip,
            "patch": patch,
            "reason": msg or "ai_model_missing",
        }
    return {
        "status": "active",
        "label": f"Aktif — {ready:,} kayıt (dino {dino:,} · clip {clip:,})",
        "ready": ready,
        "dino": dino,
        "clip": clip,
        "patch": patch,
        "reason": "",
    }


def build_ocr_status(settings: AppSettings, dashboard: dict[str, int]) -> dict[str, Any]:
    ready = int(dashboard.get("ocr_ready", 0) or 0)
    if not settings.ocr_enabled:
        return {
            "status": "disabled",
            "label": "Kapalı — ayarlarda OCR kapalı",
            "ready": ready,
            "reason": "settings.ocr_enabled=false",
        }
    ok, msg = probe_ocr_dependencies()
    if not ok:
        return {
            "status": "dependency_missing",
            "label": "Dependency eksik — EasyOCR/Tesseract yok",
            "ready": ready,
            "reason": msg or "ocr_dependency_missing",
        }
    if ready == 0:
        return {
            "status": "active_empty",
            "label": "Aktif — henüz metin bulunamadı / OCR yapılmadı",
            "ready": ready,
            "reason": "no_ocr_text_yet",
        }
    return {
        "status": "active",
        "label": f"Aktif — {ready:,} kayıtta OCR metni",
        "ready": ready,
        "reason": "",
    }


def build_full_done_status(dashboard: dict[str, int]) -> dict[str, Any]:
    full_stage = int(dashboard.get("full_done_stage", 0) or 0)
    heavy_done = int(dashboard.get("heavy_done", 0) or 0)
    return {
        "full_done_stage": full_stage,
        "heavy_done": heavy_done,
        "criteria": FULL_DONE_CRITERIA,
        "label": (
            f"Full done (stage): {full_stage:,} · Heavy done (DB): {heavy_done:,}"
        ),
    }


def format_pipeline_dashboard_log(dashboard: dict[str, int]) -> str:
    keys = (
        "preview_ready",
        "embedding_ready",
        "patch_embedding_ready",
        "deep_content_ready",
        "semantic_tag_ready",
        "pattern_dna_ready",
        "ocr_ready",
        "texture_ready",
        "full_done_stage",
        "heavy_done",
    )
    parts = [f"{k}={int(dashboard.get(k, 0) or 0):,}" for k in keys]
    return "Pipeline dashboard SQL: " + " | ".join(parts)
