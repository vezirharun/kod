"""Preview-first index durum özeti."""

from __future__ import annotations

from core.quarantine import reason_label


def build_pipeline_status(
    dashboard: dict[str, int],
    *,
    archive_total: int,
    quarantine_by_reason: dict[str, int] | None = None,
) -> dict:
    total = int(dashboard.get("total_searchable", 0) or 0)
    # UI ilerleme = bağımsız stage_* (event/önkoşul yığını değil)
    preview_ready = int(
        dashboard.get("stage_preview", dashboard.get("preview_ready", 0)) or 0
    )
    dino_ready = int(dashboard.get("stage_dino", 0) or 0)
    clip_ready = int(dashboard.get("stage_clip", 0) or 0)
    embedding_ready = int(
        dashboard.get("embedding_ready", 0)
        or (min(dino_ready, clip_ready) if dino_ready and clip_ready else 0)
        or 0
    )
    ai_ready = int(dashboard.get("ai_analyzed_ready", 0) or 0)
    dna_ready = int(
        dashboard.get("stage_dna", dashboard.get("pattern_dna_ready", 0)) or 0
    )
    sem_ready = int(
        dashboard.get("stage_semantic", dashboard.get("semantic_tag_ready", 0))
        or 0
    )
    ocr_ready = int(dashboard.get("stage_ocr", dashboard.get("ocr_ready", 0)) or 0)
    texture_ready = int(
        dashboard.get("stage_texture", dashboard.get("texture_ready", 0)) or 0
    )
    patch_ready = int(
        dashboard.get("stage_patch", dashboard.get("patch_embedding_ready", 0))
        or 0
    )
    ai_final_ready = int(dashboard.get("ai_final_ready", 0) or 0)

    def pct(ready: int) -> int:
        if total <= 0:
            return 0
        return int(ready / total * 100)

    reasons = quarantine_by_reason or {}
    reason_lines = [
        f"  • {reason_label(k)}: {v:,}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])
    ]
    quarantine_summary = "\n".join(reason_lines) if reason_lines else "  —"

    return {
        **dashboard,
        "archive_total": archive_total,
        "total": total,
        "preview_ready": preview_ready,
        "embedding_ready": embedding_ready,
        "db_dino_embeddings": dino_ready,
        "db_clip_embeddings": clip_ready,
        "patch_embedding_ready": patch_ready,
        "ai_final_ready": ai_final_ready,
        "ai_analyzed": ai_ready,
        "pattern_dna_count": dna_ready,
        "semantic_tag_count": sem_ready,
        "ocr_done": ocr_ready,
        "texture_done": texture_ready,
        "pct_preview": pct(preview_ready),
        "pct_embedding": pct(embedding_ready),
        "pct_dino": pct(dino_ready),
        "pct_clip": pct(clip_ready),
        "pct_patch": pct(patch_ready),
        "pct_ai_final": pct(ai_final_ready),
        "pct_ai_analysis": pct(ai_ready),
        "pct_pattern_dna": pct(dna_ready),
        "pct_semantic_tag": pct(sem_ready),
        "pct_ocr": pct(ocr_ready),
        "pct_texture": pct(texture_ready),
        "pending_preview": max(0, total - preview_ready),
        "pending_embedding": max(0, total - embedding_ready),
        "pending_dino": max(0, total - dino_ready),
        "pending_clip": max(0, total - clip_ready),
        "pending_patch": max(0, total - patch_ready),
        "pending_ai_final": max(0, total - ai_final_ready),
        "pending_ai_analysis": int(dashboard.get("pending_ai_analysis", 0) or 0),
        "pending_pattern_dna": max(0, total - dna_ready),
        "pending_semantic_tag": max(0, total - sem_ready),
        "pending_ocr": max(0, total - ocr_ready),
        "pending_texture": max(0, total - texture_ready),
        "quarantine_total": int(dashboard.get("quarantine_total", 0) or 0),
        "quarantine_by_reason": reasons,
        "quarantine_summary": quarantine_summary,
        "unsupported_format": int(dashboard.get("unsupported_format", 0) or 0),
        "dependency_missing": int(dashboard.get("dependency_missing", 0) or 0),
    }
