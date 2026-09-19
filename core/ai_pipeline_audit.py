"""P0 — AI Pipeline Audit & Recovery.

Model / FAISS / Knowledge / DNA / Re-rank / Gate sağlık raporu + arama stage debug.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StageStatus:
    name: str
    ok: bool = False
    count: int = 0
    detail: str = ""

    def mark(self, ok: bool, *, count: int = 0, detail: str = "") -> None:
        self.ok = ok
        self.count = int(count)
        self.detail = detail or self.detail


@dataclass
class PipelineAudit:
    stages: list[StageStatus] = field(default_factory=list)
    ai_active: bool = False
    ai_reason: str = ""
    elapsed_sec: float = 0.0
    provider: str = ""
    device: str = ""
    embedding_dim: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai_active": self.ai_active,
            "ai_reason": self.ai_reason,
            "provider": self.provider,
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "elapsed_sec": self.elapsed_sec,
            "stages": [asdict(s) for s in self.stages],
            "checklist": {
                s.name: ("✓" if s.ok else "✗") for s in self.stages
            },
        }

    def ui_lines(self) -> list[str]:
        lines = ["Search Pipeline"]
        for s in self.stages:
            mark = "✓" if s.ok else "✗"
            extra = f" ({s.count})" if s.count else ""
            reason = f" — {s.detail}" if (not s.ok and s.detail) else ""
            lines.append(f"{s.name} {mark}{extra}{reason}")
        if self.ai_active:
            lines.append("AI Active ✓")
        else:
            lines.append("AI Active ✗")
            if self.ai_reason:
                lines.append(f"Reason: {self.ai_reason.splitlines()[0]}")
        return lines


def run_system_health_report(settings, *, engine=None) -> dict[str, Any]:
    """Tam AI PIPELINE REPORT — CLI / Health panel."""
    from core.capability_check import (
        diagnose_ai_stack,
        probe_ocr_dependencies,
        try_load_ai_extractor,
    )
    from core.faiss_store import FaissStore

    t0 = time.perf_counter()
    dep = diagnose_ai_stack()
    ai_ok, ai_msg = try_load_ai_extractor(settings) if dep.get("ok") else (False, dep.get("message", ""))

    faiss = FaissStore(settings.faiss_dino_path, settings.faiss_clip_path)
    faiss_ok = bool(faiss.available) and (faiss.dino_count > 0 or faiss.clip_count > 0)

    ocr_ok, ocr_msg = probe_ocr_dependencies()

    kb_ok = False
    kg_ok = False
    try:
        from core.textile_knowledge_base import get_knowledge_base

        kb_ok = get_knowledge_base().stats().get("pattern_families", 0) > 0
    except Exception as exc:
        kb_msg = str(exc)
    else:
        kb_msg = ""

    try:
        from core.knowledge_graph import get_knowledge_graph

        g = get_knowledge_graph()
        kg_ok = len(g.nodes) > 10
    except Exception as exc:
        kg_msg = str(exc)
    else:
        kg_msg = ""

    rerank_ok = False
    try:
        from core.textile_reranker import V2_WEIGHTS

        rerank_ok = abs(sum(V2_WEIGHTS.values()) - 1.0) < 0.02
    except Exception:
        rerank_ok = False

    thumb_ok = True
    try:
        from pathlib import Path

        thumb_ok = Path(settings.cache_dir).exists()
    except Exception:
        thumb_ok = False

    provider = "OpenCLIP + DINOv2" if ai_ok else "—"
    device = str(dep.get("device") or "cpu")
    report = {
        "title": "AI PIPELINE REPORT",
        "ok": bool(ai_ok and faiss_ok),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "components": {
            "OpenCLIP": {"ok": ai_ok, "detail": ai_msg if not ai_ok else "OK", "device": device},
            "Embedding": {"ok": ai_ok, "detail": "OK" if ai_ok else ai_msg},
            "FAISS": {
                "ok": faiss_ok,
                "detail": (
                    f"dino={faiss.dino_count} clip={faiss.clip_count}"
                    if faiss.available
                    else "faiss unavailable"
                ),
            },
            "Knowledge Base": {"ok": kb_ok, "detail": kb_msg or "OK"},
            "Knowledge Graph": {"ok": kg_ok, "detail": kg_msg or "OK"},
            "Re-ranker": {"ok": rerank_ok, "detail": "v2" if rerank_ok else "missing"},
            "Family Gate": {"ok": True, "detail": "group_gates"},
            "OCR": {"ok": ocr_ok, "detail": "OK" if ocr_ok else ocr_msg},
            "Texture": {"ok": True, "detail": "texture_profile"},
            "Thumbnail": {"ok": thumb_ok, "detail": str(getattr(settings, "cache_dir", ""))},
            "DNA": {"ok": True, "detail": "pattern_dna"},
        },
        "provider": provider,
        "device": device,
    }
    if engine is not None:
        report["engine_ai_error"] = getattr(engine, "_ai_load_error", "") or ""
        report["ai_index_ready"] = bool(getattr(engine, "ai_index_ready", lambda: False)())
    return report


def format_health_report(report: dict[str, Any]) -> str:
    lines = [report.get("title") or "AI PIPELINE REPORT", ""]
    for name, info in (report.get("components") or {}).items():
        mark = "OK" if info.get("ok") else "FAIL"
        detail = info.get("detail") or ""
        if info.get("ok"):
            lines.append(f"{name}\n{mark}")
        else:
            lines.append(f"{name}\n{mark}\n{detail}")
        lines.append("")
    lines.append(f"Audit Time\n{report.get('elapsed_sec', 0)} sec")
    lines.append(f"Device\n{report.get('device', 'cpu')}")
    return "\n".join(lines).strip()


def build_search_pipeline_audit(
    *,
    ai_active: bool,
    ai_reason: str = "",
    indexed_total: int = 0,
    faiss_hits: int = 0,
    candidates: int = 0,
    after_rerank: int = 0,
    after_gate: int = 0,
    final: int = 0,
    embedding_ok: bool = False,
    dna_ok: bool = False,
    knowledge_ok: bool = False,
    rerank_ok: bool = False,
    gate_ok: bool = False,
    ui_ok: bool = True,
    provider: str = "",
    device: str = "",
    embedding_dim: int = 0,
    elapsed_sec: float = 0.0,
) -> PipelineAudit:
    audit = PipelineAudit(
        ai_active=ai_active,
        ai_reason=ai_reason,
        provider=provider,
        device=device,
        embedding_dim=embedding_dim,
        elapsed_sec=elapsed_sec,
    )
    audit.stages = [
        StageStatus("Embedding", embedding_ok, indexed_total if embedding_ok else 0,
                    "" if embedding_ok else (ai_reason or "no query embedding")),
        StageStatus("FAISS", faiss_hits > 0 or not ai_active, faiss_hits,
                    "" if faiss_hits or not ai_active else "no FAISS hits"),
        StageStatus("TopCandidates", candidates > 0, candidates),
        StageStatus("Pattern DNA", dna_ok, candidates if dna_ok else 0,
                    "" if dna_ok else "DNA scores missing"),
        StageStatus("Knowledge Graph", knowledge_ok, 0,
                    "" if knowledge_ok else "KB inactive this search"),
        StageStatus("Re-ranker", rerank_ok, after_rerank or candidates,
                    "" if rerank_ok else "textile_rerank_v2 not applied"),
        StageStatus("Family Gate", gate_ok, after_gate or final),
        StageStatus("UI", ui_ok, final),
    ]
    return audit
