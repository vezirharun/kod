"""Entity/Object Intelligence v1 — açık kavram + kanıt kanalları.

Arama sırasında index yazmaz. CLIP kutusunu detector gibi göstermez.
Mevcut DINO/OpenCLIP/FAISS/Pattern DNA alanlarını değiştirmez.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.visual_concept import (
    VisualChannelScores,
    compile_visual_query,
    english_lemma,
    fuse_visual_channels,
    load_open_concepts,
)

# Index pipeline'ına ek detector geçişi yok.
INDEX_ENTITY_OVERHEAD_MS = 0.0

_CLIP_SOURCES = frozenset({"openclip", "clip", "open"})
_DETECTOR_SOURCES = frozenset({"object_detector", "object detector", "detector", "fusion"})


@dataclass
class EntityRecord:
    entity: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    category: str = ""
    confidence: float = 0.0
    evidence_source: str = "lexicon"
    bbox: tuple[int, int, int, int] | None = None


@dataclass
class EntityQuery:
    raw: str
    entities: list[EntityRecord] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    brands: list[str] = field(default_factory=list)
    clip_prompts: list[str] = field(default_factory=list)

    def canonical_ids(self) -> list[str]:
        return [e.canonical_name for e in self.entities]


def object_detection_status() -> dict[str, Any]:
    """Probe availability without loading YOLO/RT-DETR/CLIP as a detector."""
    from core.global_object_intelligence import _TORCHVISION_OK

    root = Path(__file__).resolve().parents[1]
    rtdetr = root / ".cache_object_photos" / "detector_weights" / "rtdetr-l.pt"
    if not rtdetr.is_file():
        rtdetr = root / "rtdetr-l.pt"
    has_rtdetr = rtdetr.is_file()
    available = bool(has_rtdetr or _TORCHVISION_OK)
    backend = "unavailable"
    if has_rtdetr:
        backend = "ultralytics_rtdetr_l_coco"
    elif _TORCHVISION_OK:
        backend = "torchvision_fasterrcnn_coco"
    return {
        "object_detection_available": available,
        "backend": backend,
        "clip_is_not_detector": True,
        "index_overhead_ms": INDEX_ENTITY_OVERHEAD_MS,
    }


def _display_tr(lemma: str) -> str:
    lemma = str(lemma or "").strip().lower()
    syn = load_open_concepts().get("synonyms") or {}
    for key, val in syn.items():
        if str(val).lower() == lemma:
            return str(key)
    return lemma


def _aliases_for(lemma: str) -> tuple[str, ...]:
    lemma = str(lemma or "").strip().lower()
    syn = load_open_concepts().get("synonyms") or {}
    found = [k for k, v in syn.items() if str(v).lower() == lemma]
    return tuple(dict.fromkeys([lemma, *found]))


def parse_entity_query(text: str) -> EntityQuery:
    raw = str(text or "").strip()
    q = EntityQuery(raw=raw)
    if not raw:
        return q
    compiled = compile_visual_query(raw)
    q.colors = list(compiled.colors)
    q.clip_prompts = list(compiled.clip_prompts)
    seen: set[str] = set()
    for c in compiled.concepts:
        lemma = str(c.concept_id.split(":", 1)[-1] or "").replace("_", " ").strip()
        if not lemma or lemma in seen:
            continue
        seen.add(lemma)
        q.entities.append(
            EntityRecord(
                entity=_display_tr(lemma),
                canonical_name=lemma,
                aliases=_aliases_for(lemma),
                category=c.category,
                confidence=1.0,
                evidence_source="lexicon",
            )
        )
    try:
        from core.brand_aliases import resolve_brand_alias
        from core.textile_terms import normalize_turkish
        import re

        for tok in re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+", raw):
            brand = resolve_brand_alias(tok) or resolve_brand_alias(normalize_turkish(tok))
            if brand and brand not in q.brands:
                q.brands.append(str(brand))
    except Exception:
        pass
    return q


def _bbox_tuple(raw: Any) -> tuple[int, int, int, int] | None:
    if not raw:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in raw[:4]]
        return (x1, y1, x2, y2)
    except (TypeError, ValueError, IndexError):
        return None


def clip_is_not_detection(source: str) -> bool:
    src = str(source or "").strip().lower()
    if src in _DETECTOR_SOURCES:
        return False
    return src in _CLIP_SOURCES or src in {"openclip", "clip_open"}


def records_from_file_meta(rec: dict[str, Any] | None) -> list[EntityRecord]:
    """Read persisted evidence only. Does not run a detector."""
    if not isinstance(rec, dict):
        return []
    tm = rec.get("texture_map") or {}
    if not isinstance(tm, dict):
        tm = {}
    out: list[EntityRecord] = []

    goi = tm.get("global_object_intelligence") or rec.get("global_object_intelligence") or {}
    if isinstance(goi, dict):
        for obj in goi.get("objects") or ():
            if not isinstance(obj, dict):
                continue
            lemma = english_lemma(str(obj.get("label") or obj.get("label_tr") or ""))
            if not lemma:
                continue
            try:
                conf = float(obj.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            out.append(
                EntityRecord(
                    entity=str(obj.get("label_tr") or _display_tr(lemma)),
                    canonical_name=lemma,
                    aliases=_aliases_for(lemma),
                    category=str((load_open_concepts().get("categories") or {}).get(lemma) or ""),
                    confidence=max(0.0, min(1.0, conf)),
                    evidence_source="object_detector",
                    bbox=_bbox_tuple(obj.get("bbox")),
                )
            )
        for concept in goi.get("concepts") or ():
            if not isinstance(concept, dict):
                continue
            lemma = english_lemma(str(concept.get("label") or concept.get("label_tr") or ""))
            if not lemma:
                continue
            try:
                conf = min(0.55, float(concept.get("confidence") or 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            out.append(
                EntityRecord(
                    entity=str(concept.get("label_tr") or _display_tr(lemma)),
                    canonical_name=lemma,
                    aliases=_aliases_for(lemma),
                    category=str((load_open_concepts().get("categories") or {}).get(lemma) or ""),
                    confidence=conf,
                    evidence_source="openclip",
                )
            )

    vcdna = tm.get("visual_concept_dna") if isinstance(tm.get("visual_concept_dna"), dict) else {}
    if not vcdna and isinstance(goi, dict) and isinstance(goi.get("visual_concept_dna"), dict):
        vcdna = goi["visual_concept_dna"]
    if isinstance(vcdna, dict):
        for row in list(vcdna.get("objects") or []) + list(vcdna.get("concepts") or []):
            if not isinstance(row, dict):
                continue
            lemma = english_lemma(str(row.get("lemma") or row.get("label") or ""))
            if not lemma:
                continue
            src = str(row.get("evidence") or "visual_concept")
            if src in _DETECTOR_SOURCES or row.get("detected"):
                src = "object_detector"
            elif src in _CLIP_SOURCES:
                src = "visual_concept"
            else:
                src = "visual_concept" if src == "visual_concept" else src
            try:
                conf = float(row.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            if src not in _DETECTOR_SOURCES:
                # Stored concept DNA is evidence; live CLIP-as-detector is not.
                conf = max(0.0, min(1.0, conf))
            out.append(
                EntityRecord(
                    entity=str(row.get("label_tr") or _display_tr(lemma)),
                    canonical_name=lemma,
                    aliases=_aliases_for(lemma),
                    category=str(row.get("category") or (load_open_concepts().get("categories") or {}).get(lemma) or ""),
                    confidence=conf,
                    evidence_source=src,
                    bbox=_bbox_tuple(row.get("bbox")),
                )
            )

    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    fam = str(dna.get("family") or dna.get("main_family") or rec.get("pattern_family") or "").lower()
    motif = str(dna.get("motif") or dna.get("motif_class") or "").lower()
    blob = f"{fam} {motif}"
    if any(x in blob for x in ("floral", "flower", "cicek", "çiçek")):
        out.append(
            EntityRecord(
                entity="çiçek",
                canonical_name="flower",
                aliases=_aliases_for("flower"),
                category="plant",
                confidence=min(0.55, float(dna.get("confidence") or dna.get("pattern_dna_confidence") or 0.4) or 0.4),
                evidence_source="pattern_dna",
            )
        )
    return out


def match_score(query: EntityQuery, file_recs: Iterable[EntityRecord]) -> tuple[float, str, list[EntityRecord]]:
    """Detector evidence outranks CLIP-only similarity. Multi-entity uses AND."""
    wanted = list(dict.fromkeys(query.canonical_ids()))
    if not wanted:
        return 0.0, "none", []
    best: dict[str, EntityRecord] = {}
    for rec in file_recs:
        name = rec.canonical_name
        if name not in wanted:
            continue
        prev = best.get(name)
        if prev is None:
            best[name] = rec
            continue
        prev_det = prev.evidence_source in _DETECTOR_SOURCES
        rec_det = rec.evidence_source in _DETECTOR_SOURCES
        if rec_det and not prev_det:
            best[name] = rec
        elif rec_det == prev_det and rec.confidence > prev.confidence:
            best[name] = rec
    hits = [best[n] for n in wanted if n in best]
    if not hits:
        return 0.0, "none", []
    all_matched = len(hits) == len(wanted)
    det_hits = [h for h in hits if h.evidence_source in _DETECTOR_SOURCES]
    if all_matched and len(det_hits) == len(wanted):
        return min(h.confidence for h in det_hits), "object_detector", hits
    if all_matched and det_hits:
        fused = 0.70 * min(h.confidence for h in det_hits) + 0.20 * min(
            (h.confidence for h in hits if h.evidence_source not in _DETECTOR_SOURCES),
            default=0.0,
        )
        return min(0.93, fused), "fusion", hits
    if all_matched:
        src = hits[0].evidence_source
        if src == "visual_concept":
            return min(h.confidence for h in hits), "visual_concept", hits
        return min(0.55, min(h.confidence for h in hits)), src, hits
    if det_hits:
        return min(0.40, min(h.confidence for h in det_hits) * (len(hits) / len(wanted))), "partial", hits
    return 0.0, "none", hits


def debug_lines(hits: list[EntityRecord]) -> list[str]:
    lines: list[str] = []
    for h in hits:
        label = (h.entity or h.canonical_name).strip()
        if not label:
            continue
        src = "RT-DETR" if h.evidence_source in _DETECTOR_SOURCES else h.evidence_source
        bit = f"{label.capitalize()} {h.confidence:.2f} ({src})"
        if h.bbox:
            x1, y1, x2, y2 = h.bbox
            bit += f" [{x1},{y1},{x2 - x1},{y2 - y1}]"
        lines.append(bit)
    return lines[:8]


def _records_from_object_index(file_id: int, object_db_path: str) -> list[EntityRecord]:
    path = str(object_db_path or "").strip()
    if int(file_id) <= 0 or not path:
        return []
    from pathlib import Path

    if not Path(path).is_file():
        return []
    try:
        from core.object_index import ObjectIndexStore

        store = ObjectIndexStore(path, readonly=True)
        out: list[EntityRecord] = []
        for inst in store.instances_for_file(int(file_id)):
            lemma = english_lemma(str(inst.get("label") or ""))
            if not lemma:
                continue
            out.append(
                EntityRecord(
                    entity=str(inst.get("label_tr") or _display_tr(lemma)),
                    canonical_name=lemma,
                    aliases=_aliases_for(lemma),
                    category=str((load_open_concepts().get("categories") or {}).get(lemma) or ""),
                    confidence=max(0.0, min(1.0, float(inst.get("confidence") or 0.0))),
                    evidence_source="object_detector",
                    bbox=_bbox_tuple(inst.get("bbox")),
                )
            )
        for c in store.concepts_for_file(int(file_id)):
            lemma = english_lemma(str(c.get("lemma") or c.get("label") or ""))
            if not lemma:
                continue
            src = "object_detector" if int(c.get("detected") or 0) else str(c.get("evidence") or "visual_concept")
            if src in _CLIP_SOURCES:
                src = "visual_concept"
            out.append(
                EntityRecord(
                    entity=str(c.get("label_tr") or _display_tr(lemma)),
                    canonical_name=lemma,
                    aliases=_aliases_for(lemma),
                    category=str(c.get("category") or (load_open_concepts().get("categories") or {}).get(lemma) or ""),
                    confidence=max(0.0, min(1.0, float(c.get("confidence") or 0.0))),
                    evidence_source=src if src in _DETECTOR_SOURCES or src == "visual_concept" else "visual_concept",
                )
            )
        return out
    except Exception:
        return []


def annotate_search_results(
    results: list[Any],
    text: str,
    *,
    object_db_path: str = "",
) -> list[Any]:
    """Attach entity debug + a separate ranking signal. Search-only; no index writes."""
    query = parse_entity_query(text)
    if not query.entities:
        return results
    for result in results:
        dbg = getattr(result, "debug", None)
        if not isinstance(dbg, dict):
            dbg = {}
            result.debug = dbg
        tm = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
        rec = {
            "texture_map": tm,
            "pattern_family": getattr(result, "pattern_family", "") or "",
            "global_object_intelligence": tm.get("global_object_intelligence"),
        }
        file_recs = records_from_file_meta(rec)
        file_recs.extend(
            _records_from_object_index(int(getattr(result, "file_id", 0) or 0), object_db_path)
        )
        score, kind, hits = match_score(query, file_recs)
        lines = debug_lines(hits) or [
            f"{e.entity.capitalize()} — sorgu" for e in query.entities[:6]
        ]
        dbg["entity_debug_lines"] = lines
        dbg["entity_query"] = [e.canonical_name for e in query.entities]
        dbg["entity_colors"] = list(query.colors)
        dbg["entity_brands"] = list(query.brands)
        dbg["clip_as_entity_detection"] = False
        dbg["entity_evidence_kind"] = kind
        dbg["openclip_entity"] = next(
            (h.confidence for h in hits if h.evidence_source == "openclip"),
            0.0,
        )
        if kind in {"object_detector", "fusion", "visual_concept"}:
            dbg["entity_evidence"] = True
            dbg["entity_evidence_score"] = float(score)
            dbg["entity_semantic_score"] = 0.0
            dbg["entity_fusion_score"] = float(score)
            dbg["clip_as_detector"] = False
        elif kind in {"openclip", "pattern_dna", "ocr", "metadata"}:
            dbg["entity_evidence"] = False
            dbg["entity_evidence_score"] = 0.0
            dbg["entity_semantic_score"] = float(score)
            dbg["entity_fusion_score"] = float(score)
        else:
            dbg["entity_evidence"] = False
            dbg["entity_evidence_score"] = 0.0
            dbg["entity_fusion_score"] = 0.0
        bd = getattr(result, "breakdown", None)
        if isinstance(bd, dict):
            bd["entity_score"] = float(dbg.get("entity_evidence_score") or 0.0)
    return results


def ranking_channels_unchanged() -> dict[str, float]:
    """Document fusion weights; callers must not replace DINO/CLIP with entity-only."""
    dummy = VisualChannelScores(clip=0.5, dino=0.5)
    fuse_visual_channels(dummy)
    from core.visual_concept import DEFAULT_FUSION_WEIGHTS

    return dict(DEFAULT_FUSION_WEIGHTS)
