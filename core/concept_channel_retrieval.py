"""Concept Intelligence V3 — per-channel retrieval, then AND.

INDEX_FROZEN: read-only. No Pattern Index / FAISS writes, no reindex, no 116K scan.

AND policy
----------
1. Retrieve each intent channel independently (equal cap). Compound queries never
   replace the pool with the full brand index.
2. A channel is HAS_EVIDENCE when its own layer returns candidate ids.
3. Motif detector is not production. Floral/motif claims require real detection
   boxes (class+bbox+confidence). Filename/OCR/CLIP/heatmap is not motif evidence.
   Empty box layer → EVIDENCE_UNAVAILABLE; never tag flower present without boxes.
4. When every remaining channel HAS_EVIDENCE and the intersection is non-empty,
   candidate_ids = intersection (AND).
5. Otherwise candidate_ids = union of HAS_EVIDENCE pools (no fabricated evidence).
6. Missing channel on a result = omit the field (do not invent evidence).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.query_intent_router import ConceptChannel, QueryIntent, intent_retrieval_needles
from core.textile_terms import normalize_turkish

EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
HAS_EVIDENCE = "HAS_EVIDENCE"
NO_HITS = "NO_HITS"

MOTIF_DETECTOR_PRODUCTION = False

# Query values → detector classes (boxes only; not FTS).
MOTIF_VALUE_CLASSES: dict[str, tuple[str, ...]] = {
    "floral": ("flower", "rose"),
    "flower": ("flower",),
    "rose": ("rose",),
    "yaprak": ("leaf",),
    "leaf": ("leaf",),
    "kelebek": ("butterfly",),
    "butterfly": ("butterfly",),
    "leopard": ("leopard",),
}


def motif_box_classes(ch: ConceptChannel) -> list[str]:
    val = normalize_turkish(ch.value or ch.token or "")
    return list(MOTIF_VALUE_CLASSES.get(val, (val,) if val else ()))

CHANNEL_FIELD = {
    "brand": "brand_evidence",
    "material_color": "material_evidence",
    "pattern": "pattern_evidence",
    "motif_object": "motif_evidence",
    "category": "category_evidence",
}

SearchFn = Callable[[list[str]], list[int]]

_FLORAL_VALUES = {"floral", "cicek", "flower", "flowers"}


def is_compound_intent(intent: QueryIntent | None) -> bool:
    return bool(intent is not None and len(getattr(intent, "channels", ()) or ()) >= 2)


def channel_needles(ch: ConceptChannel) -> list[str]:
    mini = QueryIntent(
        kind="pattern",
        value=ch.value,
        tokens=(ch.token,) if ch.token else (),
        explicit=True,
        channel=ch.channel,
        label=ch.label,
        channels=(ch,),
    )
    return intent_retrieval_needles(mini)


def _is_floral_motif(ch: ConceptChannel) -> bool:
    return ch.channel == "motif_object" and normalize_turkish(ch.value) in _FLORAL_VALUES


def _haystack(rec: dict[str, Any]) -> str:
    parts = [
        rec.get("filename"),
        rec.get("path"),
        rec.get("ocr_text"),
        rec.get("category_path"),
        rec.get("manual_category_path"),
        rec.get("category_aliases"),
        rec.get("pattern_family"),
        rec.get("pattern_subfamily"),
        rec.get("animal_print_type"),
    ]
    return normalize_turkish(" ".join(str(x or "") for x in parts))


def needles_in_haystack(needles: list[str], hay: str) -> list[str]:
    hits: list[str] = []
    seen: set[str] = set()
    for n in needles:
        k = normalize_turkish(n)
        if len(k) < 2 or k in seen:
            continue
        if k in hay:
            seen.add(k)
            hits.append(n)
    return hits


@dataclass
class ChannelLayer:
    channel: str
    value: str
    token: str = ""
    status: str = NO_HITS
    needles: list[str] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "value": self.value,
            "token": self.token,
            "status": self.status,
            "needles": list(self.needles),
            "hit_count": len(self.ids),
        }


@dataclass
class CompoundRetrieval:
    layers: list[ChannelLayer] = field(default_factory=list)
    policy: str = "UNION"
    candidate_ids: list[int] = field(default_factory=list)
    and_ids: list[int] = field(default_factory=list)
    brand_pool_override: bool = False

    def status_for(self, channel: str) -> str:
        for layer in self.layers:
            if layer.channel == channel:
                return layer.status
        return ""

    def to_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "concept_v3": True,
            "concept_v3_policy": self.policy,
            "concept_v3_layers": [x.to_dict() for x in self.layers],
            "brand_pool_override": self.brand_pool_override,
        }
        for layer in self.layers:
            field_name = CHANNEL_FIELD.get(layer.channel)
            if field_name and layer.status == EVIDENCE_UNAVAILABLE:
                meta[field_name] = EVIDENCE_UNAVAILABLE
                if layer.channel == "motif_object":
                    meta["motif_status"] = EVIDENCE_UNAVAILABLE
        return meta


def retrieve_compound(
    intent: QueryIntent,
    search_fn: SearchFn,
    *,
    per_channel_limit: int = 400,
    motif_box_search: SearchFn | None = None,
) -> CompoundRetrieval:
    """Separate retrieval layers, then AND when evidence exists."""
    layers: list[ChannelLayer] = []
    for ch in intent.channels:
        needles = channel_needles(ch)
        if ch.channel == "motif_object" or _is_floral_motif(ch):
            # Boxes only. Text/FTS hits are not flower evidence.
            box_needles = list(motif_box_classes(ch)) or list(needles)
            raw_ids = motif_box_search(box_needles) if motif_box_search else []
        else:
            raw_ids = search_fn(needles) if needles else []
        ids: list[int] = []
        seen: set[int] = set()
        for raw in raw_ids:
            try:
                fid = int(raw)
            except (TypeError, ValueError):
                continue
            if fid <= 0 or fid in seen:
                continue
            seen.add(fid)
            ids.append(fid)
            if len(ids) >= max(1, int(per_channel_limit)):
                break
        if ch.channel == "motif_object" or _is_floral_motif(ch):
            status = HAS_EVIDENCE if ids else EVIDENCE_UNAVAILABLE
        elif ids:
            status = HAS_EVIDENCE
        else:
            status = NO_HITS
        layers.append(
            ChannelLayer(
                channel=ch.channel,
                value=ch.value,
                token=ch.token,
                status=status,
                needles=list(needles),
                ids=ids,
            )
        )

    evidenced = [x for x in layers if x.status == HAS_EVIDENCE]
    and_set: set[int] | None = None
    for layer in evidenced:
        chunk = set(layer.ids)
        and_set = chunk if and_set is None else (and_set & chunk)
    and_ids = sorted(and_set or set())

    if len(evidenced) >= 2 and and_ids:
        policy = "AND"
        candidate_ids = list(and_ids)
    else:
        policy = "UNION"
        uniq: list[int] = []
        seen_u: set[int] = set()
        for layer in evidenced:
            for fid in layer.ids:
                if fid not in seen_u:
                    seen_u.add(fid)
                    uniq.append(fid)
        candidate_ids = uniq
        if len(evidenced) >= 2 and not and_ids:
            policy = "UNION_NO_AND_HIT"
        brand_layers = [x for x in evidenced if x.channel == "brand"]
        if brand_layers and policy != "AND":
            # Gold/aksesuar FTS is not brand evidence. Keep the brand pool only.
            uniq_b: list[int] = []
            seen_b: set[int] = set()
            for layer in brand_layers:
                for fid in layer.ids:
                    if fid not in seen_b:
                        seen_b.add(fid)
                        uniq_b.append(fid)
            candidate_ids = uniq_b

    return CompoundRetrieval(
        layers=layers,
        policy=policy,
        candidate_ids=candidate_ids,
        and_ids=and_ids,
        brand_pool_override=any(x.channel == "brand" for x in layers),
    )


_PRIMARY_FLOOR_CHANNELS = frozenset({"brand", "pattern", "texture_style"})


def primary_evidenced_ids(
    report: CompoundRetrieval | None,
) -> tuple[set[int], str]:
    """Ids that may pass the text threshold when another channel is missing.

    Brand wins over pattern/texture. Category/color FTS (aksesuar folder, gold
    filename) is not a primary keep-set — those layers stay in the report
    without expanding recall.
    """
    if report is None:
        return set(), ""
    brand: set[int] = set()
    other: set[int] = set()
    for layer in report.layers or []:
        if str(getattr(layer, "status", "") or "") != HAS_EVIDENCE:
            continue
        ids: set[int] = set()
        for raw in getattr(layer, "ids", None) or []:
            try:
                fid = int(raw)
            except (TypeError, ValueError):
                continue
            if fid > 0:
                ids.add(fid)
        ch = str(getattr(layer, "channel", "") or "")
        if ch == "brand":
            brand |= ids
        elif ch in _PRIMARY_FLOOR_CHANNELS:
            other |= ids
    if brand:
        return brand, "brand"
    return other, "other" if other else ""


def apply_compound_evidence_floor(
    results: list[Any],
    report: CompoundRetrieval | None,
    *,
    threshold: float = 0.60,
    brand_floor: float = 0.90,
    other_floor: float = 0.62,
) -> None:
    """Restore primary-channel hits after later rankers cap them below threshold.

    UNION / missing-channel queries must not AND-zero: a leopard (or brand)
    hit stays displayable even when flower/gold evidence is unavailable.
    Does not invent missing concepts on the result.
    """
    primary_ids, kind = primary_evidenced_ids(report)
    if not primary_ids:
        return
    floor = brand_floor if kind == "brand" else other_floor
    note = (
        "Kanıtlı marka kanalı"
        if kind == "brand"
        else "Kanıtlı kanal (eksik kavram yok sayıldı)"
    )
    for row in results or []:
        try:
            fid = int(getattr(row, "file_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if fid not in primary_ids:
            continue
        score = max(float(getattr(row, "score", 0.0) or 0.0), floor)
        row.score = score
        row.score_percent = round(score * 100, 1)
        notes = list(getattr(row, "match_explanations", None) or [])
        if note not in notes:
            notes.append(note)
            row.match_explanations = notes
        dbg = getattr(row, "debug", None)
        if isinstance(dbg, dict):
            dbg["threshold_passed"] = score >= float(threshold)


def evidence_for_record(
    rec: dict[str, Any],
    intent: QueryIntent,
    report: CompoundRetrieval | None,
    *,
    db_path: str = "",
    motif_boxes_fn: Callable[[dict[str, Any]], list[Any]] | None = None,
) -> dict[str, Any]:
    """Real per-result evidence only. Missing channel → omitted key."""
    out: dict[str, Any] = {}
    hay = _haystack(rec)
    by_ch = {x.channel: x for x in (report.layers if report else [])}
    for ch in intent.channels:
        field_name = CHANNEL_FIELD.get(ch.channel)
        if not field_name:
            continue
        layer = by_ch.get(ch.channel)
        if layer and layer.status == EVIDENCE_UNAVAILABLE:
            continue
        if ch.channel == "motif_object" or _is_floral_motif(ch):
            boxes = motif_boxes_fn(rec) if motif_boxes_fn else []
            if boxes:
                out[field_name] = boxes
            continue
        needles = list(layer.needles) if layer else channel_needles(ch)
        if ch.channel == "brand":
            try:
                from core.brand_evidence import extract_brand_evidence
                from core.brand_aliases import normalize_brand_key

                found = extract_brand_evidence(rec, db_path)
                want = normalize_brand_key(ch.value or ch.token)
                hit = sorted(x for x in found if x and (not want or want in x or x in want))
                if hit:
                    out[field_name] = hit
            except Exception:
                text_hits = needles_in_haystack(needles, hay)
                if text_hits:
                    out[field_name] = text_hits
            continue
        text_hits = needles_in_haystack(needles, hay)
        if text_hits:
            out[field_name] = text_hits
    return out
