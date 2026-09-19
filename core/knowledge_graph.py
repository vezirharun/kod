"""Knowledge Graph Reasoning Engine — çok adımlı tekstil bilgi grafiği.

Textile Knowledge Base üzerine IS_A / RELATED / STYLE / COLOR / BRAND kenarları
kurar; sorgu genişletme, kontrollü aile yayılımı, dinamik Knowledge Score ve
açıklanabilir eşleşmeler üretir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

# Kenar türleri
REL_IS_A = "is_a"
REL_RELATED = "related"
REL_STYLE = "style_of"
REL_COLOR = "color_of"
REL_BRAND = "brand_of"
REL_EXPANDS = "expands"

# Açıklamada gösterilecek etiketler
_LABELS: dict[str, str] = {
    "animal_print": "Animal Print",
    "leopard": "Leopard",
    "jaguar": "Jaguar",
    "cheetah": "Cheetah",
    "tiger": "Tiger",
    "snake": "Snake",
    "python": "Python",
    "crocodile": "Crocodile",
    "zebra": "Zebra",
    "wild_cat": "Wild Cat",
    "organic_spot": "Organic Spot",
    "luxury": "Luxury",
    "safari": "Safari",
    "brown": "Brown",
    "beige": "Beige",
    "floral": "Floral",
    "rose": "Rose",
    "ditsy": "Ditsy",
    "tropical": "Tropical",
    "geometric": "Geometric",
    "stripe": "Stripe",
    "plaid_check": "Plaid / Check",
    "paisley": "Paisley",
    "baroque": "Baroque",
    "monogram_logo": "Monogram / Logo",
    "marble_abstract": "Marble / Abstract",
    "camouflage": "Camouflage",
    "lace": "Lace",
    "scarf_border": "Scarf Border",
    "typography_text": "Typography",
    "vintage": "Vintage",
    "modern": "Modern",
    "ethnic": "Ethnic",
    "romantic": "Romantic",
    "boho": "Boho",
    "resort": "Resort",
    "nautical": "Nautical",
    "military": "Military",
    "gold": "Gold",
    "black_white": "Black & White",
    "navy": "Navy",
    "burgundy": "Burgundy",
    "louis_vuitton": "Louis Vuitton",
    "gucci": "Gucci",
    "versace": "Versace",
    "burberry": "Burberry",
    "dior": "Dior",
    "amiri": "Amiri",
    "leopard_spot": "Leopard Spot",
    "cheetah_spot": "Cheetah Spot",
    "tiger_stripe": "Tiger Stripe",
    "snake_scale": "Snake Scale",
}


def _lab(node_id: str) -> str:
    if node_id in _LABELS:
        return _LABELS[node_id]
    return node_id.replace("_", " ").title()


@dataclass
class GraphEdge:
    src: str
    dst: str
    rel: str
    weight: float = 1.0


@dataclass
class ExpansionHit:
    node_id: str
    label: str
    hops: int
    path: list[str]
    score: float  # hop decay


@dataclass
class ReasoningResult:
    query_nodes: list[str]
    expanded: list[ExpansionHit]
    family_ids: set[str] = field(default_factory=set)
    motif_ids: set[str] = field(default_factory=set)
    style_ids: set[str] = field(default_factory=set)
    color_ids: set[str] = field(default_factory=set)
    match_labels: list[str] = field(default_factory=list)
    knowledge_match: float = 0.0
    knowledge_weight: float = 0.0
    knowledge_contribution: float = 0.0
    modality_weights: dict[str, float] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Yönsüz komşuluk + yönlü tip bilgisi."""

    nodes: set[str] = field(default_factory=set)
    neighbors: dict[str, list[tuple[str, str, float]]] = field(default_factory=dict)
    node_kind: dict[str, str] = field(default_factory=dict)  # family|motif|style|color|brand|concept

    def add_node(self, node_id: str, *, kind: str = "concept") -> None:
        nid = (node_id or "").strip().lower()
        if not nid:
            return
        self.nodes.add(nid)
        self.node_kind.setdefault(nid, kind)
        self.neighbors.setdefault(nid, [])

    def add_edge(
        self,
        src: str,
        dst: str,
        rel: str = REL_RELATED,
        *,
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        a = (src or "").strip().lower()
        b = (dst or "").strip().lower()
        if not a or not b or a == b:
            return
        self.add_node(a)
        self.add_node(b)
        self.neighbors.setdefault(a, []).append((b, rel, weight))
        if bidirectional:
            inv = {
                REL_IS_A: REL_EXPANDS,
                REL_EXPANDS: REL_IS_A,
                REL_STYLE: REL_RELATED,
                REL_COLOR: REL_RELATED,
                REL_BRAND: REL_RELATED,
            }.get(rel, REL_RELATED)
            self.neighbors.setdefault(b, []).append((a, inv, weight))

    def multi_hop(
        self,
        seeds: Iterable[str],
        *,
        max_hops: int = 3,
        max_nodes: int = 48,
        min_score: float = 0.18,
    ) -> list[ExpansionHit]:
        seeds_n = [s.strip().lower() for s in seeds if s and str(s).strip()]
        if not seeds_n:
            return []
        best: dict[str, ExpansionHit] = {}
        # BFS
        from collections import deque

        q: deque[tuple[str, int, list[str], float]] = deque()
        for s in seeds_n:
            if s not in self.nodes:
                # soft alias via label reverse
                continue
            q.append((s, 0, [s], 1.0))
            best[s] = ExpansionHit(s, _lab(s), 0, [s], 1.0)

        while q and len(best) < max_nodes:
            node, hops, path, score = q.popleft()
            if hops >= max_hops:
                continue
            for nxt, rel, w in self.neighbors.get(node, []):
                # IS_A yukarı / EXPANDS kardeş yayılımı
                decay = 0.72 if rel in (REL_IS_A, REL_EXPANDS, REL_RELATED) else 0.60
                if rel == REL_STYLE:
                    decay = 0.68
                if rel == REL_COLOR:
                    decay = 0.55
                ns = score * decay * float(w)
                if ns < min_score:
                    continue
                nh = hops + 1
                prev = best.get(nxt)
                if prev and prev.score >= ns:
                    continue
                npath = path + [nxt]
                hit = ExpansionHit(nxt, _lab(nxt), nh, npath, ns)
                best[nxt] = hit
                q.append((nxt, nh, npath, ns))

        return sorted(best.values(), key=lambda h: (-h.score, h.hops, h.node_id))


def _seed_core_edges(g: KnowledgeGraph) -> None:
    """Kullanıcı örneğindeki animal-print ağacı + diğer sektör ilişkileri."""
    # Animal Print tree
    for child in ("leopard", "jaguar", "cheetah", "tiger", "snake", "zebra", "crocodile"):
        g.add_edge("animal_print", child, REL_EXPANDS, weight=1.0)
        g.add_node(child, kind="family")
    g.add_node("animal_print", kind="family")

    for child in ("luxury", "safari", "organic_spot", "brown", "beige", "wild_cat"):
        kind = "style" if child in ("luxury", "safari") else (
            "color" if child in ("brown", "beige") else "concept"
        )
        g.add_edge("leopard", child, REL_RELATED, weight=0.95)
        g.add_node(child, kind=kind)

    # Wild cat cluster
    for a, b in (
        ("leopard", "jaguar"),
        ("leopard", "cheetah"),
        ("jaguar", "cheetah"),
        ("tiger", "leopard"),
        ("wild_cat", "leopard"),
        ("wild_cat", "jaguar"),
        ("wild_cat", "cheetah"),
        ("wild_cat", "tiger"),
    ):
        g.add_edge(a, b, REL_RELATED, weight=0.9)

    # Luxury → animal print hop (multi-hop entry)
    g.add_edge("luxury", "animal_print", REL_STYLE, weight=0.95)
    g.add_edge("safari", "animal_print", REL_STYLE, weight=0.9)
    g.add_edge("luxury", "leopard", REL_RELATED, weight=0.85)
    g.add_edge("luxury", "baroque", REL_RELATED, weight=0.7)
    g.add_edge("luxury", "monogram_logo", REL_RELATED, weight=0.8)

    # Reptile / stripe animals
    g.add_edge("snake", "python", REL_EXPANDS, weight=1.0)
    g.add_edge("snake", "crocodile", REL_RELATED, weight=0.75)
    g.add_edge("organic_spot", "leopard_spot", REL_RELATED, weight=0.9)
    g.add_edge("organic_spot", "cheetah_spot", REL_RELATED, weight=0.85)
    g.add_edge("leopard", "leopard_spot", REL_RELATED, weight=1.0)
    g.add_edge("cheetah", "cheetah_spot", REL_RELATED, weight=1.0)
    g.add_edge("tiger", "tiger_stripe", REL_RELATED, weight=1.0)
    g.add_edge("snake", "snake_scale", REL_RELATED, weight=1.0)

    # Floral
    g.add_node("floral", kind="family")
    for child in ("rose", "daisy", "ditsy", "tropical", "peony", "botanical"):
        g.add_edge("floral", child, REL_EXPANDS, weight=1.0)
    g.add_edge("romantic", "floral", REL_STYLE, weight=0.9)
    g.add_edge("resort", "tropical", REL_STYLE, weight=0.85)
    g.add_edge("tropical", "floral", REL_IS_A, weight=1.0)

    # Geometric / check / stripe
    for root, kids in (
        ("geometric", ("chevron", "hexagon", "optical", "mosaic")),
        ("plaid_check", ("tartan", "gingham", "houndstooth", "burberry")),
        ("stripe", ("pinstripe", "breton", "awning")),
        ("paisley", ("boteh", "scarf_paisley")),
        ("baroque", ("ornate_scroll", "medusa", "versace")),
        ("monogram_logo", ("logo_allover", "louis_vuitton", "gucci", "dior")),
        ("marble_abstract", ("marble", "watercolor", "fluid_art")),
        ("camouflage", ("military", "safari")),
    ):
        g.add_node(root, kind="family")
        for k in kids:
            g.add_edge(root, k, REL_EXPANDS, weight=0.95)

    # Colors
    for c in ("brown", "beige", "navy", "burgundy", "gold", "black_white", "cream", "olive"):
        g.add_node(c, kind="color")
    g.add_edge("brown", "beige", REL_RELATED, weight=0.8)
    g.add_edge("animal_print", "brown", REL_COLOR, weight=0.7)
    g.add_edge("leopard", "brown", REL_COLOR, weight=0.85)
    g.add_edge("leopard", "beige", REL_COLOR, weight=0.8)

    # Brands
    for brand, fam in (
        ("louis_vuitton", "monogram_logo"),
        ("gucci", "monogram_logo"),
        ("dior", "monogram_logo"),
        ("versace", "baroque"),
        ("burberry", "plaid_check"),
        ("amiri", "typography_text"),
    ):
        g.add_edge(brand, fam, REL_BRAND, weight=1.0)
        g.add_edge(brand, "luxury", REL_STYLE, weight=0.9)

    # Styles
    for st in ("vintage", "modern", "ethnic", "boho", "romantic", "military", "nautical", "resort"):
        g.add_node(st, kind="style")
    g.add_edge("ethnic", "paisley", REL_STYLE, weight=0.8)
    g.add_edge("boho", "ethnic", REL_RELATED, weight=0.75)
    g.add_edge("military", "camouflage", REL_STYLE, weight=0.95)
    g.add_edge("nautical", "stripe", REL_STYLE, weight=0.85)


def _ingest_textile_kb(g: KnowledgeGraph) -> None:
    """KB motif ilişkileri + kök aileleri grafa ekle."""
    try:
        from core.textile_knowledge_base import ROOT_FAMILIES, get_knowledge_base

        kb = get_knowledge_base()
    except Exception:
        return

    for root in ROOT_FAMILIES:
        g.add_node(root, kind="family")

    for fid, fam in kb.families.items():
        g.add_node(fid, kind="family")
        root = fam.root or fam.parent
        if root and root != fid:
            g.add_edge(root, fid, REL_EXPANDS, weight=0.85, bidirectional=True)
        if fam.parent and fam.parent != fid and fam.parent != root:
            g.add_edge(fam.parent, fid, REL_EXPANDS, weight=0.8)

    for mid, motif in kb.motifs.items():
        g.add_node(mid, kind="motif")
        for fam in motif.families:
            if fam and fam != "unknown":
                g.add_edge(fam, mid, REL_RELATED, weight=0.9)
        # synonym aliases as soft nodes pointing to motif
        for syn in motif.synonyms[:8]:
            syn_id = syn.strip().lower().replace(" ", "_")
            if syn_id and syn_id != mid and len(syn_id) > 2:
                g.add_node(syn_id, kind="concept")
                g.add_edge(syn_id, mid, REL_RELATED, weight=0.95)

    for mid, related in kb.motif_relations.items():
        for other in related:
            g.add_edge(mid, other, REL_RELATED, weight=0.88)

    for brand, meta in kb.brand_families.items():
        g.add_node(brand, kind="brand")
        pf = str(meta.get("pattern_family") or "")
        if pf:
            g.add_edge(brand, pf, REL_BRAND, weight=1.0)
        g.add_edge(brand, "luxury", REL_STYLE, weight=0.85)
        for a in meta.get("aliases", []):
            aid = str(a).lower().replace(" ", "_")
            g.add_node(aid, kind="brand")
            g.add_edge(aid, brand, REL_RELATED, weight=1.0)

    for cid in kb.color_families:
        g.add_node(cid, kind="color")
    for sid in kb.style_classes:
        g.add_node(sid, kind="style")


_ALIAS_TO_NODE: dict[str, str] = {
    "leopar": "leopard",
    "leo": "leopard",
    "leopard print": "leopard",
    "animal": "animal_print",
    "animal print": "animal_print",
    "hayvan": "animal_print",
    "hayvan deseni": "animal_print",
    "kaplan": "tiger",
    "yılan": "snake",
    "yilan": "snake",
    "çiçek": "floral",
    "cicek": "floral",
    "flower": "floral",
    "luks": "luxury",
    "lüks": "luxury",
    "luxury animal": "luxury",
    "luxury animal print": "luxury",
    "organic spot": "organic_spot",
    "wild cat": "wild_cat",
    "ekose": "plaid_check",
    "plaid": "plaid_check",
    "check": "plaid_check",
    "çizgi": "stripe",
    "cizgi": "stripe",
    "logo": "monogram_logo",
    "monogram": "monogram_logo",
    "lv": "louis_vuitton",
    "gg": "gucci",
    "kahverengi": "brown",
    "bej": "beige",
    "kamuflaj": "camouflage",
    "camo": "camouflage",
}


def resolve_query_nodes(text: str, *, extra: Iterable[str] | None = None) -> list[str]:
    """Sorgu metninden grafik düğümleri çıkar."""
    g = get_knowledge_graph()
    found: list[str] = []
    raw = (text or "").strip().lower()
    extras = [str(x).strip().lower() for x in (extra or []) if x]

    candidates = []
    if raw:
        candidates.append(raw)
        # multi-word: try full + tokens
        for alias, node in _ALIAS_TO_NODE.items():
            if alias in raw:
                candidates.append(node)
        for tok in raw.replace("-", " ").split():
            if len(tok) >= 2:
                candidates.append(tok)
                candidates.append(_ALIAS_TO_NODE.get(tok, tok))
    candidates.extend(extras)

    seen: set[str] = set()
    for c in candidates:
        c = (c or "").strip().lower().replace(" ", "_")
        if not c:
            continue
        # alias map
        c2 = _ALIAS_TO_NODE.get(c.replace("_", " "), c)
        c2 = _ALIAS_TO_NODE.get(c2, c2)
        if c2 in g.nodes and c2 not in seen:
            seen.add(c2)
            found.append(c2)
        elif c in g.nodes and c not in seen:
            seen.add(c)
            found.append(c)
    return found


@lru_cache(maxsize=1)
def get_knowledge_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    _seed_core_edges(g)
    _ingest_textile_kb(g)
    return g


def expand_query(
    text: str = "",
    *,
    family: str = "",
    motif: str = "",
    color: str = "",
    max_hops: int = 3,
) -> ReasoningResult:
    """Multi-hop family/motif/style genişletme."""
    seeds = resolve_query_nodes(
        text,
        extra=[family, motif, color],
    )
    g = get_knowledge_graph()
    expanded = g.multi_hop(seeds, max_hops=max_hops)
    result = ReasoningResult(query_nodes=list(seeds), expanded=expanded)
    for hit in expanded:
        kind = g.node_kind.get(hit.node_id, "concept")
        if kind == "family" or hit.node_id in (
            "animal_print", "floral", "geometric", "stripe", "plaid_check",
            "paisley", "baroque", "monogram_logo", "marble_abstract", "lace",
            "camouflage", "leopard", "jaguar", "cheetah", "tiger", "snake",
        ):
            result.family_ids.add(hit.node_id)
        if kind == "motif":
            result.motif_ids.add(hit.node_id)
        if kind == "style":
            result.style_ids.add(hit.node_id)
        if kind == "color":
            result.color_ids.add(hit.node_id)
    return result


def dynamic_modality_weights(
    *,
    text_query: str = "",
    has_ocr_signal: bool = False,
    kb_seed_count: int = 0,
    expansion_strength: float = 0.0,
    visual_confidence: float = 0.5,
    dna_confidence: float = 0.0,
    semantic_confidence: float = 0.0,
) -> dict[str, float]:
    """Sorgu tipine göre Visual/DNA/Knowledge/Semantic/OCR ağırlıkları (toplam 1)."""
    text = (text_query or "").strip()
    text_knowledge = kb_seed_count > 0 or expansion_strength >= 0.4

    if not text:
        # Saf görsel arama → Knowledge düşük
        w = {
            "visual": 0.52,
            "dna": 0.28,
            "knowledge": 0.03,
            "semantic": 0.12,
            "ocr": 0.05 if has_ocr_signal else 0.0,
        }
    elif text_knowledge and expansion_strength >= 0.55:
        # "Luxury animal print" tipi
        w = {
            "visual": 0.45,
            "dna": 0.25,
            "knowledge": 0.18,
            "semantic": 0.08,
            "ocr": 0.04,
        }
    elif text_knowledge:
        w = {
            "visual": 0.48,
            "dna": 0.26,
            "knowledge": 0.12,
            "semantic": 0.10,
            "ocr": 0.04,
        }
    else:
        # Metin var ama KB zayıf
        w = {
            "visual": 0.50,
            "dna": 0.27,
            "knowledge": 0.05,
            "semantic": 0.14,
            "ocr": 0.04,
        }

    if has_ocr_signal:
        w["ocr"] = max(w["ocr"], 0.06)
    # Sinyal güçlerine hafif uyarlama
    if visual_confidence >= 0.85:
        w["visual"] += 0.04
        w["knowledge"] = max(0.02, w["knowledge"] - 0.02)
    if dna_confidence >= 0.7:
        w["dna"] += 0.03
    if semantic_confidence >= 0.65:
        w["semantic"] += 0.03

    total = sum(w.values()) or 1.0
    return {k: round(v / total, 4) for k, v in w.items()}


def candidate_knowledge_match(
    expansion: ReasoningResult,
    *,
    cand_family: str = "",
    cand_motif: str = "",
    cand_color: str = "",
    cand_filename: str = "",
) -> tuple[float, list[str]]:
    """Adayın genişletilmiş graf ile örtüşme skoru + eşleşen etiketler."""
    if not expansion.expanded and not expansion.query_nodes:
        return 0.0, []

    g = get_knowledge_graph()
    cand_nodes = resolve_query_nodes(
        cand_filename,
        extra=[cand_family, cand_motif, cand_color],
    )
    # Normalize taxonomy roots
    for raw in (cand_family, cand_motif, cand_color):
        r = (raw or "").strip().lower()
        if r and r in g.nodes and r not in cand_nodes:
            cand_nodes.append(r)
        # animal subtype shortcuts
        for key in ("leopard", "jaguar", "cheetah", "tiger", "snake", "zebra"):
            if key in r and key not in cand_nodes:
                cand_nodes.append(key)

    exp_map = {h.node_id: h for h in expansion.expanded}
    matched: list[tuple[str, float]] = []
    for cn in cand_nodes:
        hit = exp_map.get(cn)
        if hit:
            matched.append((hit.label, hit.score))
            continue
        # 1-hop proximity to expanded set
        for nxt, _rel, w in g.neighbors.get(cn, []):
            hit2 = exp_map.get(nxt)
            if hit2:
                matched.append((hit2.label, hit2.score * 0.75 * float(w)))
                break

    if not matched:
        # root family overlap
        cf = (cand_family or "").strip().lower()
        if cf and cf in expansion.family_ids:
            matched.append((_lab(cf), 0.7))
        elif cf:
            for fid in expansion.family_ids:
                if fid.startswith(cf) or cf.startswith(fid):
                    matched.append((_lab(fid), 0.55))
                    break

    if not matched:
        return 0.0, []

    # Unique labels, keep best scores
    best: dict[str, float] = {}
    for label, sc in matched:
        best[label] = max(best.get(label, 0.0), sc)
    labels = [k for k, _ in sorted(best.items(), key=lambda x: -x[1])]
    score = min(1.0, sum(best.values()) / max(1.0, 1.6 + 0.15 * len(best)))
    # Boost if seed+family both hit
    if any(s in exp_map or s in cand_nodes for s in expansion.query_nodes):
        score = min(1.0, score + 0.08)
    return round(score, 4), labels[:12]


def reason_knowledge(
    *,
    text_query: str = "",
    query_family: str = "",
    query_motif: str = "",
    query_color: str = "",
    cand_family: str = "",
    cand_motif: str = "",
    cand_color: str = "",
    cand_filename: str = "",
    visual_score: float = 0.5,
    dna_score: float = 0.0,
    semantic_score: float = 0.0,
    has_ocr: bool = False,
    max_hops: int = 3,
) -> ReasoningResult:
    """Tam reasoning: genişlet → eşleştir → dinamik ağırlık → açıklama."""
    expansion = expand_query(
        text_query,
        family=query_family,
        motif=query_motif,
        color=query_color,
        max_hops=max_hops,
    )
    strength = 0.0
    if expansion.expanded:
        strength = max(h.score for h in expansion.expanded[:5])
    weights = dynamic_modality_weights(
        text_query=text_query,
        has_ocr_signal=has_ocr,
        kb_seed_count=len(expansion.query_nodes),
        expansion_strength=strength,
        visual_confidence=visual_score,
        dna_confidence=dna_score,
        semantic_confidence=semantic_score,
    )
    match_score, labels = candidate_knowledge_match(
        expansion,
        cand_family=cand_family,
        cand_motif=cand_motif,
        cand_color=cand_color,
        cand_filename=cand_filename,
    )
    # Seed labels first, then other matches
    seed_labs = [_lab(s) for s in expansion.query_nodes]
    ordered: list[str] = []
    for lab in seed_labs + labels:
        if lab not in ordered:
            ordered.append(lab)
    matched_set = set(labels)
    seed_set = set(seed_labs)
    display = [
        x for x in ordered
        if x in matched_set or (match_score > 0 and x in seed_set and x in matched_set)
    ]
    if match_score > 0 and not display:
        display = labels[:6]
    elif match_score > 0:
        # Include high-score expansion concepts that appear in match path
        for h in expansion.expanded:
            if h.hops <= 2 and h.label in matched_set and h.label not in display:
                display.append(h.label)
            if len(display) >= 10:
                break

    kw = float(weights.get("knowledge") or 0.0)
    expansion.knowledge_match = match_score
    expansion.knowledge_weight = kw
    expansion.knowledge_contribution = round(match_score * kw, 4)
    expansion.modality_weights = weights
    expansion.match_labels = display[:10]
    expansion.explanation = {
        "knowledge_match_labels": display[:10],
        "knowledge_score": round(match_score, 4),
        "knowledge_weight": kw,
        "knowledge_contribution": expansion.knowledge_contribution,
        "modality_weights": weights,
        "expanded_families": sorted(expansion.family_ids)[:20],
        "query_nodes": list(expansion.query_nodes),
        "hops": [
            {
                "id": h.node_id,
                "label": h.label,
                "hops": h.hops,
                "score": round(h.score, 3),
                "path": " → ".join(_lab(p) for p in h.path),
            }
            for h in expansion.expanded[:24]
        ],
    }
    return expansion


# Geriye uyumluluk: eski knowledge_score_boost çağrıları
def knowledge_graph_boost(**kwargs: Any) -> dict[str, float]:
    r = reason_knowledge(
        text_query=str(kwargs.get("query_text") or ""),
        query_family=str(kwargs.get("query_family") or ""),
        query_motif=str(kwargs.get("query_motif") or ""),
        query_color=str(kwargs.get("query_color") or ""),
        cand_family=str(kwargs.get("cand_family") or ""),
        cand_motif=str(kwargs.get("cand_motif") or ""),
        cand_color=str(kwargs.get("cand_color") or ""),
        cand_filename=str(kwargs.get("cand_filename") or ""),
        visual_score=float(kwargs.get("visual_score") or 0.5),
        dna_score=float(kwargs.get("dna_score") or 0.0),
        semantic_score=float(kwargs.get("semantic_score") or 0.0),
        has_ocr=bool(kwargs.get("has_ocr") or False),
    )
    return {
        "kb_family": round(1.0 if r.family_ids else 0.0, 4),
        "kb_motif": round(r.knowledge_match, 4),
        "kb_color": 0.0,
        "kb_text": round(r.knowledge_match, 4),
        "kb_combined": r.knowledge_match,
        "kb_weight": r.knowledge_weight,
        "kb_contribution": r.knowledge_contribution,
        **{f"w_{k}": v for k, v in r.modality_weights.items()},
    }
