"""Müşteri/klasör keşfi — indexed path'lerden otomatik, fuzzy öneri, scope kökleri.

SEARCH SCOPE filtresi için kullanılır. Desen motoruna (DINO/CLIP/FAISS/…) dokunmaz.
Öğrenme / kavram etiketine müşteri adı yazılmaz.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.logger import setup_logger
from core.textile_terms import normalize_turkish
from core.utils import normalize_path, normalize_source_root

logger = setup_logger(__name__)

# Fuzzy: yüksek eşik — yanlış müşteri birleşmesini engelle
_MIN_SUGGEST_SCORE = 0.72
_HIGH_CONFIDENCE = 0.88
_MAX_SUGGESTIONS = 12
_MIN_CUSTOMER_LEN = 2
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass
class CustomerEntry:
    """Tek müşteri (birden fazla kaynak kökü altında birleşebilir)."""

    name: str
    norm_key: str
    roots: list[str] = field(default_factory=list)
    file_count: int = 0
    source_ids: list[int] = field(default_factory=list)
    # "Ünal Tekstil / Numuneler" → scope kökleri
    subfolders: dict[str, list[str]] = field(default_factory=dict)

    def all_scope_prefixes(self) -> list[str]:
        return list(self.roots)


@dataclass
class CustomerSuggestion:
    label: str  # UI'da gösterilen orijinal ad (veya "Ad / Alt")
    customer_name: str
    score: float
    scope_prefixes: list[str]
    high_confidence: bool = False


def normalize_customer_key(text: str) -> str:
    """Büyük/küçük, TR karakter, noktalama, boşluk normalize."""
    if not text:
        return ""
    t = normalize_turkish(str(text))
    t = _PUNCT_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


def _is_plausible_customer_name(name: str) -> bool:
    n = (name or "").strip()
    if len(n) < _MIN_CUSTOMER_LEN:
        return False
    # Saf sayı / tek karakter klasörleri (imalat iş kodları) müşteri sayma
    if n.isdigit():
        return False
    if len(n) <= 2 and n.isascii() and n.isalnum() and not any(c.isalpha() for c in n):
        return False
    # Gizli / sistem
    if n.startswith("."):
        return False
    low = n.lower()
    if low in {"thumbs", "tmp", "temp", "cache", "system volume information"}:
        return False
    return True


def first_folder_under_root(file_path: str, root: str) -> str:
    """Kaynak kökünden sonraki ilk klasör = müşteri adayı."""
    p = normalize_path(file_path)
    r = normalize_source_root(root).rstrip("\\/")
    if not p or not r:
        return ""
    pl, rl = p.casefold(), r.casefold()
    if not pl.startswith(rl):
        return ""
    rest = p[len(r) :].lstrip("\\/")
    if not rest:
        return ""
    return rest.split("\\")[0].split("/")[0]


def second_folder_under_root(file_path: str, root: str) -> str:
    p = normalize_path(file_path)
    r = normalize_source_root(root).rstrip("\\/")
    if not p or not r:
        return ""
    pl, rl = p.casefold(), r.casefold()
    if not pl.startswith(rl):
        return ""
    rest = p[len(r) :].lstrip("\\/")
    parts = [x for x in rest.replace("/", "\\").split("\\") if x]
    if len(parts) < 2:
        return ""
    # İkinci parça dosya adı ise (uzantılı) alt klasör değil
    if "." in parts[1] and not parts[1].startswith("."):
        # yine de klasör olabilir; dosya uzantısı kontrolü
        ext = parts[1].rsplit(".", 1)[-1].lower()
        if len(ext) <= 5 and ext.isalnum() and parts[1].count(".") == 1:
            return ""
    return parts[1]


def customer_root_prefix(root: str, customer_name: str) -> str:
    r = normalize_source_root(root).rstrip("\\/")
    return f"{r}\\{customer_name}"


def path_under_any_prefix(path: str, prefixes: Iterable[str]) -> bool:
    p = normalize_path(path)
    if not p:
        return False
    pl = p.casefold()
    for raw in prefixes:
        pref = normalize_path(raw).rstrip("\\/")
        if not pref:
            continue
        pr = pref.casefold()
        if pl == pr or pl.startswith(pr + "\\") or pl.startswith(pr + "/"):
            return True
    return False


def _token_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a.startswith(b) or b.startswith(a):
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) >= 3:
            return 0.92 + 0.08 * (len(shorter) / max(len(longer), 1))
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_customer_query(query: str, candidate_name: str) -> float:
    """0..1 — aşırı geniş eşleşmeleri reddeder."""
    q = normalize_customer_key(query)
    c = normalize_customer_key(candidate_name)
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if c.startswith(q) and len(q) >= 2:
        return 0.96
    q_toks = [t for t in q.split() if t]
    c_toks = [t for t in c.split() if t]
    if not q_toks or not c_toks:
        return 0.0

    # İlk token: müşteri adının ilk kelimesiyle uyum zorunlu (güvenlik)
    first_q, first_c = q_toks[0], c_toks[0]
    first_sim = _token_similarity(first_q, first_c)
    if first_sim < 0.78:
        # İlk token başka bir kelimeye tam oturuyorsa (nadir) izin ver
        alt = max((_token_similarity(first_q, ct) for ct in c_toks), default=0.0)
        if alt < 0.90 or len(first_q) < 4:
            return 0.0

    matched = 0
    token_scores: list[float] = []
    used: set[int] = set()
    for qt in q_toks:
        best_i, best_s = -1, 0.0
        for i, ct in enumerate(c_toks):
            if i in used:
                continue
            s = _token_similarity(qt, ct)
            if s > best_s:
                best_s, best_i = s, i
        if best_s >= 0.78 and best_i >= 0:
            used.add(best_i)
            matched += 1
            token_scores.append(best_s)
        else:
            token_scores.append(best_s)

    coverage = matched / len(q_toks)
    avg_tok = sum(token_scores) / len(token_scores)
    full = difflib.SequenceMatcher(None, q, c).ratio()
    score = 0.45 * avg_tok + 0.35 * coverage + 0.20 * full
    # Tek token kısa sorgu: yalnızca güçlü prefix/benzerlik
    if len(q_toks) == 1 and len(first_q) <= 3:
        score = min(score, first_sim * 0.85)
    return round(min(1.0, score), 4)


class CustomerRegistry:
    """Path tabanlı müşteri katalogu — DB customer kolonuna bağımlı değil."""

    def __init__(self) -> None:
        self._by_key: dict[str, CustomerEntry] = {}
        self._by_display: dict[str, CustomerEntry] = {}
        self.discovered_count: int = 0
        self.last_debug: dict[str, Any] = {}

    def clear(self) -> None:
        self._by_key.clear()
        self._by_display.clear()
        self.discovered_count = 0

    def entries(self) -> list[CustomerEntry]:
        return sorted(self._by_key.values(), key=lambda e: (-e.file_count, e.name.casefold()))

    def names(self) -> list[str]:
        return [e.name for e in self.entries()]

    def get(self, name: str) -> CustomerEntry | None:
        if not name:
            return None
        if name in self._by_display:
            return self._by_display[name]
        key = normalize_customer_key(name)
        return self._by_key.get(key)

    def scope_prefixes_for(self, selection: str) -> list[str]:
        """'Ünal Tekstil' veya 'Ünal Tekstil / Numuneler'."""
        sel = (selection or "").strip()
        if not sel:
            return []
        if " / " in sel:
            parent, sub = sel.split(" / ", 1)
            entry = self.get(parent.strip())
            if not entry:
                return []
            return list(entry.subfolders.get(sub.strip(), []))
        entry = self.get(sel)
        if not entry:
            # Ham seçim — yine de tek kök gibi dene (normalize eşleşme)
            key = normalize_customer_key(sel)
            entry = self._by_key.get(key)
        if not entry:
            return []
        return entry.all_scope_prefixes()

    def path_matches(self, path: str, selection: str) -> bool:
        prefixes = self.scope_prefixes_for(selection)
        if not prefixes:
            # Kayıt yoksa güvenli taraf: eşleşme yok (yanlış geniş arama yok)
            return False
        ok = path_under_any_prefix(path, prefixes)
        if ok:
            self.last_debug = {
                "path": path,
                "selection": selection,
                "matched_via": [p for p in prefixes if path_under_any_prefix(path, [p])],
            }
        return ok

    def suggest(
        self,
        query: str,
        *,
        limit: int = _MAX_SUGGESTIONS,
        include_subfolders: bool = True,
        recent: list[str] | None = None,
    ) -> list[CustomerSuggestion]:
        q = (query or "").strip()
        out: list[CustomerSuggestion] = []
        if not q:
            # Boş: son kullanılan + en sık
            seen: set[str] = set()
            for name in recent or []:
                entry = self.get(name)
                if entry and name not in seen:
                    seen.add(name)
                    out.append(
                        CustomerSuggestion(
                            label=entry.name,
                            customer_name=entry.name,
                            score=1.0,
                            scope_prefixes=entry.all_scope_prefixes(),
                            high_confidence=True,
                        )
                    )
            for entry in self.entries():
                if entry.name in seen:
                    continue
                seen.add(entry.name)
                out.append(
                    CustomerSuggestion(
                        label=entry.name,
                        customer_name=entry.name,
                        score=0.5,
                        scope_prefixes=entry.all_scope_prefixes(),
                        high_confidence=False,
                    )
                )
                if len(out) >= limit:
                    break
            return out[:limit]

        scored: list[CustomerSuggestion] = []
        for entry in self._by_key.values():
            sc = score_customer_query(q, entry.name)
            if sc < _MIN_SUGGEST_SCORE:
                continue
            scored.append(
                CustomerSuggestion(
                    label=entry.name,
                    customer_name=entry.name,
                    score=sc,
                    scope_prefixes=entry.all_scope_prefixes(),
                    high_confidence=sc >= _HIGH_CONFIDENCE,
                )
            )
            if include_subfolders and entry.subfolders:
                # Alt klasörleri sadece sorgu alt adı ima ediyorsa ekle (UI kalabalığı yok)
                sub_q = normalize_customer_key(q)
                parent_n = normalize_customer_key(entry.name)
                wants_sub = ("/" in q) or (
                    len(sub_q) > len(parent_n) + 2 and parent_n and parent_n in sub_q
                )
                if wants_sub:
                    for sub, prefs in entry.subfolders.items():
                        label = f"{entry.name} / {sub}"
                        sc2 = score_customer_query(q, label)
                        sub_n = normalize_customer_key(sub)
                        if sc2 >= _MIN_SUGGEST_SCORE or (
                            sc >= _MIN_SUGGEST_SCORE and sub_n and sub_n in sub_q
                        ):
                            scored.append(
                                CustomerSuggestion(
                                    label=label,
                                    customer_name=entry.name,
                                    score=max(sc2, sc * 0.95),
                                    scope_prefixes=list(prefs),
                                    high_confidence=False,
                                )
                            )
        scored.sort(key=lambda s: (-s.score, s.label.casefold()))
        # Aynı label tek
        seen_l: set[str] = set()
        for s in scored:
            if s.label in seen_l:
                continue
            seen_l.add(s.label)
            out.append(s)
            if len(out) >= limit:
                break
        return out

    def discover_from_paths(
        self,
        paths_with_roots: Iterable[tuple[str, str, int]],
        *,
        collect_subfolders: bool = True,
    ) -> int:
        """(path, source_root, source_id) → katalogu doldur. Dönüş: müşteri sayısı."""
        self.clear()
        # name_key → accum
        buckets: dict[str, dict[str, Any]] = {}
        for path, root, source_id in paths_with_roots:
            cust = first_folder_under_root(path, root)
            if not _is_plausible_customer_name(cust):
                continue
            key = normalize_customer_key(cust)
            if not key:
                continue
            b = buckets.get(key)
            if b is None:
                b = {
                    "name_counts": {cust: 0},
                    "roots": set(),
                    "source_ids": set(),
                    "file_count": 0,
                    "subs": {},  # sub_name -> set(prefixes)
                }
                buckets[key] = b
            b["name_counts"][cust] = b["name_counts"].get(cust, 0) + 1
            b["file_count"] += 1
            pref = customer_root_prefix(root, cust)
            b["roots"].add(normalize_path(pref))
            if source_id:
                b["source_ids"].add(int(source_id))
            if collect_subfolders:
                sub = second_folder_under_root(path, root)
                if sub and _is_plausible_customer_name(sub):
                    sub_pref = normalize_path(f"{pref}\\{sub}")
                    b["subs"].setdefault(sub, set()).add(sub_pref)

        for key, b in buckets.items():
            # En sık görülen orijinal yazımı display name yap
            display = max(b["name_counts"].items(), key=lambda kv: kv[1])[0]
            entry = CustomerEntry(
                name=display,
                norm_key=key,
                roots=sorted(b["roots"]),
                file_count=int(b["file_count"]),
                source_ids=sorted(b["source_ids"]),
                subfolders={s: sorted(prefs) for s, prefs in b["subs"].items()},
            )
            self._by_key[key] = entry
            self._by_display[display] = entry

        self.discovered_count = len(self._by_key)
        logger.info(
            "customer_discovery: %d müşteri/klasör keşfedildi (path tabanlı)",
            self.discovered_count,
        )
        return self.discovered_count


def discover_customers_from_db(db: Any, *, limit_paths: int = 0) -> CustomerRegistry:
    """Aktif kaynaklar + indexed path'lerden müşteri kataloğu üret."""
    reg = CustomerRegistry()
    sources = db.list_sources(active_only=True) if hasattr(db, "list_sources") else []
    root_by_id: dict[int, str] = {}
    for src in sources or []:
        sid = int(src.get("id") or 0)
        root = src.get("root_path") or ""
        if sid and root:
            root_by_id[sid] = normalize_source_root(root)

    if not root_by_id:
        logger.info("customer_discovery: aktif kaynak yok")
        return reg

    triples: list[tuple[str, str, int]] = []
    # Hafif sorgu: path + source_id
    try:
        with db.connect() as conn:
            sql = """
                SELECT path, source_id FROM files
                WHERE status NOT IN ('excluded_internal','missing')
                  AND COALESCE(path,'') != ''
            """
            if limit_paths > 0:
                sql += f" LIMIT {int(limit_paths)}"
            rows = conn.execute(sql).fetchall()
    except Exception as exc:
        logger.warning("customer_discovery DB okuma hatası: %s", exc)
        return reg

    for row in rows:
        path = row["path"] if not isinstance(row, tuple) else row[0]
        sid = int((row["source_id"] if not isinstance(row, tuple) else row[1]) or 0)
        root = root_by_id.get(sid, "")
        if not root:
            continue
        triples.append((str(path), root, sid))

    n = reg.discover_from_paths(triples)
    logger.debug(
        "customer_discovery debug: sources=%d paths=%d customers=%d sample=%s",
        len(root_by_id),
        len(triples),
        n,
        [e.name for e in reg.entries()[:8]],
    )
    return reg


# Process-wide cache (UI + SearchEngine paylaşır)
_REGISTRY: CustomerRegistry | None = None


def get_customer_registry(*, refresh: bool = False, db: Any = None) -> CustomerRegistry:
    global _REGISTRY
    if _REGISTRY is not None and not refresh:
        return _REGISTRY
    if db is None:
        _REGISTRY = _REGISTRY or CustomerRegistry()
        return _REGISTRY
    _REGISTRY = discover_customers_from_db(db)
    return _REGISTRY


def set_customer_registry(reg: CustomerRegistry | None) -> None:
    global _REGISTRY
    _REGISTRY = reg


# Locative / “içindeki” scope tails — strip before fuzzy customer match.
_SCOPE_SUFFIX_RE = re.compile(
    r"(?:[''`´]?(?:daki|deki)|icin(?:de(?:ki)?)?|uzerinde(?:ki)?|altinda(?:ki)?|"
    r"yaninda(?:ki)?|arasinda(?:ki)?)$",
    re.IGNORECASE,
)

# Tokens that are never a customer name by themselves (concept/attr/noise).
_NON_CUSTOMER_TOKENS = frozenset(
    {
        "desen",
        "deseni",
        "desenler",
        "desenleri",
        "pattern",
        "print",
        "motif",
        "kumas",
        "kumaş",
        "fabric",
        "doku",
        "dokusu",
        "hayvan",
        "hayvani",
        "fotograf",
        "fotografi",
        "photo",
        "kucuk",
        "küçük",
        "buyuk",
        "büyük",
        "mavi",
        "siyah",
        "beyaz",
        "kirmizi",
        "kırmızı",
        "yesil",
        "yeşil",
        "krem",
        "bej",
        "gri",
        "cicek",
        "çiçek",
        "gul",
        "gül",
        "kaplan",
        "leopar",
        "zebra",
        "yogun",
        "yoğun",
        "seyrek",
    }
)


def peel_scope_suffix(token: str) -> str:
    """'tekstil'deki' / 'tekstildeki' → 'tekstil'."""
    t = normalize_turkish(str(token or "")).strip()
    if not t:
        return ""
    return _SCOPE_SUFFIX_RE.sub("", t).strip(" '\"`´")


def extract_customer_from_query(
    query: str,
    registry: CustomerRegistry | None,
    *,
    min_high: float = _HIGH_CONFIDENCE,
    min_suggest: float = _MIN_SUGGEST_SCORE,
    ambiguity_delta: float = 0.05,
) -> dict[str, Any]:
    """NL cümlesinden müşteri span'i (fuzzy). Belirsizde zorlama yok.

    Returns keys: customer, score, high_confidence, ambiguous, matched_span,
    search_text, scope_prefixes, candidates.
    """
    raw = str(query or "").strip()
    empty = {
        "customer": "",
        "score": 0.0,
        "high_confidence": False,
        "ambiguous": False,
        "matched_span": "",
        "search_text": raw,
        "scope_prefixes": [],
        "candidates": [],
    }
    if not raw or registry is None or not getattr(registry, "discovered_count", 0):
        # Empty registry: still try names() if entries exist without count
        if not raw or registry is None:
            return empty
        if not registry.entries():
            return empty

    # Tokenize preserving original indices; peel locatives for matching only.
    norm = normalize_turkish(raw)
    # Keep apostrophe tokens intact for span removal against raw-ish norm text.
    toks = [t for t in re.split(r"\s+", norm) if t]
    if not toks:
        return empty

    peeled = [peel_scope_suffix(t) or t for t in toks]
    n = len(toks)
    candidates: list[tuple[float, str, str, tuple[int, int]]] = []
    # Longest spans first (1..4 tokens)
    for length in range(min(4, n), 0, -1):
        for i in range(0, n - length + 1):
            span_peeled = peeled[i : i + length]
            if any(not p or p in _NON_CUSTOMER_TOKENS for p in span_peeled):
                # Allow multi-token names that include "tekstil" etc.; only skip
                # if ALL tokens are non-customer, or single-token is blocked.
                if length == 1:
                    continue
                if all(p in _NON_CUSTOMER_TOKENS for p in span_peeled):
                    continue
            span_q = " ".join(span_peeled)
            if len(span_q) < _MIN_CUSTOMER_LEN:
                continue
            best_name = ""
            best_sc = 0.0
            for entry in registry.entries():
                sc = score_customer_query(span_q, entry.name)
                if sc > best_sc:
                    best_sc = sc
                    best_name = entry.name
            if best_sc >= min_suggest and best_name:
                candidates.append((best_sc, best_name, span_q, (i, i + length)))

    if not candidates:
        return empty

    candidates.sort(key=lambda x: (-x[0], -(x[3][1] - x[3][0]), x[1].casefold()))
    # Dedupe by customer name keeping best
    seen: set[str] = set()
    uniq: list[tuple[float, str, str, tuple[int, int]]] = []
    for c in candidates:
        if c[1] in seen:
            continue
        seen.add(c[1])
        uniq.append(c)
    top = uniq[0]
    ambiguous = False
    if len(uniq) >= 2:
        second = uniq[1]
        if (
            second[0] >= min_high
            and top[0] - second[0] < ambiguity_delta
            and second[1] != top[1]
        ):
            ambiguous = True

    score, name, span_q, (i0, i1) = top
    high = (not ambiguous) and score >= min_high
    # Safer: below high confidence → do not force customer filter
    if ambiguous or not high:
        return {
            "customer": name if high else "",
            "score": score,
            "high_confidence": False,
            "ambiguous": ambiguous,
            "matched_span": span_q if high else "",
            "search_text": raw,
            "scope_prefixes": [],
            "candidates": [
                {"name": n, "score": s, "span": sp} for s, n, sp, _ in uniq[:5]
            ],
        }

    # Drop matched token range (+ trailing bare locative residue if any)
    keep_idx = [j for j in range(n) if j < i0 or j >= i1]
    # Also drop orphan "deki/daki" if somehow split
    residual_toks = []
    for j in keep_idx:
        t = toks[j]
        peeled_t = peel_scope_suffix(t)
        if normalize_turkish(t) in {"deki", "daki"} and not peeled_t:
            continue
        residual_toks.append(t if peeled_t == normalize_turkish(t) else peeled_t or t)
    # Rebuild search_text from residual (normalized spacing)
    search_text = " ".join(t for t in residual_toks if t).strip()
    if not search_text:
        search_text = raw

    entry = registry.get(name)
    prefixes = entry.all_scope_prefixes() if entry else []
    return {
        "customer": name,
        "score": score,
        "high_confidence": True,
        "ambiguous": False,
        "matched_span": span_q,
        "search_text": search_text,
        "scope_prefixes": prefixes,
        "candidates": [
            {"name": n, "score": s, "span": sp} for s, n, sp, _ in uniq[:5]
        ],
    }
