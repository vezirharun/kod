"""Vezir Pattern Foundation Dataset Builder v1.0

116k+ desenden triplet / pair eğitim seti üretir.
Knowledge Graph + Pattern DNA + Visual (phash) + aile etiketleri kullanır.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from core.logger import setup_logger

logger = setup_logger(__name__)

# Varsayılan eşikler
DEFAULT_MIN_CONFIDENCE = 0.80
DEFAULT_HIGH_CONFIDENCE = 0.95
DEFAULT_POSITIVES_PER_ANCHOR = 4
DEFAULT_HARD_POSITIVES = 2
DEFAULT_HARD_NEGATIVES = 2
DEFAULT_NEGATIVES = 3
DEFAULT_MAX_TRIPLETS = 2_000_000


@dataclass
class FoundationLabels:
    pattern_family: str = "unknown"
    sub_family: str = ""
    motif: str = ""
    repeat: str = ""
    color: str = ""
    style: str = ""
    texture: str = ""
    material: str = ""
    season: str = ""
    collection: str = ""
    brand_language: str = ""


@dataclass
class ClusterNode:
    cluster_id: str
    label: str
    parent_id: str = ""
    root_family: str = ""
    member_ids: list[int] = field(default_factory=list)
    size: int = 0


@dataclass
class DatasetRecord:
    file_id: int
    path: str
    thumbnail_path: str
    filename: str
    confidence: float
    labels: FoundationLabels
    cluster_id: str
    phash: str = ""


@dataclass
class FoundationBenchmark:
    anchors: int = 0
    triplets: int = 0
    positives: int = 0
    hard_positives: int = 0
    hard_negatives: int = 0
    negatives: int = 0
    pairs: int = 0
    clusters: int = 0
    average_confidence: float = 0.0
    skipped_low_confidence: int = 0
    elapsed_sec: float = 0.0
    output_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _hamming_hex(a: str, b: str) -> int:
    """Hex phash Hamming; uyumsuz uzunlukta büyük mesafe."""
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b or len(a) != len(b):
        return 64
    try:
        x = int(a, 16) ^ int(b, 16)
        return x.bit_count()
    except ValueError:
        return 64


def _phash_sim(a: str, b: str) -> float:
    dist = _hamming_hex(a, b)
    bits = max(len(a or "") * 4, 1)
    return max(0.0, 1.0 - dist / bits)


def compute_confidence(row: dict[str, Any], tm: dict[str, Any]) -> float:
    """0–1 dataset confidence — düşük olanlar elenir."""
    fam = str(row.get("pattern_family") or tm.get("pattern_family") or "").strip()
    conf = float(
        tm.get("classification_confidence")
        or tm.get("confidence")
        or row.get("classification_confidence")
        or 0.0
    )
    score = 0.35
    if fam and fam not in ("unknown", ""):
        score += 0.28
    if conf > 0:
        score += 0.25 * min(1.0, conf)
    else:
        # family biliniyorsa orta güven
        if fam and fam not in ("unknown", ""):
            score += 0.12
    if row.get("thumbnail_path"):
        score += 0.08
    if row.get("phash"):
        score += 0.08
    if tm.get("pattern_dna") or tm.get("animal_print_type") or tm.get("pattern_subtype"):
        score += 0.06
    if tm.get("color_family") or row.get("color_family"):
        score += 0.05
    return round(min(1.0, score), 4)


def extract_labels(row: dict[str, Any], tm: dict[str, Any]) -> FoundationLabels:
    dna = tm.get("pattern_dna") if isinstance(tm.get("pattern_dna"), dict) else {}
    fam = str(
        row.get("pattern_family") or tm.get("pattern_family") or dna.get("family") or "unknown"
    ).strip() or "unknown"
    sub = str(
        tm.get("pattern_subtype")
        or row.get("pattern_type")
        or tm.get("animal_print_type")
        or row.get("animal_print_type")
        or dna.get("subfamily")
        or ""
    ).strip()
    motif = str(dna.get("motif") or dna.get("motif_class") or sub or "").strip()
    repeat = str(dna.get("repeat") or dna.get("repeat_class") or tm.get("repeat") or "").strip()
    color = str(
        row.get("color_family") or tm.get("color_family") or dna.get("color_family") or ""
    ).strip()
    style = str(
        dna.get("designer_style") or dna.get("style") or dna.get("brand_style") or ""
    ).strip()
    texture = str(dna.get("texture") or dna.get("texture_class") or tm.get("texture_family") or "").strip()
    material = str(dna.get("material") or dna.get("material_hint") or "").strip()
    season = str(dna.get("season") or "").strip()
    collection = str(dna.get("collection") or "").strip()
    brand = str(dna.get("brand_style") or dna.get("brand") or "").strip()

    # Knowledge graph ile stil/motif zenginleştir
    try:
        from core.textile_knowledge_base import root_family as kb_root

        fam_root = kb_root(fam)
    except Exception:
        fam_root = fam

    if not style and fam_root == "animal_print":
        # brown leopard → luxury/safari ipucu
        blob = f"{row.get('filename','')} {color} {sub}".lower()
        if any(x in blob for x in ("luxe", "luxury", "versace", "scarf", "bordür", "border")):
            style = "luxury"
        elif any(x in blob for x in ("safari", "jungle")):
            style = "safari"
    if not motif and sub:
        motif = sub

    return FoundationLabels(
        pattern_family=fam_root if fam_root else fam,
        sub_family=sub,
        motif=motif,
        repeat=repeat,
        color=color,
        style=style,
        texture=texture,
        material=material,
        season=season,
        collection=collection,
        brand_language=brand,
    )


def cluster_key(labels: FoundationLabels, *, fine: bool = True) -> str:
    """Hiyerarşik küme anahtarı: family / sub / color / style."""
    parts = [labels.pattern_family or "unknown"]
    if labels.sub_family:
        parts.append(labels.sub_family)
    if fine and labels.color:
        parts.append(labels.color)
    if fine and labels.style:
        parts.append(labels.style)
    # Mini/macro scale from motif name
    m = (labels.motif or "").lower()
    if fine:
        if any(x in m for x in ("mini", "micro", "ditsy", "small")):
            parts.append("mini")
        elif any(x in m for x in ("macro", "jumbo", "big", "oversized")):
            parts.append("macro")
    return "/".join(p for p in parts if p)


def parent_cluster_key(key: str) -> str:
    parts = key.split("/")
    if len(parts) <= 1:
        return parts[0] if parts else "unknown"
    return "/".join(parts[:-1])


def root_of_key(key: str) -> str:
    return (key or "unknown").split("/")[0]


class FoundationDatasetBuilder:
    """DB → cluster → triplet/pair export."""

    def __init__(
        self,
        db,
        *,
        output_dir: str | Path,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        high_confidence: float = DEFAULT_HIGH_CONFIDENCE,
        positives_per_anchor: int = DEFAULT_POSITIVES_PER_ANCHOR,
        hard_positives: int = DEFAULT_HARD_POSITIVES,
        hard_negatives: int = DEFAULT_HARD_NEGATIVES,
        negatives: int = DEFAULT_NEGATIVES,
        max_triplets: int = DEFAULT_MAX_TRIPLETS,
        max_anchors: int | None = None,
        seed: int = 42,
        materialize_thumbs: bool = False,
        prefer_high_confidence: bool = True,
    ):
        self.db = db
        self.output_dir = Path(output_dir)
        self.min_confidence = float(min_confidence)
        self.high_confidence = float(high_confidence)
        self.positives_per_anchor = int(positives_per_anchor)
        self.hard_positives = int(hard_positives)
        self.hard_negatives = int(hard_negatives)
        self.negatives = int(negatives)
        self.max_triplets = int(max_triplets)
        self.max_anchors = max_anchors
        self.rng = random.Random(seed)
        self.materialize_thumbs = bool(materialize_thumbs)
        self.prefer_high_confidence = bool(prefer_high_confidence)

        self.records: dict[int, DatasetRecord] = {}
        self.clusters: dict[str, ClusterNode] = {}
        self.by_cluster: dict[str, list[int]] = defaultdict(list)
        self.by_root: dict[str, list[int]] = defaultdict(list)
        self.by_parent: dict[str, list[int]] = defaultdict(list)
        self._phash_bucket: dict[str, list[int]] = defaultdict(list)

    # ------------------------------------------------------------------ load
    def load_records(self, *, limit: int | None = None) -> int:
        sql = """
            SELECT f.id, f.path, f.filename, f.thumbnail_path, f.pattern_family,
                   f.pattern_type,
                   fe.phash, fe.texture_map
            FROM files f
            LEFT JOIN features fe ON fe.file_id = f.id
            WHERE f.status != 'missing'
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self.db.connect() as conn:
            rows = conn.execute(sql).fetchall()
        skipped = 0
        for row in rows:
            d = dict(row)
            tm = _safe_json(d.get("texture_map"))
            conf = compute_confidence(d, tm)
            if conf < self.min_confidence:
                skipped += 1
                continue
            if self.prefer_high_confidence and conf < self.high_confidence:
                # Orta güven: aile biliniyorsa al, değilse atla
                fam = str(d.get("pattern_family") or tm.get("pattern_family") or "")
                if not fam or fam == "unknown":
                    skipped += 1
                    continue
            labels = extract_labels(d, tm)
            ck = cluster_key(labels)
            rec = DatasetRecord(
                file_id=int(d["id"]),
                path=str(d.get("path") or ""),
                thumbnail_path=str(d.get("thumbnail_path") or ""),
                filename=str(d.get("filename") or ""),
                confidence=conf,
                labels=labels,
                cluster_id=ck,
                phash=str(d.get("phash") or ""),
            )
            self.records[rec.file_id] = rec
            self.by_cluster[ck].append(rec.file_id)
            self.by_root[root_of_key(ck)].append(rec.file_id)
            self.by_parent[parent_cluster_key(ck)].append(rec.file_id)
            if rec.phash and len(rec.phash) >= 4:
                self._phash_bucket[rec.phash[:4]].append(rec.file_id)
        self._skipped_low = skipped
        logger.info(
            "Foundation load: kept=%s skipped=%s clusters=%s",
            len(self.records),
            skipped,
            len(self.by_cluster),
        )
        return len(self.records)

    def build_clusters(self) -> list[ClusterNode]:
        nodes: dict[str, ClusterNode] = {}
        for ck, ids in self.by_cluster.items():
            parent = parent_cluster_key(ck)
            root = root_of_key(ck)
            # Enrich label via knowledge graph
            label = ck.replace("/", " / ").title()
            try:
                from core.knowledge_graph import expand_query

                seeds = [root]
                if "/" in ck:
                    seeds.append(ck.split("/")[1])
                exp = expand_query(" ".join(seeds), family=root, max_hops=1)
                if exp.expanded:
                    label = " → ".join(
                        [exp.expanded[0].label]
                        + [h.label for h in exp.expanded[1:3]]
                    )
            except Exception:
                pass
            nodes[ck] = ClusterNode(
                cluster_id=ck,
                label=label or ck,
                parent_id=parent if parent != ck else "",
                root_family=root,
                member_ids=list(ids),
                size=len(ids),
            )
        self.clusters = nodes
        return list(nodes.values())

    # --------------------------------------------------------------- sample
    def _pick(self, pool: list[int], n: int, exclude: set[int]) -> list[int]:
        cand = [i for i in pool if i not in exclude and i in self.records]
        if not cand:
            return []
        if len(cand) <= n:
            return list(cand)
        return self.rng.sample(cand, n)

    def _hard_negatives_for(self, anchor: DatasetRecord, n: int, exclude: set[int]) -> list[int]:
        """Aynı phash bucket / yüksek benzerlik ama farklı kök aile."""
        out: list[int] = []
        root = root_of_key(anchor.cluster_id)
        bucket = self._phash_bucket.get((anchor.phash or "")[:4], [])
        scored: list[tuple[float, int]] = []
        for fid in bucket:
            if fid in exclude or fid == anchor.file_id:
                continue
            other = self.records.get(fid)
            if not other:
                continue
            if root_of_key(other.cluster_id) == root:
                continue
            sim = _phash_sim(anchor.phash, other.phash)
            if sim >= 0.70:
                scored.append((sim, fid))
        scored.sort(reverse=True)
        for _sim, fid in scored[:n]:
            out.append(fid)
        if len(out) < n:
            # Farklı aile, rastgele ama parent yakın değil
            foreign_roots = [r for r in self.by_root if r != root and r != "unknown"]
            self.rng.shuffle(foreign_roots)
            for r in foreign_roots:
                need = n - len(out)
                if need <= 0:
                    break
                out.extend(self._pick(self.by_root[r], need, exclude | set(out)))
        return out[:n]

    def iter_triplets(self) -> Iterator[dict[str, Any]]:
        anchors = list(self.records.values())
        # Yüksek güven önce
        anchors.sort(key=lambda r: (-r.confidence, r.file_id))
        if self.max_anchors:
            anchors = anchors[: int(self.max_anchors)]

        produced = 0
        for anchor in anchors:
            if produced >= self.max_triplets:
                return
            same = [i for i in self.by_cluster.get(anchor.cluster_id, []) if i != anchor.file_id]
            parent = parent_cluster_key(anchor.cluster_id)
            parent_pool = [
                i
                for i in self.by_parent.get(parent, [])
                if i != anchor.file_id and self.records[i].cluster_id != anchor.cluster_id
            ]
            exclude = {anchor.file_id}

            positives = self._pick(same, self.positives_per_anchor, exclude)
            exclude |= set(positives)
            hard_pos = self._pick(parent_pool, self.hard_positives, exclude)
            exclude |= set(hard_pos)
            hard_neg = self._hard_negatives_for(anchor, self.hard_negatives, exclude)
            exclude |= set(hard_neg)
            root = root_of_key(anchor.cluster_id)
            foreign = []
            for r, ids in self.by_root.items():
                if r == root or r == "unknown":
                    continue
                foreign.extend(ids)
            negatives = self._pick(foreign, self.negatives, exclude)

            if not positives and not hard_pos:
                continue

            def _ref(fid: int) -> dict[str, Any]:
                r = self.records[fid]
                return {
                    "file_id": fid,
                    "path": r.path,
                    "thumbnail_path": r.thumbnail_path,
                    "filename": r.filename,
                    "cluster_id": r.cluster_id,
                    "confidence": r.confidence,
                    "labels": asdict(r.labels),
                }

            base = {
                "anchor": _ref(anchor.file_id),
                "anchor_confidence": anchor.confidence,
                "cluster_id": anchor.cluster_id,
            }
            # Her positive × negative kombinasyonu (kontrollü)
            pos_all = [("positive", p) for p in positives] + [
                ("hard_positive", p) for p in hard_pos
            ]
            neg_all = [("hard_negative", n) for n in hard_neg] + [
                ("negative", n) for n in negatives
            ]
            if not pos_all or not neg_all:
                continue
            for pkind, pid in pos_all:
                for nkind, nid in neg_all:
                    if produced >= self.max_triplets:
                        return
                    yield {
                        **base,
                        "positive": _ref(pid),
                        "negative": _ref(nid),
                        "positive_kind": pkind,
                        "negative_kind": nkind,
                        "triplet_id": hashlib.md5(
                            f"{anchor.file_id}:{pid}:{nid}".encode()
                        ).hexdigest()[:16],
                    }
                    produced += 1

    def iter_pairs(self) -> Iterator[dict[str, Any]]:
        """Positive / negative pair listesi (contrastive learning)."""
        for t in self.iter_triplets():
            yield {
                "pair_id": t["triplet_id"] + "p",
                "anchor": t["anchor"],
                "other": t["positive"],
                "label": 1,
                "kind": t["positive_kind"],
            }
            yield {
                "pair_id": t["triplet_id"] + "n",
                "anchor": t["anchor"],
                "other": t["negative"],
                "label": 0,
                "kind": t["negative_kind"],
            }

    # --------------------------------------------------------------- export
    def _ensure_dirs(self) -> None:
        for name in ("anchor", "positive", "negative", "hard_positive", "hard_negative"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)

    def _materialize(self, kind: str, rec: DatasetRecord) -> str:
        if not self.materialize_thumbs:
            return rec.thumbnail_path or rec.path
        src = rec.thumbnail_path or rec.path
        if not src or not os.path.isfile(src):
            return src
        ext = Path(src).suffix or ".webp"
        dest = self.output_dir / kind / f"{rec.file_id}{ext}"
        if not dest.exists():
            try:
                shutil.copy2(src, dest)
            except OSError:
                return src
        return str(dest)

    def export(self) -> FoundationBenchmark:
        t0 = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_dirs()

        if not self.records:
            self.load_records()
        if not self.clusters:
            self.build_clusters()

        triplets_path = self.output_dir / "triplets.jsonl"
        pairs_path = self.output_dir / "pairs.jsonl"
        clusters_path = self.output_dir / "clusters.jsonl"
        meta_path = self.output_dir / "metadata.json"

        n_trip = n_pos = n_hp = n_hn = n_neg = n_pair = 0
        conf_sum = 0.0
        conf_n = 0
        anchors_seen: set[int] = set()

        with triplets_path.open("w", encoding="utf-8") as ft, pairs_path.open(
            "w", encoding="utf-8"
        ) as fp:
            for t in self.iter_triplets():
                if self.materialize_thumbs:
                    a = self.records[t["anchor"]["file_id"]]
                    p = self.records[t["positive"]["file_id"]]
                    n = self.records[t["negative"]["file_id"]]
                    t["anchor"]["export_path"] = self._materialize("anchor", a)
                    pk = "hard_positive" if t["positive_kind"] == "hard_positive" else "positive"
                    nk = "hard_negative" if t["negative_kind"] == "hard_negative" else "negative"
                    t["positive"]["export_path"] = self._materialize(pk, p)
                    t["negative"]["export_path"] = self._materialize(nk, n)
                ft.write(json.dumps(t, ensure_ascii=False) + "\n")
                n_trip += 1
                anchors_seen.add(int(t["anchor"]["file_id"]))
                conf_sum += float(t["anchor_confidence"])
                conf_n += 1
                if t["positive_kind"] == "hard_positive":
                    n_hp += 1
                else:
                    n_pos += 1
                if t["negative_kind"] == "hard_negative":
                    n_hn += 1
                else:
                    n_neg += 1

                for label, other, kind in (
                    (1, t["positive"], t["positive_kind"]),
                    (0, t["negative"], t["negative_kind"]),
                ):
                    pair = {
                        "pair_id": f"{t['triplet_id']}_{label}",
                        "anchor": t["anchor"],
                        "other": other,
                        "label": label,
                        "kind": kind,
                    }
                    fp.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    n_pair += 1

        with clusters_path.open("w", encoding="utf-8") as fc:
            for node in sorted(self.clusters.values(), key=lambda c: (-c.size, c.cluster_id)):
                fc.write(
                    json.dumps(
                        {
                            "cluster_id": node.cluster_id,
                            "label": node.label,
                            "parent_id": node.parent_id,
                            "root_family": node.root_family,
                            "size": node.size,
                            "member_ids": node.member_ids[:500],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        bench = FoundationBenchmark(
            anchors=len(anchors_seen),
            triplets=n_trip,
            positives=n_pos,
            hard_positives=n_hp,
            hard_negatives=n_hn,
            negatives=n_neg,
            pairs=n_pair,
            clusters=len(self.clusters),
            average_confidence=round(conf_sum / conf_n, 4) if conf_n else 0.0,
            skipped_low_confidence=int(getattr(self, "_skipped_low", 0)),
            elapsed_sec=round(time.time() - t0, 2),
            output_dir=str(self.output_dir),
        )
        meta = {
            "version": "1.0",
            "name": "Vezir Pattern Foundation Dataset",
            "min_confidence": self.min_confidence,
            "high_confidence": self.high_confidence,
            "records_kept": len(self.records),
            "benchmark": bench.to_dict(),
            "label_schema": [
                "pattern_family",
                "sub_family",
                "motif",
                "repeat",
                "color",
                "style",
                "texture",
                "material",
                "season",
                "collection",
                "brand_language",
            ],
            "triplet_schema": {
                "anchor": "reference pattern",
                "positive": "same fine cluster",
                "hard_positive": "same parent cluster, different fine cluster",
                "hard_negative": "visually similar phash, different root family",
                "negative": "unrelated root family",
            },
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "benchmark.json").write_text(
            json.dumps(bench.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Foundation export done: triplets=%s pairs=%s avg_conf=%.3f → %s",
            n_trip,
            n_pair,
            bench.average_confidence,
            self.output_dir,
        )
        return bench


def build_foundation_dataset(
    db,
    output_dir: str | Path,
    *,
    load_limit: int | None = None,
    **kwargs: Any,
) -> FoundationBenchmark:
    builder = FoundationDatasetBuilder(db, output_dir=output_dir, **kwargs)
    builder.load_records(limit=load_limit)
    builder.build_clusters()
    return builder.export()
