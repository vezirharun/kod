"""Otonom sınıflandırma: eminsen öğren, emin değilsen sor.

FAISS/DINO/CLIP/OWLv2 eğitmez. Yalnızca kullanıcı doğrulamalı CLIP örnekleri
eşik MATCH_OK ile karşılaştırılır. Auto kayıtlar prototip olmaz.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from core.concept_registry import (
    _conn,
    _now,
    add_example,
    dismissed_file_ids,
    dismiss_file,
    pending_review_file_ids,
    record_learning_event,
    taught_file_ids,
    upsert,
    user_positive_file_ids,
)
from core.learned_concept_search import _NEIGHBOR_FLOOR, max_clip_to_examples
from core.manual_label_guard import is_manual_labeled, parse_texture_map
from core.teach_me import (
    MATCH_OK,
    TeachMeCard,
    _clip_blobs_for_files,
    _cosine,
    correct_concept_membership,
    match_clip_to_concepts,
    teach_files,
)

# Mevcut CLIP öğreti eşiği — düşürülmez.
HIGH_MIN = MATCH_OK
# learned_concept_search komşu tabanı — yeni eşik uydurulmaz.
LOW_MIN = _NEIGHBOR_FLOOR
CLUSTER_MIN = 3
# Rakip concept skoru tepeye bu kadar yakınsa AUTO değil CANDIDATE.
RIVAL_MARGIN = 0.05
# Negative boundary CLIP benzerliği ≥ bu → AUTO engel.
NEG_BLOCK = _NEIGHBOR_FLOOR


def decide_lane(hits: list[dict[str, Any]]) -> tuple[str, str]:
    """high | undecided | low | none. hits score azalan."""
    ranked = [h for h in hits if float(h.get("score") or 0) > 0]
    ranked.sort(key=lambda h: -float(h["score"]))
    strong = [h for h in ranked if float(h["score"]) >= HIGH_MIN]
    near = [h for h in ranked if float(h["score"]) >= LOW_MIN]
    if len(strong) == 1:
        name = str(strong[0].get("canonical") or "")
        pct = int(round(float(strong[0]["score"]) * 100))
        return "high", f"Tek güçlü kavram: {name} %{pct}"
    if len(strong) >= 2:
        detail = ", ".join(
            f"{h.get('canonical')} %{int(round(float(h['score']) * 100))}"
            for h in strong[:3]
        )
        return "undecided", f"Birden fazla güçlü tahmin: {detail}"
    if near:
        top = near[0]
        pct = int(round(float(top["score"]) * 100))
        return "low", f"Düşük güven: {top.get('canonical')} %{pct} — kullanıcı onayı gerekir"
    return "none", ""


def self_check_decision(
    db_path: str,
    hits: list[dict[str, Any]],
    *,
    clip_blob: bytes | None = None,
    file_id: int = 0,
) -> dict[str, Any]:
    """AUTO / CANDIDATE / IGNORE — user evidence ve boundary korunur."""
    ranked = [h for h in (hits or []) if float(h.get("score") or 0) > 0]
    ranked.sort(key=lambda h: -float(h["score"]))
    empty = {
        "decision": "ignore",
        "lane": "none",
        "reason": "no_hits",
        "top": {},
        "rival": {},
        "concept_id": 0,
        "rival_concept_id": 0,
    }
    if not ranked:
        return empty
    top = ranked[0]
    top_sc = float(top.get("score") or 0)
    second = ranked[1] if len(ranked) > 1 else None
    second_sc = float(second.get("score") or 0) if second else 0.0
    cid = int(top.get("concept_id") or 0)
    rival_id = int(second.get("concept_id") or 0) if second else 0

    # Kullanıcı başka concept'e verified yazmışsa otomatik öğrenme yok.
    if int(file_id or 0) > 0:
        try:
            from core.concept_registry import positives_for_file

            for row in positives_for_file(db_path, int(file_id)):
                if str(row.get("source") or "user") not in ("auto", "autonomous", "candidate"):
                    return {
                        "decision": "ignore",
                        "lane": "none",
                        "reason": "user_verified_exists",
                        "top": top,
                        "rival": second or {},
                        "concept_id": cid,
                        "rival_concept_id": rival_id,
                    }
        except Exception:
            pass

    # Negative boundary: önerilen concept'in negative'lerine yakınsa AUTO yok.
    if clip_blob and cid > 0:
        try:
            from core.concept_registry import negative_example_vectors

            neg_blobs = [
                bytes(ex["embedding"])
                for ex in negative_example_vectors(db_path, cid)
                if ex.get("embedding")
            ]
            if neg_blobs:
                neg_sim = max_clip_to_examples(clip_blob, neg_blobs)
                if neg_sim >= NEG_BLOCK:
                    return {
                        "decision": "candidate",
                        "lane": "undecided",
                        "reason": f"negative_boundary_close:{neg_sim:.3f}",
                        "top": top,
                        "rival": second or {},
                        "concept_id": cid,
                        "rival_concept_id": rival_id,
                    }
        except Exception:
            pass

    lane, lane_reason = decide_lane(ranked)
    # Açık ara: tek HIGH ve rakip margin dışında.
    if lane == "high" and top_sc >= HIGH_MIN:
        if second and second_sc >= (top_sc - RIVAL_MARGIN) and second_sc >= LOW_MIN:
            return {
                "decision": "candidate",
                "lane": "undecided",
                "reason": f"rival_close:{top.get('canonical')}:{top_sc:.3f}/{second.get('canonical')}:{second_sc:.3f}",
                "top": top,
                "rival": second,
                "concept_id": cid,
                "rival_concept_id": rival_id,
            }
        return {
            "decision": "auto",
            "lane": "high",
            "reason": lane_reason or "clear_leader",
            "top": top,
            "rival": second or {},
            "concept_id": cid,
            "rival_concept_id": rival_id,
        }
    if lane == "undecided":
        return {
            "decision": "candidate",
            "lane": "undecided",
            "reason": lane_reason or "multi_strong",
            "top": top,
            "rival": second or {},
            "concept_id": cid,
            "rival_concept_id": rival_id,
        }
    if lane == "low":
        return {
            "decision": "candidate",
            "lane": "low",
            "reason": lane_reason or "medium_confidence",
            "top": top,
            "rival": second or {},
            "concept_id": cid,
            "rival_concept_id": rival_id,
        }
    return {
        "decision": "ignore",
        "lane": "none",
        "reason": lane_reason or "below_threshold",
        "top": top,
        "rival": second or {},
        "concept_id": cid,
        "rival_concept_id": rival_id,
    }


def _skip_ids(db_path: str) -> set[int]:
    # User/auto evidence + dismiss + açık pending review. Candidate örnekleri inbox'ta kalır.
    return (
        dismissed_file_ids(db_path)
        | taught_file_ids(db_path)
        | user_positive_file_ids(db_path)
        | pending_review_file_ids(db_path)
    )


def _manual_or_user(tm: dict[str, Any]) -> bool:
    if is_manual_labeled(tm):
        return True
    src = str(tm.get("user_label_source") or "").strip().lower()
    return src in ("teach_me", "manual_user", "user")


def _rivals_payload(hits: list[dict[str, Any]], *, min_score: float) -> list[list]:
    out = []
    for h in hits:
        sc = float(h.get("score") or 0)
        if sc < min_score:
            continue
        out.append([str(h.get("canonical") or ""), round(sc, 4)])
    return out[:5]


def undecided_candidate_labels(
    rivals: list | None,
    suggested: str = "",
    *,
    max_n: int = 5,
) -> list[str]:
    """Kararsız one-click adayları — yalnızca gerçek sinyal (rivals / suggested).

    Rastgele kavram uydurmaz. Skor sırası korunur; boş/Belirsiz elenir.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        name = " ".join(str(raw or "").strip().split())
        if not name:
            return
        if name.casefold() in ("belirsiz", "yeni kavram"):
            return
        if " / " in name:
            name = name.split(" / ")[0].strip()
        key = name.casefold()
        if not name or key in seen:
            return
        seen.add(key)
        out.append(name)

    for item in rivals or []:
        if isinstance(item, (list, tuple)) and item:
            _add(str(item[0] or ""))
        elif isinstance(item, str):
            _add(item)
        if len(out) >= max_n:
            return out[:max_n]
    _add(str(suggested or ""))
    return out[:max_n]


def upsert_review(
    db_path: str,
    file_id: int,
    *,
    lane: str,
    suggested: str = "",
    confidence: float = 0.0,
    rivals: list | None = None,
    reason: str = "",
    cluster_id: int = 0,
    status: str = "pending",
) -> None:
    if not db_path or int(file_id or 0) <= 0:
        return
    c = _conn(db_path)
    now = _now()
    c.execute(
        """INSERT INTO autonomous_review(
            file_id,lane,suggested,confidence,rivals_json,reason,cluster_id,status,created_at,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(file_id) DO UPDATE SET
            lane=excluded.lane, suggested=excluded.suggested, confidence=excluded.confidence,
            rivals_json=excluded.rivals_json, reason=excluded.reason,
            cluster_id=excluded.cluster_id, status=excluded.status, updated_at=excluded.updated_at
        """,
        (
            int(file_id),
            str(lane),
            str(suggested or ""),
            float(confidence or 0),
            json.dumps(rivals or [], ensure_ascii=False),
            str(reason or ""),
            int(cluster_id or 0),
            str(status),
            now,
            now,
        ),
    )
    c.commit()
    c.close()


def list_pending_reviews(db_path: str, *, lane: str = "") -> list[dict[str, Any]]:
    c = _conn(db_path)
    try:
        if lane:
            rows = c.execute(
                "SELECT * FROM autonomous_review WHERE status='pending' AND lane=? ORDER BY confidence DESC",
                (lane,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM autonomous_review WHERE status='pending' ORDER BY lane, confidence DESC"
            ).fetchall()
    except Exception:
        rows = []
    c.close()
    return [dict(r) for r in rows]


def review_for_file(db_path: str, file_id: int) -> dict[str, Any] | None:
    c = _conn(db_path)
    row = c.execute(
        "SELECT * FROM autonomous_review WHERE file_id=?", (int(file_id),)
    ).fetchone()
    c.close()
    return dict(row) if row else None


def cluster_file_ids(db_path: str, cluster_id: int) -> list[int]:
    if int(cluster_id or 0) <= 0:
        return []
    c = _conn(db_path)
    rows = c.execute(
        "SELECT file_id FROM autonomous_review WHERE cluster_id=? AND status='pending'",
        (int(cluster_id),),
    ).fetchall()
    c.close()
    return [int(r["file_id"]) for r in rows]


def _set_review_status(db_path: str, file_ids: list[int], status: str) -> None:
    if not file_ids:
        return
    c = _conn(db_path)
    now = _now()
    for fid in file_ids:
        c.execute(
            "UPDATE autonomous_review SET status=?, updated_at=? WHERE file_id=?",
            (status, now, int(fid)),
        )
    c.commit()
    c.close()


def _apply_auto_memory(
    db,
    db_path: str,
    file_id: int,
    label: str,
    *,
    score: float,
    blob: bytes | None,
    path: str,
    reason: str = "",
    rival_concept_id: int = 0,
    anchor_id: int = 0,
) -> int:
    """Hafızaya auto örnek yazar; CLIP prototip havuzuna girmez. User evidence ezilmez."""
    cid = upsert(
        db_path,
        label,
        concept_type="visual_concept",
        source="auto",
        confidence=float(score),
    )
    if cid:
        add_example(
            db_path,
            cid,
            file_id=file_id,
            file_path=path,
            role="positive",
            source="auto",
            embedding=blob,
            embedding_backend="clip" if blob else "",
        )
        record_learning_event(
            db_path,
            file_id=int(file_id),
            concept_id=int(cid),
            action="auto_evidence",
            confidence=float(score),
            source="auto",
            anchor_id=int(anchor_id or 0),
            rival_concept_id=int(rival_concept_id or 0),
            reason=str(reason or "clear_leader"),
        )
    tm = {}
    try:
        feat = db.get_features(int(file_id)) if hasattr(db, "get_features") else None
        raw = (feat or {}).get("texture_map") if isinstance(feat, dict) else None
        tm = parse_texture_map(raw)
    except Exception:
        tm = {}
    tm["autonomous_label"] = label
    tm["autonomous_confidence"] = float(score)
    tm["autonomous_source"] = "clip_memory"
    tm["learned_concept"] = label
    try:
        db.upsert_texture_map(int(file_id), tm)
    except Exception:
        try:
            db.upsert_features(int(file_id), {"texture_map": tm})
        except Exception:
            pass
    return int(cid or 0)


def _apply_candidate_memory(
    db_path: str,
    file_id: int,
    label: str,
    *,
    score: float,
    blob: bytes | None,
    path: str,
    reason: str = "",
    rival_concept_id: int = 0,
    anchor_id: int = 0,
    lane: str = "undecided",
) -> int:
    """Candidate evidence — verified değil, matching prototipi değil."""
    cid = upsert(
        db_path,
        label,
        concept_type="visual_concept",
        source="candidate",
        confidence=float(score),
    )
    if cid:
        add_example(
            db_path,
            cid,
            file_id=file_id,
            file_path=path,
            role="positive",
            source="candidate",
            embedding=blob,
            embedding_backend="clip" if blob else "",
        )
        record_learning_event(
            db_path,
            file_id=int(file_id),
            concept_id=int(cid),
            action="candidate_created",
            confidence=float(score),
            source="candidate",
            anchor_id=int(anchor_id or 0),
            rival_concept_id=int(rival_concept_id or 0),
            reason=str(reason or "needs_review"),
        )
    upsert_review(
        db_path,
        int(file_id),
        lane=str(lane or "undecided"),
        suggested=str(label or ""),
        confidence=float(score),
        rivals=[],
        reason=str(reason or "candidate"),
        status="pending",
    )
    return int(cid or 0)


def _load_scan_rows(db, *, after_id: int, limit: int) -> list[dict[str, Any]]:
    sql = (
        "SELECT f.id, f.filename, f.path, f.thumbnail_path, f.feature_preview_path, "
        "f.pattern_family, f.pattern_confidence, f.needs_review, f.category_path, "
        "fe.texture_map, fe.clip_embedding "
        "FROM files f JOIN features fe ON fe.file_id=f.id "
        "WHERE f.id>? AND fe.clip_embedding IS NOT NULL AND length(fe.clip_embedding)>=16 "
        "ORDER BY f.id LIMIT ?"
    )
    try:
        with db.connect() as conn:
            rows = conn.execute(sql, (int(after_id), int(limit))).fetchall()
    except Exception:
        return []
    return [dict(r) for r in rows]


def _vec(blob: bytes | None) -> np.ndarray | None:
    if not blob or len(blob) < 16:
        return None
    try:
        v = np.frombuffer(blob, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if v.size == 0 or not np.isfinite(v).all():
        return None
    return v


def _suggest_cluster_name(filenames: list[str]) -> str:
    counts: dict[str, int] = {}
    for name in filenames:
        stem = re.sub(r"\.[^.]+$", "", str(name or "").lower())
        for tok in re.split(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ]+", stem):
            t = tok.strip()
            if len(t) < 3 or t.isdigit():
                continue
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return "Yeni kavram"
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    if best[1] < 2:
        return "Yeni kavram"
    return best[0]


def _discover_clusters(
    unmatched: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    leftover = list(unmatched)
    groups: list[list[dict[str, Any]]] = []
    while leftover:
        seed = leftover.pop(0)
        sv = seed.get("_vec")
        if sv is None:
            continue
        group = [seed]
        rest = []
        for rec in leftover:
            ov = rec.get("_vec")
            if ov is None:
                rest.append(rec)
                continue
            if _cosine(sv, ov) >= HIGH_MIN:
                group.append(rec)
            else:
                rest.append(rec)
        leftover = rest
        if len(group) >= CLUSTER_MIN:
            groups.append(group)
    return groups


def run_autonomous_pass(
    db,
    db_path: str,
    *,
    limit: int = 80,
    after_id: int = 0,
) -> dict[str, int]:
    """Bir dilim dosyayı sınıflandır. Düşük güveni gerçeğe yazmaz."""
    stats = {
        "scanned": 0,
        "high_applied": 0,
        "undecided": 0,
        "low": 0,
        "new_concept": 0,
        "skipped": 0,
        "candidates": 0,
        "ignored": 0,
    }
    skip = _skip_ids(db_path)
    rows = _load_scan_rows(db, after_id=after_id, limit=max(10, int(limit)))
    unmatched: list[dict[str, Any]] = []
    for rec in rows:
        fid = int(rec.get("id") or 0)
        stats["scanned"] += 1
        if fid <= 0 or fid in skip:
            stats["skipped"] += 1
            continue
        tm = parse_texture_map(rec.get("texture_map"))
        if _manual_or_user(tm):
            stats["skipped"] += 1
            continue
        if str(tm.get("autonomous_label") or "").strip() and float(
            tm.get("autonomous_confidence") or 0
        ) >= HIGH_MIN:
            stats["skipped"] += 1
            continue
        blob = rec.get("clip_embedding")
        if blob:
            blob = bytes(blob)
        hits = match_clip_to_concepts(db_path, blob)
        check = self_check_decision(db_path, hits, clip_blob=blob, file_id=fid)
        decision = str(check.get("decision") or "ignore")
        top = check.get("top") or (hits[0] if hits else {})
        label = str(top.get("canonical") or "")
        score = float(top.get("score") or 0)
        path = str(rec.get("path") or "")
        reason = str(check.get("reason") or "")
        rivals = _rivals_payload(hits, min_score=LOW_MIN)

        if decision == "auto" and label:
            _apply_auto_memory(
                db,
                db_path,
                fid,
                label,
                score=score,
                blob=blob,
                path=path,
                reason=reason,
                rival_concept_id=int(check.get("rival_concept_id") or 0),
            )
            upsert_review(
                db_path,
                fid,
                lane="high",
                suggested=label,
                confidence=score,
                rivals=rivals,
                reason=reason,
                status="applied",
            )
            stats["high_applied"] += 1
            continue

        if decision == "candidate" and label:
            _apply_candidate_memory(
                db_path,
                fid,
                label,
                score=score,
                blob=blob,
                path=path,
                reason=reason,
                rival_concept_id=int(check.get("rival_concept_id") or 0),
                lane=str(check.get("lane") or "undecided"),
            )
            # Rivals payload'ı review'a yaz (candidate helper boş bıraktı)
            upsert_review(
                db_path,
                fid,
                lane=str(check.get("lane") or "undecided"),
                suggested=label,
                confidence=score,
                rivals=rivals,
                reason=reason,
                status="pending",
            )
            stats["candidates"] += 1
            if str(check.get("lane")) == "low":
                stats["low"] += 1
            else:
                stats["undecided"] += 1
            continue

        if decision == "ignore":
            # Düşük ama sıfır olmayan CLIP gürültüsü yeni kavram keşfini engellemesin.
            if any(float(h.get("score") or 0) >= LOW_MIN for h in (hits or [])):
                stats["ignored"] += 1
                continue
            # Eşik altı / hiç hit → unmatched küme adayına bırak.

        vec = _vec(blob)
        if vec is None:
            stats["skipped"] += 1
            continue
        rec["_vec"] = vec
        rec["_blob"] = blob
        unmatched.append(rec)

    for group in _discover_clusters(unmatched):
        name = _suggest_cluster_name([str(r.get("filename") or "") for r in group])
        c = _conn(db_path)
        cur = c.execute(
            """INSERT INTO autonomous_clusters(suggested_name,file_count,status,reason,created_at)
               VALUES(?,?,?,?,?)""",
            (
                name,
                len(group),
                "pending",
                f"{len(group)} görsel birbirine MATCH_OK ile benziyor; mevcut kavrama uymuyor",
                _now(),
            ),
        )
        cid = int(cur.lastrowid)
        c.commit()
        c.close()
        reason = (
            f"Yeni kavram adayı «{name}» · {len(group)} görsel · "
            "kullanıcı onayı olmadan kaydedilmez"
        )
        for rec in group:
            upsert_review(
                db_path,
                int(rec["id"]),
                lane="new_concept",
                suggested=name,
                confidence=0.0,
                rivals=[],
                reason=reason,
                cluster_id=cid,
                status="pending",
            )
            stats["new_concept"] += 1
    return stats


def _card_from_review(rec: dict[str, Any], row: dict[str, Any], cluster_size: int) -> TeachMeCard:
    tm = parse_texture_map(rec.get("texture_map"))
    rivals = []
    try:
        rivals = json.loads(row.get("rivals_json") or "[]")
    except Exception:
        rivals = []
    guess = str(row.get("suggested") or "Belirsiz")
    if rivals and row.get("lane") == "undecided":
        guess = " / ".join(str(x[0]) for x in rivals[:3] if x)
    lane = str(row.get("lane") or "low")
    pool = {
        "undecided": "undecided",
        "low": "suspicious",
        "new_concept": "new_concept",
    }.get(lane, "suspicious")
    reason = str(row.get("reason") or "")
    if not reason:
        reason = {
            "undecided": "Birden fazla olası kavram / düşük ayrım var",
            "suspicious": "Sistem bir kavramdan şüpheleniyor",
            "new_concept": "Mevcut kavramlara uymayan yeni görsel kümesi",
        }.get(pool, "")
    return TeachMeCard(
        file_id=int(rec.get("id") or row.get("file_id") or 0),
        filename=str(rec.get("filename") or ""),
        path=str(rec.get("path") or ""),
        preview_path=str(rec.get("thumbnail_path") or rec.get("feature_preview_path") or ""),
        pool=pool,
        reason=reason,
        guess=guess,
        confidence=float(row.get("confidence") or 0),
        category=str(rec.get("category_path") or tm.get("category_path") or ""),
        feature_preview_path=str(rec.get("feature_preview_path") or ""),
        color_family=str(tm.get("color_family") or ""),
        pattern_family=str(rec.get("pattern_family") or tm.get("pattern_family") or ""),
        brand=str(tm.get("brand_name") or ""),
        tags=[],
        rivals=rivals,
        cluster_size=int(cluster_size or 1),
        suggested=str(row.get("suggested") or ""),
    )


def pending_review_cards(db, db_path: str) -> dict[str, list[TeachMeCard]]:
    pools: dict[str, list[TeachMeCard]] = {
        "undecided": [],
        "suspicious": [],
        "new_concept": [],
    }
    pending = list_pending_reviews(db_path)
    if not pending:
        return pools
    ids = [int(r["file_id"]) for r in pending if int(r.get("file_id") or 0) > 0]
    cluster_counts: dict[int, int] = {}
    for r in pending:
        cid = int(r.get("cluster_id") or 0)
        if cid:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
    recs: dict[int, dict[str, Any]] = {}
    if ids:
        ph = ",".join("?" * len(ids))
        try:
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT f.id, f.filename, f.path, f.thumbnail_path, f.feature_preview_path, "
                    "f.pattern_family, f.category_path, fe.texture_map "
                    "FROM files f LEFT JOIN features fe ON fe.file_id=f.id "
                    f"WHERE f.id IN ({ph})",
                    ids,
                ).fetchall()
            recs = {int(r["id"]): dict(r) for r in rows}
        except Exception:
            recs = {}
    for row in pending:
        fid = int(row.get("file_id") or 0)
        rec = recs.get(fid) or {"id": fid}
        csize = cluster_counts.get(int(row.get("cluster_id") or 0), 1)
        card = _card_from_review(rec, row, csize)
        bucket = pools.get(card.pool)
        if bucket is not None:
            bucket.append(card)
    return pools


def confirm_reviews(db, db_path: str, file_ids: list[int], *, label: str = "") -> dict[str, int]:
    """Doğru: önerilen (veya verilen) kavram kullanıcı doğrulaması olarak öğrenilir."""
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not ids:
        return {"taught": 0, "concept_id": 0}
    expand: list[int] = []
    names: list[str] = []
    for fid in ids:
        row = review_for_file(db_path, fid)
        if not row:
            expand.append(fid)
            continue
        cid = int(row.get("cluster_id") or 0)
        if cid:
            expand.extend(cluster_file_ids(db_path, cid) or [fid])
        else:
            expand.append(fid)
        names.append(str(label or row.get("suggested") or "").strip())
    uniq = list(dict.fromkeys(expand))
    name = str(label or "").strip() or next((n for n in names if n), "")
    if not name:
        return {"taught": 0, "concept_id": 0}
    stats = teach_files(db, db_path, uniq, name)
    _set_review_status(db_path, uniq, "accepted")
    return stats


def reject_reviews(db_path: str, file_ids: list[int]) -> int:
    """Yanlış: kesin kavram yazılmaz; öneri reddedilir."""
    n = 0
    seen: set[int] = set()
    for fid in file_ids:
        row = review_for_file(db_path, int(fid))
        targets = [int(fid)]
        if row and int(row.get("cluster_id") or 0) > 0:
            targets = cluster_file_ids(db_path, int(row["cluster_id"])) or targets
        for tid in targets:
            if tid in seen:
                continue
            seen.add(tid)
            dismiss_file(db_path, tid, note="autonomous_reject")
            other = review_for_file(db_path, tid) or row
            suggested = str((other or {}).get("suggested") or "")
            if suggested and other and other.get("lane") != "new_concept":
                cid = upsert(db_path, suggested, source="user")
                if cid:
                    add_example(
                        db_path,
                        cid,
                        file_id=tid,
                        role="negative",
                        source="user",
                    )
            n += 1
    _set_review_status(db_path, list(seen), "rejected")
    return n


def _enqueue_anchor_expansion(
    db,
    db_path: str,
    anchor_ids: list[int],
    label: str,
) -> int:
    """Anchor CLIP komşularını yeni concept için pending review adayı yap. Verified teach etmez."""
    name = " ".join(str(label or "").strip().split())
    ids = [int(x) for x in anchor_ids if int(x) > 0]
    if not db_path or not name or not ids:
        return 0
    blobs = _clip_blobs_for_files(db, ids)
    anchor_vecs: list[np.ndarray] = []
    for fid in ids:
        raw = blobs.get(fid)
        if not raw:
            continue
        try:
            v = np.frombuffer(raw, dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if v.size == 0 or not np.isfinite(v).all():
            continue
        anchor_vecs.append(v)
    if not anchor_vecs:
        return 0
    skip = taught_file_ids(db_path) | dismissed_file_ids(db_path) | set(ids)
    candidates: list[tuple[int, float]] = []
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT file_id, clip_embedding FROM features "
                "WHERE clip_embedding IS NOT NULL AND length(clip_embedding)>=16 "
                "LIMIT 2500"
            ).fetchall()
    except Exception:
        rows = []
    for row in rows:
        fid = int(row["file_id"] if hasattr(row, "keys") else row[0] or 0)
        if fid <= 0 or fid in skip:
            continue
        raw = row["clip_embedding"] if hasattr(row, "keys") else row[1]
        if not raw:
            continue
        try:
            ov = np.frombuffer(bytes(raw), dtype=np.float32)
        except (ValueError, TypeError):
            continue
        best = 0.0
        for av in anchor_vecs:
            if ov.size != av.size:
                continue
            best = max(best, float(_cosine(av, ov)))
        if best >= LOW_MIN:
            candidates.append((fid, best))
    if not candidates:
        return 0
    candidates.sort(key=lambda x: -x[1])
    top = candidates[: min(80, max(8, len(ids) * 24))]
    c = _conn(db_path)
    cur = c.execute(
        """INSERT INTO autonomous_clusters(suggested_name,file_count,status,reason,created_at)
           VALUES(?,?,?,?,?)""",
        (
            name,
            len(top),
            "pending",
            f"Correction anchor «{name}» · {len(top)} benzer aday · doğrulama bekliyor",
            _now(),
        ),
    )
    cluster_id = int(cur.lastrowid)
    c.commit()
    c.close()
    reason = f"«{name}» anchor benzeri · kullanıcı onayı olmadan verified yazılmaz"
    anchor_id = int(ids[0])
    for fid, sc in top:
        path = ""
        blob = None
        try:
            with db.connect() as conn:
                row = conn.execute(
                    "SELECT f.path, fe.clip_embedding FROM files f "
                    "LEFT JOIN features fe ON fe.file_id=f.id WHERE f.id=?",
                    (int(fid),),
                ).fetchone()
            if row:
                path = str(row["path"] or "")
                raw = row["clip_embedding"]
                if raw:
                    blob = bytes(raw)
        except Exception:
            path = ""
            blob = None
        _apply_candidate_memory(
            db_path,
            int(fid),
            name,
            score=float(sc),
            blob=blob,
            path=path,
            reason=reason,
            anchor_id=anchor_id,
            lane="new_concept",
        )
        upsert_review(
            db_path,
            fid,
            lane="new_concept",
            suggested=name,
            confidence=float(sc),
            rivals=[],
            reason=reason,
            cluster_id=cluster_id,
            status="pending",
        )
    return len(top)


def correct_reviews(
    db, db_path: str, file_ids: list[int], label: str, overlay: dict | None = None
) -> dict[str, int]:
    """Cluster correction: yalnız seçilen yanlış üyeler.

    Tüm eski cluster'ı yeni label'a çevirmez. Seçilenler demote + yeni concept user anchor;
    sonra anchor merkezli kontrollü aday expansion (pending, verified değil).
    confirm_reviews toplu onay davranışını korur.
    """
    name = " ".join(str(label or "").strip().split())
    ids = [int(x) for x in file_ids if int(x) > 0]
    if not name or not ids:
        return {"taught": 0, "concept_id": 0, "demoted": 0, "expanded": 0}
    rivals: list[str] = []
    for fid in ids:
        row = review_for_file(db_path, fid)
        if not row:
            continue
        sug = str(row.get("suggested") or "").strip()
        if sug and sug.casefold() != name.casefold():
            rivals.append(sug)
        try:
            for pair in json.loads(row.get("rivals_json") or "[]"):
                if isinstance(pair, (list, tuple)) and pair:
                    rname = str(pair[0] or "").strip()
                    if rname and rname.casefold() != name.casefold():
                        rivals.append(rname)
        except Exception:
            pass
    rivals = list(dict.fromkeys(rivals))
    stats = correct_concept_membership(
        db, db_path, ids, name, overlay=overlay, rival_labels=rivals
    )
    expanded = _enqueue_anchor_expansion(db, db_path, ids, name)
    _set_review_status(db_path, ids, "corrected")
    out = dict(stats)
    out["expanded"] = int(expanded)
    return out
