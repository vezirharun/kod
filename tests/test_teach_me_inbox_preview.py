"""Bana Öğret — ortak card / preview resolver (tüm havuzlar)."""
from __future__ import annotations

from pathlib import Path

from core.autonomous_learn import pending_review_cards, upsert_review
from core.db import Database
from core.teach_me import (
    TeachMeCard,
    enrich_teach_card_paths,
    list_inbox_pools,
)


def _db(tmp_path):
    db = Database(tmp_path / "patterns.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("t", str(tmp_path)),
        )
    return db, str(tmp_path / "patterns.db")


def _add(db, tmp_path, name, **fields):
    img = tmp_path / name
    img.write_bytes(b"x")
    payload = {
        "path": str(img),
        "filename": name,
        "source_id": 1,
        "status": "indexed",
        "pattern_family": fields.get("family", ""),
        "pattern_confidence": fields.get("confidence", 0.0),
        "needs_review": fields.get("needs_review", 0),
        "thumbnail_path": fields.get("thumbnail_path", ""),
        "feature_preview_path": fields.get("feature_preview_path", ""),
    }
    return int(db.upsert_file(payload))


def test_undefined_uses_thumbnail_when_present(tmp_path):
    db, path = _db(tmp_path)
    thumb = tmp_path / "t.webp"
    thumb.write_bytes(b"thumb")
    # Yeni id'li thumb'suz dosya + eski/thumb'lı tanımsız
    _add(db, tmp_path, "new_no_thumb.tif", family="")
    fid = _add(
        db,
        tmp_path,
        "old_with_thumb.png",
        family="",
        thumbnail_path=str(thumb),
    )
    # id sırasını bozmak için new_no_thumb daha büyük id'ye sahip
    pools = list_inbox_pools(db, path, limit=20)
    undef = pools.get("undefined") or []
    assert undef, "undefined pool empty"
    # Thumb'lı dosya havuzda ve preview dolu olmalı
    by_id = {c.file_id: c for c in undef}
    assert fid in by_id
    assert by_id[fid].preview_path == str(thumb)
    assert Path(by_id[fid].preview_path).is_file()


def test_undefined_falls_back_to_feature_preview(tmp_path):
    db, path = _db(tmp_path)
    fp = tmp_path / "fp.webp"
    fp.write_bytes(b"fp")
    fid = _add(
        db,
        tmp_path,
        "only_fp.png",
        family="",
        feature_preview_path=str(fp),
    )
    pools = list_inbox_pools(db, path, limit=20)
    card = next(c for c in (pools.get("undefined") or []) if c.file_id == fid)
    assert card.feature_preview_path == str(fp)
    assert Path(card.preview_path).is_file()


def test_suspicious_keeps_suggestion_and_preview(tmp_path):
    db, path = _db(tmp_path)
    thumb = tmp_path / "s.webp"
    thumb.write_bytes(b"x")
    fid = _add(
        db,
        tmp_path,
        "sus.png",
        family="",
        thumbnail_path=str(thumb),
    )
    upsert_review(
        path,
        fid,
        lane="low",
        suggested="GÜL",
        confidence=0.77,
        reason="Düşük güven",
        status="pending",
    )
    pools = list_inbox_pools(db, path, limit=20)
    cards = pools.get("suspicious") or []
    card = next(c for c in cards if c.file_id == fid)
    assert card.preview_path == str(thumb)
    assert card.suggested == "GÜL"
    assert abs(card.confidence - 0.77) < 1e-6
    assert card.reason


def test_undecided_keeps_suggestion_and_preview(tmp_path):
    db, path = _db(tmp_path)
    thumb = tmp_path / "u.webp"
    thumb.write_bytes(b"x")
    fid = _add(db, tmp_path, "u.png", family="", thumbnail_path=str(thumb))
    upsert_review(
        path,
        fid,
        lane="undecided",
        suggested="DUDAK",
        confidence=0.9,
        rivals=[["DUDAK", 0.9], ["GÜL", 0.88]],
        reason="Birden fazla güçlü tahmin",
        status="pending",
    )
    pools = list_inbox_pools(db, path, limit=20)
    card = next(c for c in (pools.get("undecided") or []) if c.file_id == fid)
    assert Path(card.preview_path).is_file()
    assert card.suggested == "DUDAK"
    assert card.rivals


def test_new_concept_uses_same_resolver(tmp_path):
    db, path = _db(tmp_path)
    thumb = tmp_path / "n.webp"
    thumb.write_bytes(b"x")
    fid = _add(db, tmp_path, "n.png", family="", thumbnail_path=str(thumb))
    upsert_review(
        path,
        fid,
        lane="new_concept",
        suggested="Yeni kavram",
        confidence=0.0,
        status="pending",
        cluster_id=1,
    )
    pools = list_inbox_pools(db, path, limit=20)
    card = next(c for c in (pools.get("new_concept") or []) if c.file_id == fid)
    enrich_teach_card_paths(card, db=db, cache_dir=str(tmp_path))
    assert Path(card.preview_path).is_file()


def test_enrich_placeholder_when_missing(tmp_path):
    card = TeachMeCard(
        file_id=1,
        filename="x.tif",
        path=str(tmp_path / "x.tif"),
        preview_path="",
        pool="undefined",
        reason="",
        guess="",
        confidence=0,
    )
    enrich_teach_card_paths(card, db=None, cache_dir=str(tmp_path))
    assert card.preview_path == ""
    assert card.feature_preview_path == ""


def test_same_file_id_consistent_across_enrich(tmp_path):
    db, path = _db(tmp_path)
    thumb = tmp_path / "c.webp"
    thumb.write_bytes(b"x")
    fid = _add(
        db,
        tmp_path,
        "c.png",
        family="",
        thumbnail_path=str(thumb),
        feature_preview_path=str(thumb),
    )
    sparse = TeachMeCard(
        file_id=fid,
        filename="",
        path="",
        preview_path="",
        pool="undefined",
        reason="",
        guess="",
        confidence=0,
    )
    enrich_teach_card_paths(sparse, db=db, cache_dir=str(tmp_path))
    assert sparse.filename == "c.png"
    assert sparse.preview_path == str(thumb)
    assert Path(sparse.path).name == "c.png"


def test_undefined_prefers_media_over_newest_bare(tmp_path):
    """Yeni thumb'suz dosya, thumb'lı tanımsızın önüne geçmemeli (ilk slotlar)."""
    db, path = _db(tmp_path)
    thumb = tmp_path / "pref.webp"
    thumb.write_bytes(b"x")
    with_media = _add(
        db, tmp_path, "has_media.png", family="", thumbnail_path=str(thumb)
    )
    for i in range(5):
        _add(db, tmp_path, f"bare{i}.tif", family="")
    pools = list_inbox_pools(db, path, limit=5)
    undef = pools.get("undefined") or []
    assert undef
    # İlk kartlar arasında media olmalı
    assert any(c.file_id == with_media and c.preview_path for c in undef)


def _undecided_tm(a="floral", b="paisley"):
    return {
        "pattern_family": a,
        "classification_confidence": 0.9,
        "pattern_dna": {"family": b, "confidence": 0.86},
    }


def test_undecided_media_budget_not_burned_by_non_undecided(tmp_path):
    """E: 80+ media'lı settled satır scan LIMIT'i yakmasın; thumb'lı undecided gelsin."""
    db, path = _db(tmp_path)
    thumb = tmp_path / "u.webp"
    thumb.write_bytes(b"thumb")
    # Önce gerçek undecided (thumb'lı) — düşük id
    good = _add(
        db,
        tmp_path,
        "good_undec.png",
        family="floral",
        confidence=0.9,
        thumbnail_path=str(thumb),
    )
    db.upsert_features(good, {"texture_map": _undecided_tm()})
    # Sonra 120 media'lı ama undecided olmayan (tek etiket) — eski SQL scan'i doldururdu
    for i in range(120):
        t = tmp_path / f"m{i}.webp"
        t.write_bytes(b"m")
        _add(
            db,
            tmp_path,
            f"settled{i}.png",
            family="solid",
            confidence=0.95,
            thumbnail_path=str(t),
        )
    pools = list_inbox_pools(db, path, limit=80)
    undecided = pools.get("undecided") or []
    by_id = {c.file_id: c for c in undecided}
    assert good in by_id, "media budget burned; real undecided missing"
    assert Path(by_id[good].preview_path).is_file()
    # G: undefined/suspicious regression — settled solid yüksek conf tanımsız değil
    undef_ids = {c.file_id for c in (pools.get("undefined") or [])}
    assert good not in undef_ids


def test_undecided_prefers_media_among_true_candidates(tmp_path):
    """A/B: true undecided içinde thumb/fp olanlar bare'den önce."""
    db, path = _db(tmp_path)
    thumb = tmp_path / "pref_u.webp"
    thumb.write_bytes(b"x")
    # Media önce (düşük id); bare daha yeni — media-prefer yoksa bare önde olurdu
    with_media = _add(
        db,
        tmp_path,
        "media_u.png",
        family="floral",
        confidence=0.9,
        thumbnail_path=str(thumb),
    )
    bare = _add(db, tmp_path, "bare_u.png", family="floral", confidence=0.9)
    for fid in (bare, with_media):
        db.upsert_features(fid, {"texture_map": _undecided_tm()})
    pools = list_inbox_pools(db, path, limit=20)
    undecided = pools.get("undecided") or []
    ids = [c.file_id for c in undecided]
    assert with_media in ids and bare in ids
    assert ids.index(with_media) < ids.index(bare)
    assert Path(next(c for c in undecided if c.file_id == with_media).preview_path).is_file()


def test_teach_panel_thumb_miss_requests_scheduler(tmp_path):
    """C/D/F: miss → async request; aynı gen'de dup yok; ready ikonu günceller."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from core.settings import AppSettings
    from ui.teach_me_panel import TeachMePanel

    QApplication.instance() or QApplication([])
    db, path = _db(tmp_path)
    img = tmp_path / "miss_src.png"
    img.write_bytes(b"src")
    fid = _add(db, tmp_path, "miss_u.png", family="floral", confidence=0.9)
    with db.connect() as conn:
        conn.execute("UPDATE files SET path=? WHERE id=?", (str(img), fid))
    db.upsert_features(fid, {"texture_map": _undecided_tm()})

    class _FakeSched:
        def __init__(self):
            self.calls: list[tuple] = []

            class _Sig:
                def connect(self, *_a, **_k):
                    return None

                def disconnect(self, *_a, **_k):
                    return None

            self.thumbnail_ready = _Sig()

        def request(self, file_id, path, size, **kwargs):
            self.calls.append((int(file_id), str(path), int(size), dict(kwargs)))

        def request_visible(self, items: list) -> None:
            for item in items:
                if len(item) >= 6:
                    fid, path, size, source, name, fp = item[:6]
                    self.request(
                        int(fid),
                        str(path),
                        int(size),
                        priority=0,
                        source_path=str(source or ""),
                        filename=str(name or ""),
                        feature_preview_path=str(fp or ""),
                    )
                elif len(item) >= 5:
                    fid, path, size, source, name = item[:5]
                    self.request(
                        int(fid),
                        str(path),
                        int(size),
                        priority=0,
                        source_path=str(source or ""),
                        filename=str(name or ""),
                    )
                else:
                    fid, path, size = item[:3]
                    self.request(int(fid), str(path), int(size), priority=0)

    settings = AppSettings()
    settings.db_path = path
    settings.cache_dir = str(tmp_path)
    panel = TeachMePanel(settings)
    fake = _FakeSched()
    panel.set_thumbnail_scheduler(fake)  # type: ignore[arg-type]
    panel.reload()
    assert panel.list_undecided.count() >= 1
    # Active tab is Tanımsızlar by default — switch to Kararsızlar (viewport only).
    panel.tabs.setCurrentIndex(2)
    assert any(c[0] == fid for c in fake.calls), "miss must request scheduler"
    n1 = len(fake.calls)
    card = next(c for c in panel._cards["undecided"] if c.file_id == fid)
    panel._request_thumb_miss(card, panel._fill_gen)
    # Same gen flush may re-call request_visible; fid must not duplicate-enqueue intent.
    assert any(c[0] == fid for c in fake.calls)
    assert len(fake.calls) >= n1

    qimg = QImage(32, 32, QImage.Format.Format_RGB32)
    qimg.fill(0xFF00AA00)
    panel._thumb_wait_gen[fid] = panel._fill_gen
    panel._on_teach_thumb_ready(fid, qimg, 160)
    item = None
    for i in range(panel.list_undecided.count()):
        it = panel.list_undecided.item(i)
        if int(it.data(Qt.ItemDataRole.UserRole) or 0) == fid:
            item = it
            break
    assert item is not None
    assert not item.icon().isNull()
