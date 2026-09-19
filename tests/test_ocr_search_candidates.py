"""OCR text → text-search candidate pool. Ranking/FAISS/UVI değişmez."""

from __future__ import annotations

from pathlib import Path

from core.db import Database


def _db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "ocr_cand.db")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("marka", str(tmp_path)),
        )
    return db


def _file(
    db: Database,
    *,
    name: str,
    ocr: str = "",
    status: str = "pending",
    preview: int = 1,
    processed: int = 1,
    source_id: int = 1,
) -> int:
    fid = db.upsert_file(
        {
            "path": str(Path("F:/marka") / name),
            "filename": name,
            "source_id": source_id,
            "status": status,
            "ocr_text": ocr,
        }
    )
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE files
            SET physical_preview_ready=?, ocr_processed=?, ocr_text=?
            WHERE id=?
            """,
            (int(preview), int(processed), ocr, int(fid)),
        )
    return int(fid)


def test_ocr_only_pending_files_enter_gucci_candidate_pool(tmp_path: Path):
    db = _db(tmp_path)
    # Filename dolu olsun ki eski kod OCR adımını atlayabilsin
    for i in range(25):
        _file(db, name=f"gucci-logo-{i:02d}.jpg", ocr="", status="indexed", processed=0)
    a = _file(
        db,
        name="5807664_175959259129_2.jpg",
        ocr="GUCCIFICATION? 3\nMO ONMO",
        status="pending",
    )
    b = _file(
        db,
        name="5807664_180017260121_2.jpg",
        ocr="GUCCI\n\nSEXINESS",
        status="pending",
    )
    # OCR hiç bitmemiş — adaya girmesin
    ghost = _file(
        db,
        name="empty-pending.jpg",
        ocr="",
        status="pending",
        processed=0,
    )
    hits = db.search_text_candidates(["gucci"], limit=80)
    ids = {int(r["id"]) for r in hits}
    assert a in ids
    assert b in ids
    assert ghost not in ids
    names = {r["filename"] for r in hits if int(r["id"]) in {a, b}}
    assert "5807664_175959259129_2.jpg" in names
    assert "5807664_180017260121_2.jpg" in names


def test_search_ocr_text_does_not_use_filename(tmp_path: Path):
    db = _db(tmp_path)
    named = _file(db, name="gucci-file.jpg", ocr="", status="indexed", processed=0)
    ocr_only = _file(
        db, name="no-brand.jpg", ocr="GUCCI SEXINESS", status="pending"
    )
    rows = db.search_ocr_text("gucci", limit=20)
    ids = {int(r["id"]) for r in rows}
    assert ocr_only in ids
    assert named not in ids
