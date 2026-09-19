"""Global Preview: renderability, not extension-as-other-type."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.db import Database
from core.global_preview import (
    STATUS_RENDERABLE,
    STATUS_UNSUPPORTED,
    materialize_visual,
    renderer_availability,
    sniff_family,
)
from core.index_v3 import IndexEngineV3, Mode
from core.index_v3.queues import JobStore
from core.index_v3.types import Artifact, Job, QueueKind
from core.index_v3.ui_bridge import count_v3_ssot
from core.light_asset_extractor import extract_light_assets_once
from core.preview_cache import FeaturePreviewCache
from core.settings import AppSettings
from core.thumbnailer import Thumbnailer


def test_sniff_not_extension_only(tmp_path: Path):
    pdf = tmp_path / "x.bin"
    pdf.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    fam, _ = sniff_family(str(pdf))
    assert fam == "pdf"


def test_empty_file_is_empty_not_other_type(tmp_path: Path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"")
    vis = materialize_visual(str(p), cache_dir=str(tmp_path / "c"))
    assert vis.status == "empty_file"
    assert not vis.ok


def test_raster_jpg_materialize(tmp_path: Path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), (10, 80, 200)).save(p, "JPEG")
    vis = materialize_visual(str(p), cache_dir=str(tmp_path / "c"))
    assert vis.ok
    assert vis.renderer == "native_raster"
    assert vis.status == STATUS_RENDERABLE


def test_pdf_renders_real_preview(tmp_path: Path):
    pytest.importorskip("fitz")
    import fitz

    pdf = tmp_path / "p.pdf"
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((40, 80), "VEZIR")
    doc.save(str(pdf))
    doc.close()
    vis = materialize_visual(str(pdf), cache_dir=str(tmp_path / "c"))
    assert vis.ok, vis.error
    assert Path(vis.raster_path).is_file()
    with Image.open(vis.raster_path) as img:
        assert img.size[0] >= 16 and img.size[1] >= 16


def test_cdr_without_inkscape_is_unsupported_visual(tmp_path: Path):
    p = tmp_path / "x.cdr"
    p.write_bytes(b"RIFF....CDR")
    vis = materialize_visual(str(p), cache_dir=str(tmp_path / "c"))
    avail = renderer_availability()
    if avail.get("cdr"):
        pytest.skip("CDR renderer installed")
    assert vis.status == STATUS_UNSUPPORTED
    assert "renderer_missing" in (vis.error or "")


def test_embroidery_dst_real_stitch_png(tmp_path: Path):
    pytest.importorskip("pyembroidery")
    import pyembroidery

    dst = tmp_path / "s.dst"
    pat = pyembroidery.EmbPattern()
    pat.add_stitch_absolute(pyembroidery.STITCH, 0, 0)
    pat.add_stitch_absolute(pyembroidery.STITCH, 100, 0)
    pat.add_stitch_absolute(pyembroidery.STITCH, 100, 80)
    pat.add_stitch_absolute(pyembroidery.STITCH, 0, 80)
    pat.add_stitch_absolute(pyembroidery.END, 0, 80)
    pyembroidery.write_dst(pat, str(dst))
    vis = materialize_visual(str(dst), cache_dir=str(tmp_path / "c"))
    assert vis.ok, vis.error
    assert vis.renderer == "pyembroidery"
    with Image.open(vis.raster_path) as img:
        assert img.size[0] >= 8


def test_light_extract_pdf_produces_thumb_preview(tmp_path: Path):
    pytest.importorskip("fitz")
    import fitz

    cache = tmp_path / "cache"
    cache.mkdir()
    pdf = tmp_path / "q.pdf"
    doc = fitz.open()
    doc.new_page(width=120, height=120)
    doc.save(str(pdf))
    doc.close()
    thumb = Thumbnailer(str(cache), 64, "webp")
    fp = FeaturePreviewCache(str(cache), 128, "webp")
    r = extract_light_assets_once(
        str(pdf), thumbnailer=thumb, feature_preview=fp, source_mtime=0
    )
    assert r.success, r.error
    assert Path(r.thumbnail_path).is_file()
    assert Path(r.feature_preview_path).is_file()


def test_reopen_skips_unsupported_but_retries_pillow(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.db")
    store.enqueue(
        [Job(file_id=1, artifact=Artifact.THUMBNAIL, queue=QueueKind.LIGHT, source_id=9)]
    )
    store.fail(
        1, Artifact.THUMBNAIL, error="cannot identify image file", permanent=True, max_attempts=1
    )
    store.enqueue(
        [Job(file_id=2, artifact=Artifact.PREVIEW, queue=QueueKind.LIGHT, source_id=9)]
    )
    store.fail(
        2,
        Artifact.PREVIEW,
        error="unsupported_visual:renderer_missing:inkscape/libreoffice",
        permanent=True,
        max_attempts=1,
    )
    n = store.reopen_failed_permanent(source_ids=[9])
    assert n == 1
    assert store.count_pending(QueueKind.LIGHT, source_ids=[9]) == 1


def _stitch_pattern():
    pytest.importorskip("pyembroidery")
    import pyembroidery

    pat = pyembroidery.EmbPattern()
    pat.add_stitch_absolute(pyembroidery.STITCH, 0, 0)
    pat.add_stitch_absolute(pyembroidery.STITCH, 80, 0)
    pat.add_stitch_absolute(pyembroidery.STITCH, 80, 50)
    pat.add_stitch_absolute(pyembroidery.STITCH, 0, 50)
    pat.add_stitch_absolute(pyembroidery.END, 0, 50)
    return pyembroidery, pat


def _assert_png(vis):
    assert vis.ok, vis.error
    assert vis.status == STATUS_RENDERABLE
    with Image.open(vis.raster_path) as img:
        assert img.size[0] >= 8 and img.size[1] >= 8


def test_tiff_png_real_preview(tmp_path: Path):
    tif = tmp_path / "a.tif"
    png = tmp_path / "a.png"
    Image.new("RGB", (28, 18), (12, 90, 40)).save(tif, "TIFF")
    Image.new("RGB", (28, 18), (12, 90, 40)).save(png)
    cache = str(tmp_path / "c")
    for p in (tif, png):
        vis = materialize_visual(str(p), cache_dir=cache)
        assert vis.ok
        assert vis.renderer == "native_raster"


def test_svg_real_preview(tmp_path: Path):
    pytest.importorskip("cairosvg")
    svg = tmp_path / "a.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="32">'
        '<rect width="48" height="32" fill="#cc2020"/></svg>',
        encoding="utf-8",
    )
    vis = materialize_visual(str(svg), cache_dir=str(tmp_path / "c"))
    _assert_png(vis)
    assert vis.renderer == "cairosvg"


def test_eps_real_preview(tmp_path: Path):
    eps = tmp_path / "a.eps"
    eps.write_text(
        "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 80 80\n"
        "newpath 10 10 moveto 70 10 lineto 70 70 lineto 10 70 lineto closepath "
        "0.8 0.1 0.1 setrgbcolor fill showpage\n",
        encoding="ascii",
    )
    vis = materialize_visual(str(eps), cache_dir=str(tmp_path / "c"))
    if vis.status in (STATUS_UNSUPPORTED, "renderer_missing"):
        pytest.skip("Ghostscript/PyMuPDF EPS renderer unavailable")
    _assert_png(vis)


def test_ai_pdf_compatible_real_preview(tmp_path: Path):
    pytest.importorskip("fitz")
    import fitz

    ai = tmp_path / "a.ai"
    doc = fitz.open()
    page = doc.new_page(width=90, height=90)
    page.draw_rect(page.rect, color=(1, 0, 0), fill=(0.7, 0.1, 0.1))
    doc.save(str(ai))
    doc.close()
    vis = materialize_visual(str(ai), cache_dir=str(tmp_path / "c"))
    _assert_png(vis)


def test_psd_real_preview(tmp_path: Path):
    pytest.importorskip("psd_tools")
    from psd_tools import PSDImage

    psd = tmp_path / "a.psd"
    PSDImage.frompil(Image.new("RGB", (24, 24), (180, 20, 20))).save(str(psd))
    vis = materialize_visual(str(psd), cache_dir=str(tmp_path / "c"))
    _assert_png(vis)
    assert vis.renderer in ("psd-tools", "preview_renderer")


def test_dxf_real_preview(tmp_path: Path):
    pytest.importorskip("ezdxf")
    import ezdxf

    dxf = tmp_path / "a.dxf"
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 40))
    msp.add_circle((50, 20), 15)
    doc.saveas(dxf)
    vis = materialize_visual(str(dxf), cache_dir=str(tmp_path / "c"))
    _assert_png(vis)
    assert vis.renderer == "ezdxf"


def test_hpgl_plt_real_preview(tmp_path: Path):
    plt = tmp_path / "a.plt"
    plt.write_text("IN;SP1;PU0,0;PD120,0,120,70,0,70,0,0;SP0;\n", encoding="ascii")
    vis = materialize_visual(str(plt), cache_dir=str(tmp_path / "c"))
    _assert_png(vis)
    assert vis.renderer == "hpgl"


def test_dwg_without_oda_is_unsupported(tmp_path: Path):
    dwg = tmp_path / "a.dwg"
    dwg.write_bytes(b"AC1021" + b"\x00" * 80)
    vis = materialize_visual(str(dwg), cache_dir=str(tmp_path / "c"))
    assert vis.status == STATUS_UNSUPPORTED
    assert "renderer_missing" in (vis.error or "")
    assert not vis.ok


def test_emb_ofm_are_unsupported_not_placeholder(tmp_path: Path):
    for name in ("a.emb", "a.ofm"):
        p = tmp_path / name
        p.write_bytes(b"WILCOM-OFM" + b"\x00" * 64)
        vis = materialize_visual(str(p), cache_dir=str(tmp_path / "c"))
        assert vis.status == STATUS_UNSUPPORTED
        assert "no_stitch_reader" in (vis.error or "")
        assert not vis.ok
        assert vis.raster_path == ""


@pytest.mark.parametrize(
    "ext,writer",
    [
        ("pes", "write_pes"),
        ("jef", "write_jef"),
        ("exp", "write_exp"),
        ("vp3", "write_vp3"),
    ],
)
def test_embroidery_writer_formats_real_png(tmp_path: Path, ext: str, writer: str):
    pyembroidery, pat = _stitch_pattern()
    path = tmp_path / f"s.{ext}"
    getattr(pyembroidery, writer)(pat, str(path))
    vis = materialize_visual(str(path), cache_dir=str(tmp_path / "c"))
    _assert_png(vis)
    assert vis.renderer == "pyembroidery"


def test_pyembroidery_auto_discovers_readers():
    pytest.importorskip("pyembroidery")
    from core.global_preview import pyembroidery_stitch_extensions

    exts = pyembroidery_stitch_extensions()
    for need in (".dst", ".pes", ".jef", ".exp", ".vp3", ".hus", ".xxx"):
        assert need in exts
    assert ".emb" not in exts
    assert ".ofm" not in exts
    assert ".csv" not in exts


def test_fast_dxf_and_plt_then_hash(tmp_path: Path):
    pytest.importorskip("ezdxf")
    import ezdxf

    db = Database(tmp_path / "t.db")
    root = tmp_path / "src"
    root.mkdir()
    doc = ezdxf.new()
    doc.modelspace().add_circle((0, 0), 20)
    doc.saveas(root / "p.dxf")
    (root / "p.plt").write_text("IN;PU0,0;PD40,0,40,30,0,30,0,0;\n", encoding="ascii")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("g", str(root)),
        )
    settings = AppSettings()
    settings.cache_dir = str(tmp_path / "cache")
    settings.db_path = str(tmp_path / "t.db")
    settings.ensure_dirs()
    eng = IndexEngineV3(
        db,
        job_db_path=tmp_path / "jobs.db",
        settings=settings,
        use_real_extractors=True,
    )
    eng.run(mode=Mode.FAST, sources=[{"id": 1, "root_path": str(root)}], walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["total"] == 2
    assert c["preview"] == 2
    assert c["thumbnail"] == 2
    assert c["hash"] == 0


def test_cross_format_preview_similarity_jpg_pdf(tmp_path: Path):
    """Same drawing in JPG and PDF → visual preview, not extension class."""
    pytest.importorskip("fitz")
    import fitz
    import imagehash

    db = Database(tmp_path / "t.db")
    root = tmp_path / "src"
    root.mkdir()
    Image.new("RGB", (120, 120), (200, 30, 30)).save(root / "same.jpg", "JPEG")
    doc = fitz.open()
    page = doc.new_page(width=120, height=120)
    page.draw_rect(page.rect, color=(0.78, 0.12, 0.12), fill=(0.78, 0.12, 0.12))
    doc.save(str(root / "same.pdf"))
    doc.close()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("g", str(root)),
        )
    settings = AppSettings()
    settings.cache_dir = str(tmp_path / "cache")
    settings.db_path = str(tmp_path / "t.db")
    settings.ensure_dirs()
    eng = IndexEngineV3(
        db,
        job_db_path=tmp_path / "jobs.db",
        settings=settings,
        use_real_extractors=True,
    )
    eng.run(mode=Mode.FAST, sources=[{"id": 1, "root_path": str(root)}], walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["preview"] == 2
    rows = []
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT path, feature_preview_path FROM files"
        ).fetchall()
    hashes = []
    for row in rows:
        prev = str(row["feature_preview_path"] or "")
        assert Path(prev).is_file()
        with Image.open(prev) as img:
            hashes.append(imagehash.phash(img.convert("RGB")))
    assert len(hashes) == 2
    dist = hashes[0] - hashes[1]
    assert dist <= 12, f"preview phash distance {dist} — expected visual match"
    store = JobStore(tmp_path / "jobs.db")
    job = Job(file_id=7, artifact=Artifact.THUMBNAIL, queue=QueueKind.LIGHT, source_id=9)
    store.enqueue([job])
    store.fail(7, Artifact.THUMBNAIL, error="cannot identify", permanent=True, max_attempts=1)
    n = store.reopen_failed_permanent(source_ids=[9])
    assert n == 1
    assert store.count_pending(QueueKind.LIGHT, source_ids=[9]) == 1


def test_fast_pdf_then_hash(tmp_path: Path):
    pytest.importorskip("fitz")
    import fitz

    db = Database(tmp_path / "t.db")
    root = tmp_path / "src"
    root.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=80, height=80)
    page.draw_rect(page.rect, color=(1, 0, 0), fill=(0.2, 0.4, 0.8))
    doc.save(str(root / "d.pdf"))
    doc.close()
    Image.new("RGB", (40, 40), (9, 9, 9)).save(root / "a.jpg", "JPEG")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",
            ("g", str(root)),
        )
    settings = AppSettings()
    settings.cache_dir = str(tmp_path / "cache")
    settings.db_path = str(tmp_path / "t.db")
    settings.ensure_dirs()
    eng = IndexEngineV3(
        db,
        job_db_path=tmp_path / "jobs.db",
        settings=settings,
        use_real_extractors=True,
    )
    eng.run(mode=Mode.FAST, sources=[{"id": 1, "root_path": str(root)}], walk_disk=True)
    c = count_v3_ssot(db, [1])
    assert c["total"] == 2
    assert c["preview"] == 2
    assert c["thumbnail"] == 2
    assert c["hash"] == 0
