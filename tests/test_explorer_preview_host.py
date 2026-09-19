"""Explorer preview host unit tests — no COM/Explorer required."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from explorer_preview.host.blank import is_valid_preview_image
from explorer_preview.host.rasterize import (
    EXPLORER_PRIORITY_EXTS,
    NATIVE_WINDOWS_EXTS,
    explorer_cache_key,
    rasterize_to_file,
)


def test_priority_exts_exclude_native_photo_formats():
    assert ".eps" in EXPLORER_PRIORITY_EXTS
    assert ".jpg" in NATIVE_WINDOWS_EXTS
    assert ".jpg" not in EXPLORER_PRIORITY_EXTS
    assert ".png" not in EXPLORER_PRIORITY_EXTS


def test_blank_white_rejected(tmp_path: Path):
    p = tmp_path / "white.png"
    Image.new("RGB", (128, 128), (255, 255, 255)).save(p)
    ok, reason = is_valid_preview_image(p)
    assert not ok
    assert "white" in reason or "blank" in reason


def test_real_image_accepted(tmp_path: Path):
    p = tmp_path / "ok.png"
    img = Image.new("RGB", (64, 64), (20, 40, 200))
    for x in range(64):
        img.putpixel((x, x % 64), (255, 0, 0))
    img.save(p)
    ok, reason = is_valid_preview_image(p)
    assert ok, reason


def test_rasterize_png(tmp_path: Path):
    src = tmp_path / "src.png"
    out = tmp_path / "out.png"
    Image.new("RGB", (80, 60), (10, 120, 40)).save(src)
    ok, msg = rasterize_to_file(str(src), str(out), max_edge=64)
    assert ok, msg
    assert out.is_file()
    vok, _ = is_valid_preview_image(out)
    assert vok


@pytest.mark.parametrize("ext", [".eps", ".ai", ".psd", ".pdf", ".svg", ".tif"])
def test_priority_list_complete(ext):
    assert ext in EXPLORER_PRIORITY_EXTS or ext == ".tiff"


def test_host_does_not_import_search_stack():
    import ast
    from pathlib import Path

    path = Path("explorer_preview/host/rasterize.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = " ".join(imported).lower()
    for banned in ("faiss", "search_engine", "jobstore", "open_clip", "dino"):
        assert banned not in joined


def test_cache_key_distinguishes_same_stem_different_ext(tmp_path: Path):
    """Same basename EPS vs AI vs PDF must never share a cache identity."""
    eps = tmp_path / "motif.eps"
    ai = tmp_path / "motif.ai"
    pdf = tmp_path / "motif.pdf"
    for p in (eps, ai, pdf):
        p.write_bytes(b"%PDF-1.4\n")
    keys = {explorer_cache_key(eps), explorer_cache_key(ai), explorer_cache_key(pdf)}
    assert len(keys) == 3


def test_render_cache_paths_no_stem_collision(tmp_path: Path):
    from core.utils import file_id_from_path

    cache = tmp_path / "cache" / "render_previews"
    cache.mkdir(parents=True)
    paths = []
    for ext in (".eps", ".ai", ".pdf"):
        src = tmp_path / f"design{ext}"
        src.write_bytes(b"x")
        paths.append(cache / f"{file_id_from_path(str(src))}_render.png")
    assert len(set(paths)) == 3
    # Must not all collapse to design_render.png
    assert not all(p.name == "design_render.png" for p in paths)


def test_rasterize_missing_file_safe(tmp_path: Path):
    ok, msg = rasterize_to_file(
        str(tmp_path / "nope.eps"), str(tmp_path / "out.png"), max_edge=64
    )
    assert not ok
    assert msg == "source_missing"


def test_blank_cache_rejected_then_regen(tmp_path: Path, monkeypatch):
    """Blank host cache is deleted; next path regenerates from source."""
    from explorer_preview.host import rasterize as rz

    src = tmp_path / "a.png"
    out = tmp_path / "out.png"
    Image.new("RGB", (48, 48), (30, 90, 150)).save(src)
    monkeypatch.setattr(rz, "_cache_dir", lambda: tmp_path / "cache")
    host_cache = tmp_path / "cache" / "explorer_preview"
    host_cache.mkdir(parents=True)
    blank = host_cache / f"{explorer_cache_key(src)}.png"
    Image.new("RGB", (48, 48), (255, 255, 255)).save(blank)

    ok, msg = rasterize_to_file(str(src), str(out), max_edge=64)
    assert ok, msg
    assert out.is_file()
    vok, _ = is_valid_preview_image(out)
    assert vok
    # blank identity cache should have been removed or overwritten with valid
    if blank.is_file():
        bok, _ = is_valid_preview_image(blank)
        assert bok


def test_register_script_excludes_jpg_png():
    ps1 = Path("explorer_preview/install/register.ps1").read_text(encoding="utf-8")
    # Primary Exts assignment line
    assert '$Exts = @(".eps", ".ai", ".psd", ".svg", ".tif", ".tiff", ".pdf")' in ps1
    assert "Do NOT register JPG/PNG" in ps1
    # Cleanup block references photo formats but must not register them in $Exts
    exts_line = [ln for ln in ps1.splitlines() if ln.strip().startswith("$Exts =")][0]
    for photo in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"):
        assert photo not in exts_line

