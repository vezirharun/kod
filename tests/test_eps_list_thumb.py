"""EPS/AI list-thumbnail white-stub fix — unittest (no PySide/Ghostscript)."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


# ---------------------------------------------------------------------------
# Lightweight stubs so core.preview_renderer / blank can be patched without
# pulling product deps (logger, utils, PySide, Ghostscript).
# ---------------------------------------------------------------------------
def _ensure_stub(name: str, **attrs):
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install_minimal_core_stubs():
    # core.logger
    log_mod = _ensure_stub("core.logger")
    if not hasattr(log_mod, "setup_logger"):
        class _L:
            def info(self, *a, **k):
                pass

            def debug(self, *a, **k):
                pass

            def warning(self, *a, **k):
                pass

            def error(self, *a, **k):
                pass

        log_mod.setup_logger = lambda name=None: _L()  # type: ignore

    # core.utils — only what thumbnailer/preview_renderer import at top level
    util = _ensure_stub("core.utils")
    if not hasattr(util, "file_id_from_path"):
        util.file_id_from_path = lambda p: "deadbeef"  # type: ignore
        util.fs_access_path = lambda p: str(p)  # type: ignore
        util.iter_fs_path_candidates = lambda p: [str(p)]  # type: ignore
        util.normalize_path = lambda p: str(p).replace("\\", "/")  # type: ignore

    # explorer_preview.host.blank — optional; tests patch gate directly
    _ensure_stub("explorer_preview")
    _ensure_stub("explorer_preview.host")
    blank = _ensure_stub("explorer_preview.host.blank")
    if not hasattr(blank, "is_valid_preview_image"):
        blank.is_valid_preview_image = lambda path: (True, "ok")  # type: ignore


_install_minimal_core_stubs()

# Import under test after stubs (thumb_resolve itself needs no stubs).
from core import thumb_resolve as tr  # noqa: E402


def _write(path: Path, nbytes: int, fill: bytes = b"\xff") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fill * nbytes if len(fill) == 1 else (fill * ((nbytes // len(fill)) + 1))[:nbytes])
    return path


class TestEpsListThumbResolve(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.cache = self.root / "cache"
        (self.cache / "thumbnails").mkdir(parents=True)
        (self.cache / "feature_previews").mkdir(parents=True)
        self.hash = "abc123"
        self.thumb = self.cache / "thumbnails" / f"{self.hash}.webp"
        self.fp = self.cache / "feature_previews" / f"{self.hash}_fp.webp"
        self.eps_src = str(self.root / "logo.eps")
        self.jpg_src = str(self.root / "photo.jpg")
        self.png_src = str(self.root / "icon.png")

    def tearDown(self):
        self._td.cleanup()

    def test_prefer_valid_fp_over_white_150byte_thumb(self):
        """List must serve FP (~80KB) instead of tiny white thumb (~150B) for .eps."""
        _write(self.thumb, 150)
        _write(self.fp, 80_000)
        path, status = tr.resolve_thumb_path(
            str(self.thumb),
            cache_dir=str(self.cache),
            feature_preview_path=str(self.fp),
            source_path=self.eps_src,
        )
        self.assertEqual(status, "hit")
        self.assertEqual(Path(path).resolve(), self.fp.resolve())
        self.assertTrue(path.endswith("_fp.webp") or "feature_previews" in path)

    def test_invalid_tiny_thumb_not_returned_when_fp_exists(self):
        _write(self.thumb, 100)
        _write(self.fp, 60_000)
        path, status = tr.resolve_thumb_path(
            str(self.thumb),
            cache_dir=str(self.cache),
            feature_preview_path=str(self.fp),
            source_path=self.eps_src,
        )
        self.assertEqual(status, "hit")
        self.assertEqual(Path(path).resolve(), self.fp.resolve())
        # Tiny thumb must not win
        self.assertNotEqual(Path(path).resolve(), self.thumb.resolve())

    def test_jpg_still_prefers_existing_thumb(self):
        """Regression: non-EPS keeps thumb-first order."""
        _write(self.thumb, 150)  # tiny is OK for JPG
        _write(self.fp, 80_000)
        path, status = tr.resolve_thumb_path(
            str(self.thumb),
            cache_dir=str(self.cache),
            feature_preview_path=str(self.fp),
            source_path=self.jpg_src,
        )
        self.assertEqual(status, "hit")
        self.assertEqual(Path(path).resolve(), self.thumb.resolve())

    def test_png_non_eps_thumb_hit_unchanged(self):
        _write(self.thumb, 200)
        path, status = tr.resolve_thumb_path(
            str(self.thumb),
            cache_dir=str(self.cache),
            source_path=self.png_src,
        )
        self.assertEqual(status, "hit")
        self.assertEqual(Path(path).resolve(), self.thumb.resolve())

    def test_tiff_style_non_eps_thumb_hit(self):
        tiff_src = str(self.root / "scan.tiff")
        _write(self.thumb, 400)
        path, status = tr.resolve_thumb_path(
            str(self.thumb),
            cache_dir=str(self.cache),
            source_path=tiff_src,
        )
        self.assertEqual(status, "hit")
        self.assertEqual(Path(path).resolve(), self.thumb.resolve())

    def test_only_valid_fp_returns_fp(self):
        _write(self.fp, 70_000)
        path, status = tr.resolve_thumb_path(
            "",
            cache_dir=str(self.cache),
            feature_preview_path=str(self.fp),
            source_path=self.eps_src,
        )
        self.assertEqual(status, "hit")
        self.assertEqual(Path(path).resolve(), self.fp.resolve())

    def test_tiny_thumb_alone_is_miss_for_eps(self):
        _write(self.thumb, 150)
        path, status = tr.resolve_thumb_path(
            str(self.thumb),
            cache_dir=str(self.cache),
            source_path=self.eps_src,
        )
        self.assertEqual(status, "miss")
        self.assertEqual(path, "")

    def test_is_eps_ai_thumb_usable_size_gate(self):
        tiny = _write(self.thumb, 150)
        big = _write(self.fp, 2000)
        self.assertFalse(tr.is_eps_ai_thumb_usable(tiny))
        self.assertFalse(tr.is_eps_ai_thumb_usable(self.root / "missing.webp"))
        # Without gate module success path: size >= 512 is enough
        self.assertTrue(tr.is_eps_ai_thumb_usable(big))

    def test_gate_rejects_white_accepts_solid_ok(self):
        """Mock/stub blank gate: white rejected; accepted solid mono still OK."""
        good = _write(self.cache / "thumbnails" / "good.webp", 2048)
        white = _write(self.cache / "thumbnails" / "white.webp", 2048)

        def fake_gate(path):
            p = Path(path)
            if p.name.startswith("white"):
                return False, "blank_white_preview:near_white"
            return True, "ok"

        # Install a stub preview_renderer that only exposes the gate.
        fake_pr = types.ModuleType("core.preview_renderer")
        fake_pr._eps_ai_preview_gate = fake_gate  # type: ignore
        with mock.patch.dict(sys.modules, {"core.preview_renderer": fake_pr}):
            # Force re-import path inside helper (it imports each call)
            self.assertTrue(tr.is_eps_ai_thumb_usable(good))
            self.assertFalse(tr.is_eps_ai_thumb_usable(white))

            # resolve: white thumb + valid FP → FP
            _write(self.thumb, 2048)  # large but gate-white
            # Rename semantics: use white.webp as thumb via absolute path
            white_thumb = white
            _write(self.fp, 80_000)
            # Patch gate so thumb "white.webp" fails, FP passes
            def gate2(path):
                name = Path(path).name
                if name == "white.webp":
                    return False, "blank_white"
                return True, "ok"

            fake_pr._eps_ai_preview_gate = gate2  # type: ignore
            path, status = tr.resolve_thumb_path(
                str(white_thumb),
                cache_dir=str(self.cache),
                feature_preview_path=str(self.fp),
                source_path=self.eps_src,
            )
            self.assertEqual(status, "hit")
            self.assertEqual(Path(path).resolve(), self.fp.resolve())

    def test_lookup_valid_preview_stubbed(self):
        """When lookup returns FP, resolve prefers it over tiny thumb."""
        _write(self.thumb, 150)
        _write(self.fp, 90_000)

        with mock.patch.object(tr, "lookup_valid_preview", return_value=str(self.fp)):
            path, status = tr.resolve_thumb_path(
                str(self.thumb),
                cache_dir=str(self.cache),
                source_path=self.eps_src,
            )
        self.assertEqual(status, "hit")
        self.assertEqual(Path(path).resolve(), self.fp.resolve())


class TestEpsHelpers(unittest.TestCase):
    def test_is_eps_ai(self):
        self.assertTrue(tr._is_eps_ai("/x/a.eps"))
        self.assertTrue(tr._is_eps_ai("/x/a.AI"))
        self.assertFalse(tr._is_eps_ai("/x/a.jpg"))
        self.assertFalse(tr._is_eps_ai(""))


if __name__ == "__main__":
    unittest.main()
