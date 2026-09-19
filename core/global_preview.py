"""Global Preview — format-agnostic visual materialization.

Does not invent a second renderer stack. Routes to existing:
  Thumbnailer / pyvips / Pillow  (raster)
  preview_renderer               (PDF/AI/EPS/PSD + Ghostscript/PyMuPDF/psd-tools)
  VectorPreviewPlugin            (PDF/SVG)
  pyembroidery.write_png         (stitch geometry, if installed)
  ezdxf + PyMuPDF                (DXF)
  builtin HPGL parser            (PLT/HPGL)

Status is honest: unsupported_visual ≠ render_failed.
No placeholder images. Proprietary formats without a reader stay unsupported.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.logger import setup_logger
from core.utils import normalize_path

logger = setup_logger(__name__)

STATUS_RENDERABLE = "visual_renderable"
STATUS_UNSUPPORTED = "unsupported_visual"
STATUS_RENDER_FAILED = "render_failed"
STATUS_CORRUPT = "corrupt_file"
STATUS_EMPTY = "empty_file"
STATUS_MISSING_DEP = "renderer_missing"
STATUS_RENDERER_ERROR = "renderer_error"

RASTER_EXTS = frozenset(
    {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
)
VECTOR_EXTS = frozenset({".pdf", ".ai", ".eps", ".svg"})
ADOBE_LAYER_EXTS = frozenset({".psd"})
COREL_EXTS = frozenset({".cdr"})
# Known stitch families (discovery + sniff). Reader presence is resolved at runtime.
EMBROIDERY_EXTS = frozenset(
    {
        ".dst",
        ".dsb",
        ".dsz",
        ".emb",
        ".pes",
        ".pec",
        ".jef",
        ".sew",
        ".exp",
        ".vp3",
        ".hus",
        ".xxx",
        ".pcs",
        ".tap",
        ".ofm",
        ".ksm",
        ".art",
        ".ngs",
        ".wings",
        ".vip",
        ".shv",
        ".csd",
        ".tbf",
        ".pcd",
        ".pcm",
        ".pcq",
        ".pmv",
        ".stc",
        ".zhs",
        ".emd",
        ".u01",
        ".bro",
        ".jpx",
        ".spx",
        ".stx",
        ".phb",
        ".phc",
        ".exy",
        ".fxy",
        ".inb",
        ".mit",
        ".zxy",
    }
)
# pyembroidery has no reader (commented out or never registered).
PROPRIETARY_EMBROIDERY_NO_READER = frozenset(
    {".emb", ".ofm", ".art", ".vip", ".csd", ".ngs", ".wings"}
)
# Do not treat these pyembroidery "readers" as visual stitch sources.
_PYEMB_NON_STITCH = frozenset(
    {".csv", ".json", ".txt", ".gcode", ".png", ".svg", ".edr", ".col", ".inf"}
)
CAD_EXTS = frozenset({".dxf", ".dwg", ".dgn", ".plt", ".hpgl"})
HPGL_EXTS = frozenset({".plt", ".hpgl", ".hpg", ".hpgl2"})
_PYEMB_STITCH_CACHE: frozenset[str] | None = None


@dataclass
class VisualMaterial:
    status: str
    raster_path: str = ""
    renderer: str = ""
    family: str = ""
    error: str = ""
    extension: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_RENDERABLE and bool(self.raster_path)


def pyembroidery_stitch_extensions() -> frozenset[str]:
    """Auto-discover stitch-readable extensions from pyembroidery (no hardcoded subset)."""
    global _PYEMB_STITCH_CACHE
    if _PYEMB_STITCH_CACHE is not None:
        return _PYEMB_STITCH_CACHE
    if not _module("pyembroidery"):
        _PYEMB_STITCH_CACHE = frozenset()
        return _PYEMB_STITCH_CACHE
    try:
        import pyembroidery

        found: set[str] = set()
        for ft in pyembroidery.supported_formats():
            if not ft.get("reader"):
                continue
            cat = str(ft.get("category") or "")
            if cat in {"color", "debug", "image", "vector"}:
                continue
            names = [str(ft.get("extension") or "")]
            names.extend(str(x) for x in (ft.get("extensions") or ()))
            for name in names:
                ext = "." + name.lower().lstrip(".")
                if (
                    ext in _PYEMB_NON_STITCH
                    or "code" in ext
                    or ext[1:].isdigit()
                    or len(ext) < 3
                ):
                    continue
                found.add(ext)
        _PYEMB_STITCH_CACHE = frozenset(found)
    except Exception:
        _PYEMB_STITCH_CACHE = frozenset()
    return _PYEMB_STITCH_CACHE


def sniff_family(path: str) -> tuple[str, str]:
    """Return (family, extension). Magic bytes preferred; suffix fallback."""
    p = Path(path)
    ext = p.suffix.lower()
    magic = b""
    try:
        with open(path, "rb") as fh:
            magic = fh.read(24)
    except OSError:
        magic = b""
    if magic.startswith(b"%PDF"):
        return "pdf", ext or ".pdf"
    if magic.startswith(b"8BPS"):
        return "psd", ext or ".psd"
    if magic.startswith(b"\x89PNG"):
        return "raster", ext or ".png"
    if magic[:2] in (b"\xff\xd8",):
        return "raster", ext or ".jpg"
    if magic[:2] in (b"II", b"MM") and ext in {".tif", ".tiff", ""}:
        return "raster", ext or ".tif"
    if magic[:6] in (b"%!PS-A", b"%!PS-A") or magic.startswith(b"%!PS"):
        return "postscript", ext or ".eps"
    if magic.startswith(b"AC10"):
        return "cad", ext or ".dwg"
    head = magic.lstrip(b" \t\r\n")
    if head.startswith(b"0\nSECTION") or head.startswith(b"0\r\nSECTION"):
        return "cad", ext or ".dxf"
    if ext in RASTER_EXTS:
        return "raster", ext
    if ext == ".pdf":
        return "pdf", ext
    if ext == ".psd":
        return "psd", ext
    if ext in {".ai", ".eps"}:
        return "postscript", ext
    if ext == ".svg":
        return "svg", ext
    if ext in COREL_EXTS:
        return "cdr", ext
    stitch = pyembroidery_stitch_extensions()
    if ext in EMBROIDERY_EXTS | stitch | PROPRIETARY_EMBROIDERY_NO_READER:
        return "embroidery", ext
    if ext in HPGL_EXTS:
        return "hpgl", ext
    if ext in CAD_EXTS:
        return "cad", ext
    return "unknown", ext


def renderer_availability() -> dict[str, Any]:
    """Startup capability map — does not load heavy models."""
    from core.capability_check import probe_format_dependencies

    deps = probe_format_dependencies()
    pyemb = _module("pyembroidery")
    cairo = _module("cairosvg")
    inkscape = bool(shutil.which("inkscape"))
    soffice = bool(shutil.which("soffice") or shutil.which("libreoffice"))
    gs = bool(deps.get("ghostscript"))
    pymupdf = bool(deps.get("pymupdf"))
    ezdxf = _module("ezdxf")
    oda = bool(shutil.which("ODAFileConverter"))
    stitch_exts = sorted(pyembroidery_stitch_extensions()) if pyemb else []
    return {
        "raster": bool(deps.get("pillow") or deps.get("pyvips")),
        "psd": bool(deps.get("psd_tools") or deps.get("pillow")),
        "pdf": bool(pymupdf or deps.get("poppler") or gs),
        "ai_eps": bool(gs or pymupdf),
        "svg": cairo,
        "embroidery": pyemb,
        "embroidery_formats": stitch_exts,
        "emb": False,
        "ofm": False,
        "cdr": inkscape or soffice,
        "dxf": ezdxf,
        "dwg": oda,
        "dgn": False,
        "plt": bool(deps.get("pillow")),
        "hpgl": bool(deps.get("pillow")),
        "cad": ezdxf,
        "ghostscript": gs,
        "pymupdf": pymupdf,
        "psd_tools": bool(deps.get("psd_tools")),
        "cairosvg": cairo,
        "pyembroidery": pyemb,
        "ezdxf": ezdxf,
        "oda": oda,
        "inkscape": inkscape,
        "libreoffice": soffice,
        "pyvips": bool(deps.get("pyvips")),
        "pillow": bool(deps.get("pillow")),
    }


def _module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _cache_out(source: str, cache_dir: str, suffix: str) -> Path:
    root = Path(cache_dir) / "render_previews"
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(normalize_path(source).encode("utf-8")).hexdigest()[:20]
    return root / f"{key}_{suffix}.png"


def materialize_visual(
    source_path: str,
    *,
    cache_dir: str = "",
) -> VisualMaterial:
    """Produce a real raster preview path, or an honest failure status."""
    source = normalize_path(source_path)
    p = Path(source)
    family, ext = sniff_family(source) if p.exists() else ("unknown", p.suffix.lower())
    if not p.is_file():
        return VisualMaterial(
            status=STATUS_CORRUPT,
            family=family,
            extension=ext,
            error="file_missing",
        )
    try:
        size = int(p.stat().st_size)
    except OSError as exc:
        return VisualMaterial(
            status=STATUS_CORRUPT, family=family, extension=ext, error=str(exc)
        )
    if size <= 0:
        return VisualMaterial(
            status=STATUS_EMPTY, family=family, extension=ext, error="empty_file"
        )

    if not cache_dir:
        try:
            from core.settings import AppSettings

            cache_dir = str(AppSettings.load().cache_dir)
        except Exception:
            cache_dir = str(p.parent / ".v3_cache")

    if family == "raster":
        return VisualMaterial(
            status=STATUS_RENDERABLE,
            raster_path=source,
            renderer="native_raster",
            family=family,
            extension=ext,
        )

    if family in ("pdf", "postscript", "psd") or ext in {".pdf", ".ai", ".eps"} | ADOBE_LAYER_EXTS:
        from core.preview_renderer import render_preview_for_index

        rendered = render_preview_for_index(source, cache_dir)
        if rendered.success and rendered.image_path and Path(rendered.image_path).is_file():
            return VisualMaterial(
                status=STATUS_RENDERABLE,
                raster_path=str(rendered.image_path),
                renderer=rendered.renderer or "preview_renderer",
                family=family,
                extension=ext,
            )
        if rendered.unsupported_preview and (
            "gerekli" in (rendered.error or "").lower()
            or str(rendered.error or "").startswith("renderer_missing")
            or "dependency eksik" in (rendered.error or "").lower()
        ):
            return VisualMaterial(
                status=STATUS_MISSING_DEP,
                family=family,
                extension=ext,
                error=rendered.error,
                renderer=rendered.renderer or "",
            )
        return VisualMaterial(
            status=STATUS_RENDER_FAILED,
            family=family,
            extension=ext,
            error=rendered.error or "render_failed",
            renderer=rendered.renderer or "preview_renderer",
        )

    if family == "svg" or ext == ".svg":
        return _render_svg(source, cache_dir)

    if family == "embroidery":
        return _render_embroidery(source, cache_dir)

    if family == "cdr":
        return _render_cdr(source, cache_dir)

    if family in ("cad", "hpgl") or ext in CAD_EXTS | HPGL_EXTS:
        return _render_cad(source, cache_dir, ext)

    return VisualMaterial(
        status=STATUS_UNSUPPORTED,
        family=family or "unknown",
        extension=ext,
        error="unsupported_visual",
    )


def _render_svg(source: str, cache_dir: str) -> VisualMaterial:
    try:
        from plugins.base import PluginRuntime
        from plugins.vector.plugin import VectorPreviewPlugin

        plugin = VectorPreviewPlugin(
            PluginRuntime(cache_dir=cache_dir, thumbnail_max_edge=1024)
        )
        result = plugin.extract_preview(source)
        if result.success and result.artifact_path:
            return VisualMaterial(
                status=STATUS_RENDERABLE,
                raster_path=result.artifact_path,
                renderer="cairosvg",
                family="svg",
                extension=".svg",
            )
        status = STATUS_MISSING_DEP if result.status == "missing_dependency" else STATUS_RENDER_FAILED
        return VisualMaterial(
            status=status,
            family="svg",
            extension=".svg",
            error=result.error or result.status,
            renderer="cairosvg",
        )
    except Exception as exc:
        return VisualMaterial(
            status=STATUS_RENDERER_ERROR,
            family="svg",
            extension=".svg",
            error=str(exc),
        )


def _render_embroidery(source: str, cache_dir: str) -> VisualMaterial:
    ext = Path(source).suffix.lower()
    readable = pyembroidery_stitch_extensions()
    if ext in PROPRIETARY_EMBROIDERY_NO_READER or (ext and ext not in readable):
        return VisualMaterial(
            status=STATUS_UNSUPPORTED,
            family="embroidery",
            extension=ext,
            error=f"renderer_missing:no_stitch_reader:{ext}",
            renderer="",
        )
    if not _module("pyembroidery"):
        return VisualMaterial(
            status=STATUS_MISSING_DEP,
            family="embroidery",
            extension=ext,
            error="pyembroidery yok",
        )
    out = _cache_out(source, cache_dir, "emb")
    if out.is_file() and out.stat().st_size > 0:
        src_mtime = Path(source).stat().st_mtime
        if out.stat().st_mtime >= src_mtime - 1.0:
            return VisualMaterial(
                status=STATUS_RENDERABLE,
                raster_path=str(out),
                renderer="pyembroidery",
                family="embroidery",
                extension=ext,
            )
    try:
        import pyembroidery

        pattern = pyembroidery.read(source)
        if pattern is None:
            return VisualMaterial(
                status=STATUS_RENDER_FAILED,
                family="embroidery",
                extension=ext,
                error="stitch_read_empty",
                renderer="pyembroidery",
            )
        stitches = getattr(pattern, "stitches", None) or []
        if not stitches:
            return VisualMaterial(
                status=STATUS_RENDER_FAILED,
                family="embroidery",
                extension=ext,
                error="no_stitch_geometry",
                renderer="pyembroidery",
            )
        pyembroidery.write_png(pattern, str(out))
        if not out.is_file() or out.stat().st_size <= 0:
            return VisualMaterial(
                status=STATUS_RENDER_FAILED,
                family="embroidery",
                extension=ext,
                error="png_empty",
                renderer="pyembroidery",
            )
        return VisualMaterial(
            status=STATUS_RENDERABLE,
            raster_path=str(out),
            renderer="pyembroidery",
            family="embroidery",
            extension=ext,
        )
    except Exception as exc:
        return VisualMaterial(
            status=STATUS_RENDER_FAILED,
            family="embroidery",
            extension=ext,
            error=str(exc)[:300],
            renderer="pyembroidery",
        )


def _render_cdr(source: str, cache_dir: str) -> VisualMaterial:
    ext = ".cdr"
    inkscape = shutil.which("inkscape")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not inkscape and not soffice:
        return VisualMaterial(
            status=STATUS_UNSUPPORTED,
            family="cdr",
            extension=ext,
            error="renderer_missing:inkscape/libreoffice",
        )
    out = _cache_out(source, cache_dir, "cdr")
    try:
        import subprocess

        if inkscape:
            subprocess.run(
                [inkscape, source, "--export-type=png", f"--export-filename={out}"],
                check=True,
                capture_output=True,
                timeout=120,
            )
            renderer = "inkscape"
        else:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "png", "--outdir", str(out.parent), source],
                check=True,
                capture_output=True,
                timeout=120,
            )
            renderer = "libreoffice"
        if out.is_file() and out.stat().st_size > 0:
            return VisualMaterial(
                status=STATUS_RENDERABLE,
                raster_path=str(out),
                renderer=renderer,
                family="cdr",
                extension=ext,
            )
        return VisualMaterial(
            status=STATUS_RENDER_FAILED,
            family="cdr",
            extension=ext,
            error="cdr_export_empty",
            renderer=renderer,
        )
    except Exception as exc:
        return VisualMaterial(
            status=STATUS_RENDER_FAILED,
            family="cdr",
            extension=ext,
            error=str(exc)[:300],
        )


def _render_cad(source: str, cache_dir: str, ext: str) -> VisualMaterial:
    if ext in HPGL_EXTS:
        return _render_hpgl(source, cache_dir)
    if ext in {".dwg", ".dgn"}:
        oda = shutil.which("ODAFileConverter")
        if not oda:
            return VisualMaterial(
                status=STATUS_UNSUPPORTED,
                family="cad",
                extension=ext,
                error=f"renderer_missing:no_open_source_{ext.lstrip('.')}_converter",
            )
        return VisualMaterial(
            status=STATUS_UNSUPPORTED,
            family="cad",
            extension=ext,
            error="renderer_missing:oda_not_wired",
        )
    if ext == ".dxf" or ext == "":
        return _render_dxf(source, cache_dir)
    return VisualMaterial(
        status=STATUS_UNSUPPORTED,
        family="cad",
        extension=ext,
        error=f"renderer_missing:cad:{ext}",
    )


def _render_dxf(source: str, cache_dir: str) -> VisualMaterial:
    ext = Path(source).suffix.lower() or ".dxf"
    if not _module("ezdxf"):
        return VisualMaterial(
            status=STATUS_MISSING_DEP,
            family="cad",
            extension=ext,
            error="renderer_missing:ezdxf",
        )
    out = _cache_out(source, cache_dir, "dxf")
    if out.is_file() and out.stat().st_size > 0:
        src_mtime = Path(source).stat().st_mtime
        if out.stat().st_mtime >= src_mtime - 1.0:
            return VisualMaterial(
                status=STATUS_RENDERABLE,
                raster_path=str(out),
                renderer="ezdxf",
                family="cad",
                extension=ext,
            )
    try:
        import ezdxf
        from ezdxf.addons.drawing import Frontend, RenderContext
        from ezdxf.addons.drawing.layout import Page
        from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend

        try:
            doc = ezdxf.readfile(source)
        except ezdxf.DXFTypeError:
            return VisualMaterial(
                status=STATUS_UNSUPPORTED,
                family="cad",
                extension=ext,
                error="renderer_missing:dwg_binary_needs_oda",
                renderer="ezdxf",
            )
        except ezdxf.DXFStructureError as exc:
            return VisualMaterial(
                status=STATUS_CORRUPT,
                family="cad",
                extension=ext,
                error=str(exc)[:300],
                renderer="ezdxf",
            )
        backend = PyMuPdfBackend()
        Frontend(RenderContext(doc), backend).draw_layout(doc.modelspace())
        data = backend.get_pixmap_bytes(Page(0, 0), fmt="png", dpi=150)
        if not data or not data.startswith(b"\x89PNG"):
            return VisualMaterial(
                status=STATUS_RENDER_FAILED,
                family="cad",
                extension=ext,
                error="dxf_png_empty",
                renderer="ezdxf",
            )
        out.write_bytes(data)
        return VisualMaterial(
            status=STATUS_RENDERABLE,
            raster_path=str(out),
            renderer="ezdxf",
            family="cad",
            extension=ext,
        )
    except Exception as exc:
        msg = str(exc)[:300]
        status = STATUS_CORRUPT if "structure" in msg.lower() else STATUS_RENDER_FAILED
        return VisualMaterial(
            status=status,
            family="cad",
            extension=ext,
            error=msg,
            renderer="ezdxf",
        )


def _parse_hpgl_polylines(text: str) -> list[list[tuple[float, float]]]:
    text = re.sub(r"\x1b.[^;]*;?", " ", text)
    text = text.upper().replace("\n", "").replace("\r", "")
    tokens = re.findall(r"([A-Z]{2})([^;A-Z]*)", text)
    x = y = 0.0
    down = False
    polylines: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal current
        if len(current) >= 2:
            polylines.append(current)
        current = []

    for cmd, raw in tokens:
        nums = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", raw)]
        pairs = list(zip(nums[0::2], nums[1::2]))
        if cmd == "IN":
            flush()
            x = y = 0.0
            down = False
        elif cmd == "PU":
            flush()
            down = False
            if pairs:
                x, y = pairs[-1]
        elif cmd == "PD":
            if not current:
                current = [(x, y)]
            down = True
            if not pairs:
                continue
            for px, py in pairs:
                x, y = px, py
                current.append((x, y))
        elif cmd == "PA":
            for px, py in pairs:
                if down:
                    if not current:
                        current = [(x, y)]
                    x, y = px, py
                    current.append((x, y))
                else:
                    flush()
                    x, y = px, py
        elif cmd == "PR":
            for dx, dy in pairs:
                nx, ny = x + dx, y + dy
                if down:
                    if not current:
                        current = [(x, y)]
                    current.append((nx, ny))
                else:
                    flush()
                x, y = nx, ny
    flush()
    return polylines


def _render_hpgl(source: str, cache_dir: str) -> VisualMaterial:
    ext = Path(source).suffix.lower() or ".plt"
    out = _cache_out(source, cache_dir, "hpgl")
    if out.is_file() and out.stat().st_size > 0:
        src_mtime = Path(source).stat().st_mtime
        if out.stat().st_mtime >= src_mtime - 1.0:
            return VisualMaterial(
                status=STATUS_RENDERABLE,
                raster_path=str(out),
                renderer="hpgl",
                family="hpgl",
                extension=ext,
            )
    try:
        raw = Path(source).read_bytes()
        text = raw.decode("latin-1", errors="ignore")
        polylines = _parse_hpgl_polylines(text)
        pts = [p for line in polylines for p in line]
        if len(pts) < 2:
            return VisualMaterial(
                status=STATUS_RENDER_FAILED,
                family="hpgl",
                extension=ext,
                error="hpgl_no_drawable_paths",
                renderer="hpgl",
            )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w = max(maxx - minx, 1.0)
        h = max(maxy - miny, 1.0)
        edge = 1024.0
        pad = 32.0
        scale = (edge - 2 * pad) / max(w, h)
        img_w = max(64, int(w * scale + 2 * pad))
        img_h = max(64, int(h * scale + 2 * pad))
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for line in polylines:
            mapped = [
                (
                    pad + (px - minx) * scale,
                    img_h - (pad + (py - miny) * scale),
                )
                for px, py in line
            ]
            if len(mapped) >= 2:
                draw.line(mapped, fill=(20, 20, 20, 255), width=2)
        img.save(out, "PNG")
        if not out.is_file() or out.stat().st_size <= 0:
            return VisualMaterial(
                status=STATUS_RENDER_FAILED,
                family="hpgl",
                extension=ext,
                error="hpgl_png_empty",
                renderer="hpgl",
            )
        return VisualMaterial(
            status=STATUS_RENDERABLE,
            raster_path=str(out),
            renderer="hpgl",
            family="hpgl",
            extension=ext,
        )
    except Exception as exc:
        return VisualMaterial(
            status=STATUS_RENDER_FAILED,
            family="hpgl",
            extension=ext,
            error=str(exc)[:300],
            renderer="hpgl",
        )


class RendererRegistry:
    """Single entry: family → real renderer. Missing renderer stays unsupported."""

    def render(self, source: str, cache_dir: str = "") -> VisualMaterial:
        return materialize_visual(source, cache_dir=cache_dir)


def capability_labels() -> dict[str, str]:
    """Startup READY / UNAVAILABLE map for UI and logs."""
    a = renderer_availability()
    def _lab(ok: bool) -> str:
        return "READY" if ok else "UNAVAILABLE"
    return {
        "PSD": _lab(bool(a.get("psd"))),
        "PDF": _lab(bool(a.get("pdf"))),
        "AI/EPS": _lab(bool(a.get("ai_eps"))),
        "SVG": _lab(bool(a.get("svg"))),
        "Embroidery": _lab(bool(a.get("embroidery"))),
        "CDR": _lab(bool(a.get("cdr"))),
        "DXF": _lab(bool(a.get("dxf"))),
        "DWG": _lab(bool(a.get("dwg"))),
        "PLT/HPGL": _lab(bool(a.get("plt"))),
        "EMB": "UNAVAILABLE",
        "OFM": "UNAVAILABLE",
    }
