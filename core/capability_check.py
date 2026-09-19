"""AI / OCR / format bağımlılık kontrolleri — güvenli aç/kapa + detaylı sebep."""

from __future__ import annotations

import importlib.util
import shutil
import time
from typing import Any

AI_FALLBACK_MSG = (
    "AI modeli bulunamadı veya yüklenemedi. Arama hash + doku ile devam edecek."
)
OCR_FALLBACK_MSG = (
    "OCR kütüphanesi bulunamadı (EasyOCR veya Tesseract). OCR devre dışı bırakıldı."
)

# UI thread'i kilitlememek için probe sonuçlarını kısa süre önbellekle.
_OCR_PROBE_CACHE: tuple[float, bool, str] | None = None
_AI_PROBE_CACHE: tuple[float, bool, str] | None = None
_PROBE_TTL_S = 30.0


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def probe_ai_dependencies() -> tuple[bool, str]:
    """Geriye uyumlu: (ok, detailed_reason_or_empty)."""
    global _AI_PROBE_CACHE
    now = time.monotonic()
    if _AI_PROBE_CACHE and (now - _AI_PROBE_CACHE[0]) < _PROBE_TTL_S:
        return _AI_PROBE_CACHE[1], _AI_PROBE_CACHE[2]
    report = diagnose_ai_stack()
    if report["ok"]:
        _AI_PROBE_CACHE = (now, True, "")
        return True, ""
    _AI_PROBE_CACHE = (now, False, report["message"])
    return False, report["message"]


def diagnose_ai_stack() -> dict[str, Any]:
    """OpenCLIP / DINO / torch durumunu ayrıntılı raporla (ağır model yüklemez)."""
    detail: dict[str, Any] = {
        "ok": False,
        "torch": False,
        "open_clip": False,
        "torchvision": False,
        "cuda": False,
        "device": "cpu",
        "message": "",
        "reasons": [],
    }
    if not _module_available("torch"):
        detail["message"] = "OpenCLIP/DINO yüklenemedi\n\nSebep:\ntorch import failed"
        detail["reasons"].append("torch import failed")
        return detail
    detail["torch"] = True
    # torch.cuda.is_available() ilk çağrıda pahalı olabilir — sadece paket varlığını kontrol et
    detail["open_clip"] = _module_available("open_clip")
    detail["torchvision"] = _module_available("torchvision")
    try:
        # Hafif: find_spec yeterli; cuda bilgisini opsiyonel tut
        import torch

        detail["cuda"] = bool(torch.cuda.is_available())
        detail["device"] = "cuda" if detail["cuda"] else "cpu"
        detail["torch_version"] = str(getattr(torch, "__version__", ""))
    except Exception as exc:
        detail["message"] = f"OpenCLIP/DINO yüklenemedi\n\nSebep:\ntorch init failed: {exc}"
        detail["reasons"].append(f"torch init failed: {exc}")
        return detail

    if not detail["open_clip"]:
        detail["reasons"].append("open_clip package missing")
    if not detail["torchvision"]:
        detail["reasons"].append("torchvision package missing (DINOv2 preprocess)")

    if not detail["open_clip"] and not detail["torchvision"]:
        detail["message"] = (
            "OpenCLIP yüklenemedi\n\nSebep:\n"
            "open_clip and torchvision packages missing"
        )
        return detail

    detail["ok"] = True
    detail["message"] = ""
    return detail


def probe_ocr_dependencies() -> tuple[bool, str]:
    """EasyOCR/Tesseract paket varlığı — Reader oluşturmaz."""
    global _OCR_PROBE_CACHE
    now = time.monotonic()
    if _OCR_PROBE_CACHE and (now - _OCR_PROBE_CACHE[0]) < _PROBE_TTL_S:
        return _OCR_PROBE_CACHE[1], _OCR_PROBE_CACHE[2]
    # Ağır easyocr import'unu atla; sadece paket kurulumu
    easy = _module_available("easyocr")
    tess = _module_available("pytesseract")
    if easy or tess:
        _OCR_PROBE_CACHE = (now, True, "")
        return True, ""
    _OCR_PROBE_CACHE = (now, False, OCR_FALLBACK_MSG)
    return False, OCR_FALLBACK_MSG


def probe_format_dependencies() -> dict[str, Any]:
    """Harici araç ve kütüphane durumu."""
    gs = bool(
        shutil.which("gswin64c") or shutil.which("gswin32c") or shutil.which("gs")
    )
    poppler = bool(shutil.which("pdftoppm") or shutil.which("pdftocairo"))
    psd_tools = _module_available("psd_tools")
    pymupdf = _module_available("fitz")
    pyvips = _module_available("pyvips")
    pillow = _module_available("PIL")
    return {
        "ghostscript": gs,
        "poppler": poppler,
        "psd_tools": psd_tools,
        "pymupdf": pymupdf,
        "pyvips": pyvips,
        "pillow": pillow,
        "tif": pillow or pyvips,
        "psd": psd_tools or pillow,
        "ai_eps": gs or pymupdf,
        "pdf": pymupdf or poppler or gs,
        "cdr": bool(shutil.which("inkscape") or shutil.which("soffice") or shutil.which("libreoffice")),
        "svg": _module_available("cairosvg"),
        "embroidery": _module_available("pyembroidery"),
        "dxf": _module_available("ezdxf"),
        "dwg": bool(shutil.which("ODAFileConverter")),
        "plt": _module_available("PIL"),
        "ezdxf": _module_available("ezdxf"),
    }


def try_load_ai_extractor(settings) -> tuple[bool, str]:
    """DINO/CLIP yüklemeyi dene; başarısızsa detaylı sebep."""
    dep = diagnose_ai_stack()
    if not dep["ok"]:
        return False, dep["message"] or AI_FALLBACK_MSG
    try:
        from core.feature_extractor import FeatureExtractor

        t0 = time.perf_counter()
        ext = FeatureExtractor(
            use_ai=True,
            use_gpu=bool(getattr(settings, "use_gpu", False)),
            fast_hash_only=False,
        )
        load_s = time.perf_counter() - t0
        if not ext.ai_available:
            reasons = []
            if getattr(ext, "_dino_model", None) is None:
                reasons.append("DINOv2 failed to load into memory")
            if getattr(ext, "_clip_model", None) is None:
                reasons.append("OpenCLIP failed to load into memory")
            msg = (
                "OpenCLIP/DINO yüklenemedi\n\nSebep:\n"
                + ("\n".join(reasons) or "Model belleğe alınamadı")
            )
            return False, msg
        # smoke embed
        try:
            import numpy as np

            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            feats = ext.extract_from_array(
                arr, include_patches=False, deep_analysis=False
            )
            has_emb = bool(feats.dino_embedding or feats.clip_embedding)
            if not has_emb:
                return False, (
                    "OpenCLIP/DINO yüklendi ama embedding üretilemedi\n\n"
                    "Sebep:\nfirst embedding returned empty"
                )
        except Exception as exc:
            return False, (
                f"OpenCLIP/DINO yüklendi ama embedding testi başarısız\n\nSebep:\n{exc}"
            )
        provider = []
        if getattr(ext, "_clip_model", None) is not None:
            provider.append("OpenCLIP ViT-B-32")
        if getattr(ext, "_dino_model", None) is not None:
            provider.append("DINOv2 ViT-S/14")
        ok_msg = (
            f"AI Model OK ({', '.join(provider) or 'unknown'}) "
            f"device={ext.device} load={load_s:.1f}s"
        )
        return True, ok_msg
    except Exception as exc:
        return False, f"OpenCLIP/DINO yüklenemedi\n\nSebep:\n{exc}"
