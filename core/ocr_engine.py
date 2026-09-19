"""OCR motoru — opsiyonel EasyOCR / Tesseract. Varsayılan kapalı."""

from __future__ import annotations

from core.logger import setup_logger

logger = setup_logger(__name__)

try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps, ImageStat

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    try:
        from PIL import Image, ImageFilter, ImageOps, ImageStat
    except ImportError:
        Image = None  # type: ignore


def _deskew_pil(img):
    """Küçük açı kaymasını düzelt (kumaş / etiket OCR)."""
    try:
        import numpy as np

        gray = img.convert("L")
        arr = np.array(gray)
        coords = np.column_stack(np.where(arr < 180))
        if coords.size < 80:
            return img
        ys = coords[:, 0].astype(float)
        xs = coords[:, 1].astype(float)
        if xs.max() - xs.min() < 8:
            return img
        slope = np.polyfit(xs, ys, 1)[0]
        angle = float(np.degrees(np.arctan(slope)))
        if abs(angle) < 0.4 or abs(angle) > 18:
            return img
        return img.rotate(-angle, expand=True, fillcolor=255)
    except Exception:
        return img


def _prep_for_small_fabric_text(img):
    """Küçük / kumaş üzeri yazı: büyüt, kontrast, hafif keskinleştir."""
    w, h = img.size
    scale = 2 if max(w, h) < 900 else 1
    if scale > 1:
        img = img.resize((w * scale, h * scale))
    img = ImageOps.autocontrast(img.convert("L")).convert("RGB")
    img = img.filter(ImageFilter.SHARPEN)
    return img


def ocr_to_brand_entities(text: str, db_path: str | None = None) -> list[str]:
    """OCR metninden kanonik marka varlıkları."""
    try:
        from core.brand_aliases import expand_brand_terms

        return expand_brand_terms(text or "", db_path)
    except Exception:
        return []


class OCREngine:
    def __init__(self, enabled: bool = False, languages: list[str] | None = None):
        self.enabled = enabled
        self.languages = languages or ["tr", "en"]
        self._reader = None

        if not enabled:
            return

        if HAS_EASYOCR:
            try:
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=False,
                    verbose=False,
                )
                logger.info("EasyOCR yüklendi (tr+en)")
            except Exception as exc:
                logger.warning("EasyOCR başlatılamadı: %s", exc)
        elif HAS_TESSERACT:
            logger.info("Tesseract OCR kullanılacak (tur+eng)")
        else:
            logger.warning("OCR kütüphanesi bulunamadı — OCR devre dışı")
            self.enabled = False

    def extract_text(self, image_path: str) -> str:
        return str(self.extract_detailed(image_path).get("text") or "")

    def extract_detailed(self, image_path: str) -> dict:
        """TR+EN, güven, deskew. Index'e yazmaz."""
        empty = {
            "text": "",
            "confidence": 0.0,
            "languages": ",".join(self.languages),
            "brands": [],
        }
        if not self.enabled:
            return empty
        try:
            prepared = None
            if Image is not None:
                with Image.open(image_path) as raw:
                    img = raw.convert("RGB")
                    img = _deskew_pil(img)
                    prepared = _prep_for_small_fabric_text(img)
            text = ""
            conf = 0.0
            if self._reader is not None:
                source = prepared if prepared is not None else image_path
                if prepared is not None:
                    import numpy as np

                    results = self._reader.readtext(np.array(prepared), detail=1)
                else:
                    results = self._reader.readtext(source, detail=1)
                parts: list[str] = []
                scores: list[float] = []
                for item in results or []:
                    if len(item) >= 3:
                        parts.append(str(item[1]))
                        scores.append(float(item[2] or 0))
                    elif len(item) >= 2:
                        parts.append(str(item[1]))
                text = " ".join(parts).strip()
                conf = sum(scores) / len(scores) if scores else 0.0
            elif HAS_TESSERACT:
                lang = "tur+eng"
                if prepared is not None:
                    text = pytesseract.image_to_string(prepared, lang=lang).strip()
                    data = pytesseract.image_to_data(
                        prepared, lang=lang, output_type=pytesseract.Output.DICT
                    )
                    confs = [
                        float(c)
                        for c in (data.get("conf") or [])
                        if str(c) not in {"-1", ""}
                    ]
                    conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
                else:
                    with Image.open(image_path) as img:
                        text = pytesseract.image_to_string(img, lang=lang).strip()
            brands = ocr_to_brand_entities(text)
            return {
                "text": text,
                "confidence": round(float(conf or 0.0), 4),
                "languages": ",".join(self.languages),
                "brands": brands,
            }
        except Exception as exc:
            logger.warning("OCR hatası %s: %s", image_path, exc)
        return empty
